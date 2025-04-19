"""
Dataset classes for loading and processing breast MRI data.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset
from collections import Counter


class BreastDataset(Dataset):
    """
    Dataset for loading paired pre-treatment and post-treatment MRI scans with segmentation masks.
    
    This dataset handles loading of the MRI volumes, their corresponding segmentation masks,
    and extracts sub-regions based on the segmentation masks.
    """
    def __init__(self, pre_dir, post_dir, csv_path, 
                 pre_seg_dir, post_seg_dir,
                 target_shape=(150, 150, 150), transform=None):
        """
        Initialize the dataset.
        
        Args:
            pre_dir (str): Directory containing pre-treatment MRI scans
            post_dir (str): Directory containing post-treatment MRI scans
            csv_path (str): Path to CSV file with patient metadata (ID, PCR status, time difference)
            pre_seg_dir (str): Directory containing pre-treatment segmentation masks
            post_seg_dir (str): Directory containing post-treatment segmentation masks
            target_shape (tuple): Target shape for resizing the volumes (default: (150, 150, 150))
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.pre_dir = pre_dir
        self.post_dir = post_dir
        self.pre_seg_dir = pre_seg_dir
        self.post_seg_dir = post_seg_dir
        self.target_shape = target_shape
        self.transform = transform
        self.data = self._load_data(csv_path)
        self.file_pairs = self._get_file_pairs()

    def _load_data(self, csv_path):
        """
        Load patient data from CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            dict: Dictionary mapping patient IDs to their metadata
        """
        df = pd.read_csv(csv_path)
        data = {str(row['ID']): {'PCR': row['PCR'], 'Diff': row['Diff']} 
                for _, row in df.iterrows() if row['PCR'] in [0, 1]}
        print(f"Loaded {len(data)} valid samples from CSV file")
        return data

    def _get_file_pairs(self):
        """
        Match pre-treatment and post-treatment files for each patient.
        
        Returns:
            list: List of tuples (patient_id, pre_file, post_file)
        """
        pre_files = {f.split('_')[0]: f for f in os.listdir(self.pre_dir) if f.endswith('.nii.gz')}
        post_files = {f.split('_')[0]: f for f in os.listdir(self.post_dir) if f.endswith('.nii.gz')}

        file_pairs = []
        for patient_id in set(pre_files.keys()) & set(post_files.keys()):
            if str(patient_id) in self.data:
                file_pairs.append((patient_id, pre_files[patient_id], post_files[patient_id]))
        print(f"Found {len(file_pairs)} valid file pairs")
        return file_pairs

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.file_pairs)

    def _get_seg_path(self, img_file):
        """
        Get the path to segmentation file corresponding to the image file.
        
        Args:
            img_file (str): Image filename
            
        Returns:
            str: Segmentation filename
        """
        return img_file.replace('.nii.gz', '_RadiomicsClass.nii.gz')

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (pre_treatment_img, post_treatment_img, 
                   pre_treatment_regions, post_treatment_regions,
                   time_difference, label)
        """
        patient_id, pre_file, post_file = self.file_pairs[idx]

        # Load original images
        pre_path = os.path.join(self.pre_dir, pre_file)
        post_path = os.path.join(self.post_dir, post_file)
        pre_data = self.load_and_preprocess(pre_path)
        post_data = self.load_and_preprocess(post_path)

        # Load segmentation masks
        pre_seg_file = self._get_seg_path(pre_file)
        post_seg_file = self._get_seg_path(post_file)
        pre_seg_path = os.path.join(self.pre_seg_dir, pre_seg_file)
        post_seg_path = os.path.join(self.post_seg_dir, post_seg_file)

        # Load and process segmentation masks
        pre_seg = self.load_and_preprocess(pre_seg_path)
        post_seg = self.load_and_preprocess(post_seg_path)

        # Extract each sub-region
        pre_regions = []
        post_regions = []
        unique_regions = np.unique(pre_seg)[1:]  # Exclude background (0)

        for region_id in unique_regions:
            pre_mask = (pre_seg == region_id)
            post_mask = (post_seg == region_id)

            # Apply mask to original image
            pre_region = pre_data * pre_mask
            post_region = post_data * post_mask

            pre_regions.append(pre_region)
            post_regions.append(post_region)

        # Pad with zeros if number of regions is less than the maximum expected
        max_regions = 3  # Assuming maximum 3 sub-regions
        while len(pre_regions) < max_regions:
            pre_regions.append(np.zeros_like(pre_data))
            post_regions.append(np.zeros_like(post_data))

        label = int(self.data[patient_id]['PCR'])
        diff_time = self.data[patient_id]['Diff']

        if self.transform:
            pre_data = self.transform(pre_data)
            post_data = self.transform(post_data)
            for i in range(len(pre_regions)):
                pre_regions[i] = self.transform(pre_regions[i])
                post_regions[i] = self.transform(post_regions[i])

        return (torch.from_numpy(pre_data).float(),
                torch.from_numpy(post_data).float(),
                torch.from_numpy(np.stack(pre_regions)).float(),
                torch.from_numpy(np.stack(post_regions)).float(),
                torch.tensor(diff_time, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long))

    def load_and_preprocess(self, file_path):
        """
        Load and preprocess a NIfTI file.
        
        Args:
            file_path (str): Path to NIfTI file
            
        Returns:
            np.ndarray: Preprocessed volume with added channel dimension
        """
        nifti_img = nib.load(file_path)
        data = nifti_img.get_fdata()
        data = np.resize(data, self.target_shape)
        return np.expand_dims(data, axis=0)  # Add channel dimension

    def oversample_minority_class(self):
        """
        Oversample the minority class to balance the dataset.
        
        Returns:
            list: Oversampled file pairs
        """
        label_counts = Counter([data['PCR'] for data in self.data.values()])
        max_count = max(label_counts.values())
        oversampled_pairs = self.file_pairs.copy()

        for patient_id, pre_file, post_file in self.file_pairs:
            label = self.data[patient_id]['PCR']
            count = label_counts[label]
            oversampling_factor = max_count // count - 1
            oversampled_pairs.extend([(patient_id, pre_file, post_file)] * oversampling_factor)

        random.shuffle(oversampled_pairs)
        return oversampled_pairs


class CustomAugmentation:
    """
    Custom data augmentation for 3D medical images.
    
    This class provides basic augmentations like flipping and rotation
    that are suitable for 3D medical images.
    """
    def __init__(self, flip_prob=0.5, rotate_prob=0.5, max_rotate=10):
        """
        Initialize the augmentation.
        
        Args:
            flip_prob (float): Probability of flipping the image
            rotate_prob (float): Probability of rotating the image
            max_rotate (float): Maximum rotation angle in degrees
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.max_rotate = max_rotate

    def __call__(self, img):
        """
        Apply augmentation to an image.
        
        Args:
            img (np.ndarray): Input image
            
        Returns:
            np.ndarray: Augmented image
        """
        if random.random() < self.flip_prob:
            img = np.flip(img, axis=2)
        if random.random() < self.rotate_prob:
            angle = random.uniform(-self.max_rotate, self.max_rotate)
            img = self.rotate_3d(img, angle)
        return np.ascontiguousarray(img)

    def rotate_3d(self, img, angle):
        """
        Rotate a 3D image around z-axis.
        
        Args:
            img (np.ndarray): Input image
            angle (float): Rotation angle in degrees
            
        Returns:
            np.ndarray: Rotated image
        """
        # Placeholder implementation
        # For proper implementation, consider using scipy.ndimage.rotate
        return img