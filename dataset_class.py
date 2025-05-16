from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp
import tqdm
import numpy as np
import albumentations as A

class GlacierSegDataset(Dataset):
    """"
   
    This class loads a dataset of SAR images and their corresponding segmentation masks.

    Each SAR image has:
    - A corresponding segmentation mask (ice vs. background)
    - A corresponding front mask (glacier front detection)

    Args:
        - mode (str): 'train', 'val', or 'test'
        - parent_dir (str): Root directory containing 'sar_images', 'zones', and 'fronts' folders.
    """

    def __init__(self, mode, parent_dir, label_type = "mask"):
        print(f"Initializing GlacierSegDataset in {mode} mode...")
        self.label_type = label_type #choose between 'mask' and 'front'

        self.image_dir = os.path.join(parent_dir, "sar_images", mode)
        self.mask_dir = os.path.join(parent_dir, "zones", mode)
        self.front_dir = os.path.join(parent_dir, "fronts", mode)

        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.fronts = sorted(os.listdir(self.front_dir))
        print(f"Number of images: {len(self.images)}, masks: {len(self.masks)}, fronts: {len(self.fronts)}")
        assert len(self.images) == len(self.masks) == len(self.fronts), "Mismatch in dataset size!"

        # Use Albumentations for consistent transformations
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), p=1.0),
            # A.OneOf([
            #     A.GaussNoise(p=0.5),
            #     A.RandomBrightnessContrast(p=0.5),
            # ], p=0.3),
        ])
        
        # Separate normalization for image only (after joint transformations)
        self.normalize = A.Normalize(mean=0.3047126829624176, std=0.32187142968177795)
        self.thresholds = [0.0, 0.1, 0.3, 0.7, 1.1]
        # Awkward hard coded greyscale values for the label
        # add numpy.isclose ensure all values are within a certain range
        # self.l0 = np.isclose(0.0, 0.0, atol=1e-5)
        # self.l1 = np.isclose(0.2509804, 0.2509804, atol=1e-5)
        # self.l2 = np.isclose(0.49803922, 0.49803922, atol=1e-5)
        # self.l3 = np.isclose(0.99607843, 0.99607843, atol=1e-5)

    def convert_label(self, label_tensor):
        """
        Convert grayscale label tensor to class indices based on thresholds.
        
        Args:
            label_tensor: Tensor with values in range [0, 1]
            
        Returns:
            Tensor with class indices (0, 1, 2, 3)
        """
        classes = torch.zeros_like(label_tensor, dtype=torch.long)
        # Assign classes based on thresholds
        for i, threshold in enumerate(self.thresholds[:-1]):
            mask = torch.logical_and(label_tensor >= threshold, label_tensor < self.thresholds[i+1])
            classes[mask] = i
        
        return classes


    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """"
        Loads a SAR image,  and its corresponding label (either mask or front)
        """
        image_name = self.images[idx]
        
        # Load image with OpenCV and convert to PIL Image
        image_path = os.path.join(self.image_dir, image_name)
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # image_pil = Image.fromarray(image_np)

        # Choose label type dynamically
        if self.label_type == "mask":
            label_path = os.path.join(self.mask_dir, self.masks[idx])
            
        elif self.label_type == "front":
            label_path = os.path.join(self.front_dir, self.fronts[idx])
            
        else:
            raise ValueError("Invalid label_type! Choose 'mask' or 'front'.")
        
       # Load label with OpenCV and convert to PIL Image
        label_np = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        label_pil = Image.fromarray(label_np)
        
        # image_np = np.array(image_pil)
        label_np = np.array(label_pil)
        
        # Apply same spatial transformations to both
        transformed = self.transform(image=image_np, mask=label_np)
        image_transformed, label_transformed = transformed['image'], transformed['mask']
        
        # Apply normalization only to image
        image_normalized = self.normalize(image=image_transformed)['image']
        image_tensor = torch.from_numpy(image_normalized).float().unsqueeze(0)
        label_tensor = torch.from_numpy(label_transformed).float().unsqueeze(0)

        # Normalize label to [0, 1] before thresholding
        label_tensor = label_tensor / 255.0

        # Convert label to class indices
        label_classes = self.convert_label(label_tensor)

        # # Debug prints
        # if idx < 10:
        #     print(f"Sample {idx} mask min/max: {label_np.min()}, {label_np.max()}")
        #     print(f"Sample {idx} mask unique: {np.unique(label_np)}")
        #     print(f"Sample {idx} label unique values: {torch.unique(label_classes)}")

        return image_tensor, label_classes.squeeze(0)


        # # Apply transformations to label
        # label = self.label_transform(label_pil)

        # # convert our silly greyscale to beefy one-hot encoding
        # label_copy = torch.zeros(label.shape)
        # label_copy[label == self.l0] = 0
        # label_copy[label == self.l1] = 1
        # label_copy[label == self.l2] = 2
        # label_copy[label == self.l3] = 3
        # label_copy = label_copy.long()
        # label_copy = torch.nn.functional.one_hot(label_copy, num_classes=4).permute(0, 3, 1, 2)

        # return image, label_copy # Returns (1, 4, 256, 256) tensors
    