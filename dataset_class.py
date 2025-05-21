from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import segmentation_models_pytorch as smp
import tqdm
import numpy as np
import albumentations as A
import pandas as pd


class GlacierSegDataset(Dataset):
    """"
   
    This class loads a dataset of SAR images and their corresponding segmentation masks.

    Each SAR image has:
    - A corresponding segmentation mask (ice vs. background)
    - A corresponding front mask (glacier front detection)

    Args:
        - mode (str): 'train', 'val', or 'test'
        - parent_dir (str): Root directory containing 'sar_images', 'zones', and 'fronts' folders.
        - label_type (str): 'mask' or 'front'
        - bg_threshold (float): Maximum allowed background percentage (0-1.0), defaults to None (no filtering)
        - /gws/nopw/j04/iecdt/amorgan/data_copy/train_cleaned_patches.csv is the csv file with filenames of non-blank images

    """

    def __init__(self, mode, parent_dir, label_type = "mask", filtered_csv_path=None):
        print(f"Initializing GlacierSegDataset in {mode} mode...")
        self.label_type = label_type #choose between 'mask' and 'front'
        
        self.mode = mode

        self.image_dir = os.path.join(parent_dir, "sar_images", mode)
        self.mask_dir = os.path.join(parent_dir, "zones", mode)
        self.front_dir = os.path.join(parent_dir, "fronts", mode)

        # Get all available files first
        all_images = sorted(os.listdir(self.image_dir))
        all_masks = sorted(os.listdir(self.mask_dir))
        all_fronts = sorted(os.listdir(self.front_dir))

        self.images = all_images
        self.masks = all_masks  
        self.fronts = all_fronts

        print(f"Number of images: {len(self.images)}, masks: {len(self.masks)}, fronts: {len(self.fronts)}")
        assert len(self.images) == len(self.masks) == len(self.fronts), "Mismatch in dataset size!"

         # Load filtered filenames from CSV if specified
        if filtered_csv_path is not None and os.path.exists(filtered_csv_path):
            print(f"Loading filtered filenames from {filtered_csv_path}...")
            self.images, self.masks, self.fronts = self._load_filtered_filenames_from_csv(
                filtered_csv_path, all_images, all_masks, all_fronts
            )
            print(f"After filtering - images: {len(self.images)}, masks: {len(self.masks)}, fronts: {len(self.fronts)}")
        else:
            print(f"No CSV file specified or found. Using all available images.")
            self.images = all_images
            self.masks = all_masks
            self.fronts = all_fronts

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


    def _load_filtered_filenames_from_csv(self, csv_path, all_images, all_masks, all_fronts):
        """
        Loads filtered filenames from a CSV file.
        
        The CSV should contain at least one of these columns:
        - 'image_name': Filenames of filtered images
        - 'is_bg_heavy': boolean indicating if the image is background dominated
        
        Args:
            csv_path (str): Path to the CSV file
            all_images (list): List of all image filenames
            all_masks (list): List of all mask filenames
            all_fronts (list): List of all front filenames
            
        Returns:
            tuple: (filtered_images, filtered_masks, filtered_fronts)
        """
        try:
            # Load the CSV file
            df = pd.read_csv(csv_path)
            
            if 'is_bg_heavy' in df.columns:
                #keep only images not dominated by background
                df = df[df['is_bg_heavy'] == False]
                print(f"After is_bg_heavy filtering: {len(df)} rows remaining")
            
            # Check which columns are present
            has_image_col = 'image_name' in df.columns
            has_mask_col = 'mask_name' in df.columns
            has_front_col = 'front_name' in df.columns
            
            # Initialize filtered lists
            filtered_images = []
            filtered_masks = []
            filtered_fronts = []
            
            if has_image_col:
                # Filter based on image names in CSV
                filtered_images = df['image_name'].tolist()
                
                # If mask or front columns are not in CSV, match them by index
                if not has_mask_col or not has_front_col:
                    # Create dictionaries to map image names to corresponding mask/front names
                    image_to_mask = {img: mask for img, mask in zip(all_images, all_masks)}
                    image_to_front = {img: front for img, front in zip(all_images, all_fronts)}
                    
                    if not has_mask_col:
                        # For each filtered image, find corresponding mask
                        filtered_masks = [image_to_mask[img] for img in filtered_images if img in image_to_mask]
                    else:
                        filtered_masks = df['mask_name'].tolist()
                        
                    if not has_front_col:
                        # For each filtered image, find corresponding front
                        filtered_fronts = [image_to_front[img] for img in filtered_images if img in image_to_front]
                    else:
                        filtered_fronts = df['front_name'].tolist()
                else:
                    # If all columns are present, use them directly
                    filtered_masks = df['mask_name'].tolist()
                    filtered_fronts = df['front_name'].tolist()
            else:
                # If no image column, check for mask column
                if has_mask_col:
                    filtered_masks = df['mask_name'].tolist()
                    
                    # Map masks to corresponding images and fronts
                    mask_to_image = {mask: img for img, mask in zip(all_images, all_masks)}
                    mask_to_front = {mask: front for mask, front in zip(all_masks, all_fronts)}
                    
                    filtered_images = [mask_to_image[mask] for mask in filtered_masks if mask in mask_to_image]
                    
                    if not has_front_col:
                        filtered_fronts = [mask_to_front[mask] for mask in filtered_masks if mask in mask_to_front]
                    else:
                        filtered_fronts = df['front_name'].tolist()
                elif has_front_col:
                    # If only front column exists
                    filtered_fronts = df['front_name'].tolist()
                    
                    # Map fronts to corresponding images and masks
                    front_to_image = {front: img for img, front in zip(all_images, all_fronts)}
                    front_to_mask = {front: mask for mask, front in zip(all_masks, all_fronts)}
                    
                    filtered_images = [front_to_image[front] for front in filtered_fronts if front in front_to_image]
                    filtered_masks = [front_to_mask[front] for front in filtered_fronts if front in front_to_mask]
                else:
                    raise ValueError("CSV must contain at least one of 'image_name', 'mask_name', or 'front_name' columns")
            
            # Verify we have matching counts
            if len(filtered_images) != len(filtered_masks) or len(filtered_images) != len(filtered_fronts):
                print(f"Warning: Mismatch in filtered dataset counts - images: {len(filtered_images)}, masks: {len(filtered_masks)}, fronts: {len(filtered_fronts)}")
                
                # Find the smallest common set
                common_size = min(len(filtered_images), len(filtered_masks), len(filtered_fronts))
                filtered_images = filtered_images[:common_size]
                filtered_masks = filtered_masks[:common_size]
                filtered_fronts = filtered_fronts[:common_size]
                
                print(f"Truncated to common size: {common_size}")
            
            print(f"Loaded {len(filtered_images)} filtered images from CSV")
            return filtered_images, filtered_masks, filtered_fronts
            
        except Exception as e:
            print(f"Error loading filtered filenames from CSV: {e}")
            print("Using all images without filtering.")
            return all_images, all_masks, all_fronts
    

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
    