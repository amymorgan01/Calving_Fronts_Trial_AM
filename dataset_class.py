from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torchvision
import segmentation_models_pytorch as smp
import tqdm
import torch.nn as nn
import torch.optim as optim

class GlacierSegDataset(Dataset):
    """"
   
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

        self.transform = transforms.Compose([
            transforms.ToTensor(), #converts HxW numpy array to 1xHxW Pytorch tensor
            transforms.Normalize(mean=0.3047126829624176, std=0.32187142968177795) #mean and sd of the data

         ])


    def __len__(self):
        """"
        Returns the number of samples in the dataset
        """
        return len(self.images)
    
    def __getitem__(self, idx):
        """"
        Loads a SAR image,  and its corresponding label (either mask or front)
        """

        image_name = self.images[idx]
        #print(f"Loading image: {image_name}")
        image = cv2.imread(os.path.join(self.image_dir, image_name), cv2.IMREAD_GRAYSCALE)


        # Choose label type dynamically
        if self.label_type == "mask":
            label_path = os.path.join(self.mask_dir, self.masks[idx])
            
        elif self.label_type == "front":
            label_path = os.path.join(self.front_dir, self.fronts[idx])
            
        else:
            raise ValueError("Invalid label_type! Choose 'mask' or 'front'.")
        
            
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        image = self.transform(image)
        label = torchvision.transforms.ToTensor()(label)  # No normalization for label
    # image_path = os.path.join(self.image_dir, self.images[idx])
        # mask_path = os.path.join(self.mask_dir, self.masks[idx])
        # front_path = os.path.join(self.front_dir, self.fronts[idx])

        # image = Image.open(image_path).convert("L")
        # mask = Image.open(mask_path).convert("L")
        # front = Image.open(front_path).convert("L")

        # # Apply transformations
        # image = self.transform(image)
        # mask = self.transform(mask)
        # front = self.transform(front)

        # print(f"Loading image: {image}, mask: {mask}, front: {front}")

        # print(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Front shape: {front.shape}")

        # Print tensor shapes after transformation
        #print(f"Transformed image shape: {image.shape}, Transformed label shape: {label.shape}")


        return image, label # Returns (1, 256, 256) tensors
    