import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import cv2
from torchvision import transforms
from dataset_class import GlacierSegDataset

# Function to debug dataset by visualizing image-mask pairs
def debug_dataset(dataset, num_samples=5):
    """
    Visualize the first num_samples from the dataset to check if images and masks match
    
    Args:
        dataset: GlacierSegDataset instance
        num_samples: Number of samples to visualize
    """
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples*5))
    
    for i in range(min(num_samples, len(dataset))):
        # Get a sample from the dataset
        image, mask = dataset[i]
        
        # Convert tensors to numpy arrays for visualization
        image_np = image.squeeze().numpy()  # Remove channel dim and convert to numpy
        mask_np = mask.squeeze().numpy()
        
        # For normalized images, we need to unnormalize for better visualization
        # Assuming your normalization was with mean=0.3047, std=0.3219
        image_np = image_np * 0.32187142968177795 + 0.3047126829624176
        
        # Clip values to be between 0 and 1 after unnormalization
        image_np = np.clip(image_np, 0, 1)
        
        # Display the image and mask
        axs[i, 0].imshow(image_np, cmap='gray')
        axs[i, 0].set_title(f"Image {i}: {dataset.images[i]}")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(mask_np, cmap='gray')
        axs[i, 1].set_title(f"Mask {i}: {dataset.masks[i]}")
        axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def check_filename_matching(dataset):
    """
    Check if the filenames for images and masks match
    
    Args:
        dataset: GlacierSegDataset instance
    
    Returns:
        List of indices where filenames don't match
    """
    mismatches = []
    
    for i in range(len(dataset.images)):
        # Extract base filenames (without extension)
        img_base = os.path.splitext(dataset.images[i])[0]
        mask_base = os.path.splitext(dataset.masks[i])[0]
        
        # Check if base filenames match
        if img_base != mask_base:
            mismatches.append((i, dataset.images[i], dataset.masks[i]))
    
    return mismatches


def check_direct_file_loading(image_dir, mask_dir, num_samples=5):
    """
    Directly load and visualize image-mask pairs to check if they match
    
    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        num_samples: Number of samples to visualize
    """
    # Get sorted lists of files
    images = sorted(os.listdir(image_dir))
    masks = sorted(os.listdir(mask_dir))
    
    fig, axs = plt.subplots(num_samples, 2, figsize=(10, num_samples*5))
    
    for i in range(min(num_samples, len(images))):
        # Load image with OpenCV
        image_path = os.path.join(image_dir, images[i])
        image_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask with OpenCV
        mask_path = os.path.join(mask_dir, masks[i])
        mask_np = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Display the image and mask
        axs[i, 0].imshow(image_np, cmap='gray')
        axs[i, 0].set_title(f"Image {i}: {images[i]}")
        axs[i, 0].axis('off')
        
        axs[i, 1].imshow(mask_np, cmap='gray')
        axs[i, 1].set_title(f"Mask {i}: {masks[i]}")
        axs[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

parent_dir = "/gws/nopw/j04/iecdt/amorgan/data_copy"  # Update with your data pathls

train_dataset = GlacierSegDataset(mode="train", parent_dir=parent_dir, label_type="mask")
val_dataset = GlacierSegDataset(mode="val", parent_dir=parent_dir, label_type="mask")
test_dataset = GlacierSegDataset(mode="test", parent_dir=parent_dir, label_type="mask")



print("Checking training dataset...")
debug_dataset(train_dataset)
mismatches = check_filename_matching(train_dataset)
if mismatches:
    print(f"Found {len(mismatches)} mismatches in training set!")
    for idx, img, mask in mismatches[:10]:  # Print first 10 mismatches
        print(f"Index {idx}: Image {img} vs Mask {mask}")
else:
    print("No filename mismatches found in training set.")

# Check validation dataset
print("\nChecking validation dataset...")
debug_dataset(val_dataset)
mismatches = check_filename_matching(val_dataset)
if mismatches:
    print(f"Found {len(mismatches)} mismatches in validation set!")
    for idx, img, mask in mismatches[:10]:  # Print first 10 mismatches
        print(f"Index {idx}: Image {img} vs Mask {mask}")
else:
    print("No filename mismatches found in validation set.")



# Check test dataset
print("\nChecking test dataset...")
debug_dataset(test_dataset)
mismatches = check_filename_matching(test_dataset)
if mismatches:
    print(f"Found {len(mismatches)} mismatches in test set!")
    for idx, img, mask in mismatches[:10]:  # Print first 10 mismatches
        print(f"Index {idx}: Image {img} vs Mask {mask}")
else:
    print("No filename mismatches found in test set.")
