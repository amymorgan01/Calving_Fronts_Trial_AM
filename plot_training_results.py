from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import transforms
import cv2
from dataset_class import GlacierSegDataset  # Your dataset class

# Set paths
model_path = '/gws/nopw/j04/iecdt/amorgan/trained_models/seg_model_w_bkgrd_bs4_end.pth'
parent_dir = "/gws/nopw/j04/iecdt/amorgan/data_copy"  # Update with your data pathls
output_dir = '/home/users/amorgan/Calving_Fronts_and_Where_to_Find_Them/results'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

CLASS_NAMES = ["Background", "Rock", "Glacier", "Ocean&Ice"]
N_CLASSES = len(CLASS_NAMES)
COLORMAP = plt.cm.get_cmap('viridis', N_CLASSES)

# Function to visualize predictions
def visualize_prediction(image, mask, pred, class_names=None, save_path=None):
    """
    Visualize the original image, ground truth mask, and prediction
    Args:
        image: normalized tensor image
        mask: ground truth mask tensor
        pred: prediction mask tensor
        save_path: path to save the visualization
    """
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Prediction shape: {pred.shape}")
  
    # Convert tensors to numpy arrays
    image = image.squeeze().cpu().numpy()  # Remove channel dimension
    mask = mask.squeeze().cpu().numpy()
    pred = pred.squeeze().cpu().numpy()

    print(f"Image shape numpy arrays: {image.shape}")
    print(f"Mask shape numpy arrays: {mask.shape}")
    print(f"Prediction shape numpy arrays: {pred.shape}")
  
        # Handle image - remove batch dimension if present
    if len(image.shape) == 3 and image.shape[0] == 1:  # [1, H, W]
        image = image.squeeze(0)
    
    # Handle mask - it has shape [1, 4, 256, 256] or [4, 256, 256]
    # # First remove batch dimension if present
    # if len(mask.shape) == 4:  # [1, 4, H, W]
    #     mask = mask.squeeze(0)  # Now it's [4, H, W]
    
    # # Convert from one-hot encoding to class indices
    # if mask.shape[0] == 4:  # One-hot encoded mask [4, H, W]
    #     mask = np.argmax(mask, axis=0)  # Now it's [H, W]

    # Denormalize image (if needed)
    mean = 0.3047126829624176
    std = 0.32187142968177795
    image = image * std + mean
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # if mask.shape[0] == 4 and len(mask.shape) == 3:
    # # Convert from one-hot [4, H, W] back to class indices [H, W]
    #     mask_display = np.argmax(mask, axis=0)

    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    # For multi-class segmentation, use a colormap
    
    axes[1].imshow(mask, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot prediction
    # For prediction, we get class with highest probability
    axes[2].imshow(pred, cmap=COLORMAP, vmin=0, vmax=N_CLASSES-1)
    axes[2].set_title('Prediction')
    axes[2].axis('off')

    if class_names:
            import matplotlib.patches as mpatches
            patches = []
            for i, name in enumerate(class_names):
                color = COLORMAP(i / (N_CLASSES-1))
                patches.append(mpatches.Patch(color=color, label=name))
            fig.legend(handles=patches, loc='lower center', ncol=N_CLASSES)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def get_colormap_for_segmentation(n_classes=4):
    """
    Creates a colormap for segmentation visualization
    """
    cmap = plt.cm.get_cmap('viridis', n_classes)
    return cmap

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model = smp.Unet(
        encoder_name="resnet50", 
        encoder_weights="imagenet",
        in_channels=1,
        classes=4,  # Assuming 4 classes as in your training
    )
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    # Create test dataset
    test_dataset = GlacierSegDataset(mode='test', parent_dir=parent_dir, label_type="mask")
    
    # Generate predictions for a few samples
    num_samples = min(5, len(test_dataset))  # Visualize up to 5 samples
    
    print(f"Generating predictions for {num_samples} test samples...")
    
    with torch.no_grad():  # No gradient computation needed for inference
        for i in range(num_samples):
            # Get a sample
            image, mask = test_dataset[i]
            
            # Add batch dimension and move to device
            image = image.unsqueeze(0).to(device)  # [1, 1, H, W]
            mask = mask.to(device)  # [1, H, W]
            
            # Get prediction
            output = model(image)  # [1, num_classes, H, W]
            print(f"Output shape: {output.shape}")
            print(f"Example output: {output[0, :, :5, :5]}")  # Print a small part of the output
            
            # Convert output probabilities to class predictions
            pred = torch.argmax(output, dim=1)  # [1, H, W]
            print(f"Predicted shape: {pred.shape}")
            print(f"Min pred: {pred.min()}, Max pred: {pred.max()}")
            
            # Visualize
            save_path = os.path.join(output_dir, f'prediction_{i}.png')
            visualize_prediction(image[0], mask, pred[0], save_path)
    
    print(f"Visualizations saved to {output_dir}")
    
    # Calculate metrics on the test set (optional)
    print("Evaluating model on test set...")
    iou_scores = []
    pixel_accuracies = []
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            image, mask = test_dataset[i]
            image = image.unsqueeze(0).to(device)
            mask = mask.squeeze(0).to(device).long()  # [H, W]
            
            output = model(image)  # [1, num_classes, H, W]
            pred = torch.argmax(output, dim=1).squeeze(0)  # [H, W]
            
            # Calculate IoU for each class and average
            intersection = torch.zeros(4).to(device)
            union = torch.zeros(4).to(device)
            
            for cls in range(4):
                intersection[cls] = ((pred == cls) & (mask == cls)).sum().float()
                union[cls] = ((pred == cls) | (mask == cls)).sum().float()
            
            iou = (intersection / (union + 1e-8)).mean().item()
            iou_scores.append(iou)
            
            # Calculate pixel accuracy
            accuracy = (pred == mask).sum().float() / (mask.numel())
            pixel_accuracies.append(accuracy.item())
            
            if i % 10 == 0:
                print(f"Processed {i}/{len(test_dataset)} test samples")
    
    # Print average metrics
    avg_iou = sum(iou_scores) / len(iou_scores)
    avg_pixel_accuracy = sum(pixel_accuracies) / len(pixel_accuracies)
    
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Pixel Accuracy: {avg_pixel_accuracy:.4f}")
    
    # Plot metrics distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(iou_scores, bins=20, alpha=0.7)
    plt.axvline(avg_iou, color='r', linestyle='--', label=f'Mean IoU: {avg_iou:.4f}')
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('IoU Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(pixel_accuracies, bins=20, alpha=0.7)
    plt.axvline(avg_pixel_accuracy, color='r', linestyle='--', 
                label=f'Mean Accuracy: {avg_pixel_accuracy:.4f}')
    plt.xlabel('Pixel Accuracy')
    plt.ylabel('Frequency')
    plt.title('Pixel Accuracy Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'))
    print(f"Metrics plot saved to {os.path.join(output_dir, 'metrics_distribution.png')}")

if __name__ == "__main__":
    main()