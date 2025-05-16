from PIL import Image
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from dataset_class import GlacierSegDataset
import segmentation_models_pytorch as smp
import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gc
import psutil
import numpy as np
# wandb.login()

def visualize_prediction(image, mask, pred, save_path=None):
    """
    Visualize the original image, ground truth mask, and prediction
    Args:
        image: normalized tensor image
        mask: ground truth mask tensor
        pred: prediction mask tensor
        save_path: path to save the visualization
    """
    # Convert tensors to numpy arrays
    image = image.squeeze().cpu().numpy()  # Remove channel dimension
    mask = mask.squeeze().cpu().numpy()
    
    # Denormalize image (if needed)
    mean = 0.3047126829624176
    std = 0.32187142968177795
    image = image * std + mean
    # Clip to [0, 1] range
    image = np.clip(image, 0, 1)
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Plot ground truth mask
    # For multi-class segmentation, use a colormap
    axes[1].imshow(mask, cmap='viridis')
    axes[1].set_title('Ground Truth Mask')
    axes[1].axis('off')
    
    # Plot prediction
    # For prediction, we get class with highest probability
    axes[2].imshow(pred, cmap='viridis')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    plt.close()
    


# A class to handle the segmentation loss
class IoULoss(nn.Module):
    """
    A class for the Jaccard index loss function.
    """
    def __init__(self, smooth=1e-6):
        """
        Initializes the Jaccard index loss function.

        Args:
            smooth (float): A small constant to avoid division by zero.
        """
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Computes the Jaccard index loss.

        Args:
            y_pred (torch.Tensor): The predicted output.
            y_true (torch.Tensor): The ground truth labels.

        Returns:
            torch.Tensor: The computed Jaccard index loss.
        """
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection
        return 1 - (intersection + self.smooth) / (union + self.smooth)
    

# Check memory status
def print_memory_stats():
    process = psutil.Process(os.getpid())
    print(f"Process memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB")
            print(f"GPU {i} memory cached: {torch.cuda.memory_cached(i) / (1024 * 1024):.2f} MB")

#Check memory before and after training
print_memory_stats()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    print("Starting training for one epoch...")
    print("Memory before training:")
    print_memory_stats()

    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (images, masks) in pbar:
        
        images = images.to(device)  # Shape: [batch_size, 1, height, width]
        masks = masks.to(device).squeeze(1).long()  # Shape: [batch_size, height, width]
        
        optimizer.zero_grad()
        outputs = model(images)
        #print(f"for batch {i}, input shape: {images.shape} outputs shape: {outputs.shape}, masks shape: {masks.shape}")
        loss = criterion(outputs, masks)
        print(f"loss: {loss}, image: {images.shape}, outputs: {outputs.shape}, masks: {masks.shape}")
        loss.backward()
        optimizer.step()
        print(f"sad: {torch.argmax(outputs[0], dim=0).shape}")

        visualize_prediction(images[0], torch.argmax(masks[0], dim=0), torch.argmax(outputs[0], dim=0), save_path=f"output_{i}.png")
        
        epoch_loss += loss.item()

        # Log batch-level metrics (less frequently to avoid too many logs)
        
        
        pbar.set_postfix(loss=loss.item())      

    print("Finished training for one epoch.")

    print("Memory after training:")
    print_memory_stats()
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    

    return epoch_loss / len(loader)


def main():
  
    #all the images are different sizes -BUT 
    #this was accounted for in data_preprocessing!!

    #now the correct images are stored in these 3 paths

    train_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/train'
    val_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/val'
    test_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/test'

    parent_dir = "/gws/nopw/j04/iecdt/amorgan/data_copy"
    #original_images_file = "/home/users/amorgan/Calving_Fronts_and_Where_to_Find_Them/original_square_images_actual_names.txt"

    # dataset = GlacierSegDataset(mode='train', parent_dir = parent_dir)
    # image, mask, front = dataset[0]  # Get first sample


    train_dataset_mask = GlacierSegDataset(mode='train', parent_dir=parent_dir, label_type="mask")
    # val_dataset_mask = GlacierSegDataset(mode='val', parent_dir=parent_dir, label_type="mask")
    # test_dataset_mask = GlacierSegDataset(mode='test', parent_dir=parent_dir, label_type="mask")

    # Use 'front' as labels
    # train_dataset_front = GlacierSegDataset(mode='train', parent_dir=parent_dir, label_type="front")
    # val_dataset_front = GlacierSegDataset(mode='val', parent_dir=parent_dir, label_type="front")
    # test_dataset_front = GlacierSegDataset(mode='test', parent_dir=parent_dir, label_type="front")

    # image, label = train_dataset_mask[0]  # Get first sample

    # print(f"Image shape: {image.shape}") 
    # print(f"Label shape: {label.shape}") # label can be either mask or front depending on the label_type

    try:
        image, label = train_dataset_mask[0]  # Get first sample
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
    except Exception as e:
        print(f"Error accessing first sample: {e}")
        return


    model = smp.Unet(  
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,                      # model output channels (number of classes in your dataset), # class ids area=0, stone=1, glacier=2, ocean with ice melange=3
    )

    # Loss and optimizer
    criterion = IoULoss()  # Use the custom IoULoss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_loader = DataLoader(train_dataset_mask, batch_size=2, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)


    print("Starting training loop...")

    epoch_losses = []  # To store loss values for visualization
    for epoch in range(10):
        print(f"Epoch {epoch + 1} starting...")
    
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_losses.append(train_loss)  # Save loss for visualization
    
     

        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
    print("Training loop completed.")

    # Save the model
    torch.save(model.state_dict(), "stusy_amy_trained_model_256.pth")
    
    # Close wandb run

if __name__ == "__main__":
    print("about to go!")
    main()