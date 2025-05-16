from PIL import Image
import os
import torch
from torch.utils.data import DataLoader
from dataset_class import GlacierSegDataset
import segmentation_models_pytorch as smp
import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gc
import psutil
import wandb
import numpy as np
from monai.losses import DiceLoss
import matplotlib.pyplot as plt

#loss - Multiclass IoU
class MultiClassIoULoss(nn.Module):
    def __init__(self, smooth=1e-6, weights=None):
        super(MultiClassIoULoss, self).__init__()
        self.smooth = smooth
        self.weights = weights  # Class weights tensor
        
    def forward(self, y_pred, y_true):
        # y_pred: [B, C, H, W] - softmax probabilities
        # y_true: [B, H, W] - class indices
        
        # Get number of classes from prediction
        num_classes = y_pred.shape[1]
        
        # Convert target to one-hot if it's not already
        if len(y_true.shape) == 3:  # [B, H, W]
            y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).permute(0, 3, 1, 2).float()
        else:  # Assume it's already one-hot [B, C, H, W]
            y_true_one_hot = y_true
            
        # Initialize loss
        class_iou = []
        
        # Calculate IoU for each class
        for cls in range(num_classes):
            pred_cls = y_pred[:, cls]  # [B, H, W]
            true_cls = y_true_one_hot[:, cls]  # [B, H, W]
            
            # Calculate intersection and union
            intersection = torch.sum(pred_cls * true_cls)
            union = torch.sum(pred_cls) + torch.sum(true_cls) - intersection
            
            # Calculate IoU for this class
            iou = (intersection + self.smooth) / (union + self.smooth)
            class_iou.append(iou)
        
        # Convert to tensor
        class_iou = torch.stack(class_iou)
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights.to(y_pred.device)
            class_iou = class_iou * weights
            
        # Return 1 - mean IoU as the loss
        return 1 - torch.mean(class_iou)
    

#loss - Dice
class WeightedDiceLoss(nn.Module):
    def __init__(self, include_background=True, to_onehot_y=True, sigmoid=False, 
                 softmax=True, squared_pred=False, weights=None):
        super(WeightedDiceLoss, self).__init__()
        
        # Initialize MONAI DiceLoss
        self.dice = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,  # Convert target to one-hot
            sigmoid=sigmoid,
            softmax=softmax,  # Apply softmax to predictions
            squared_pred=squared_pred
        )
        self.weights = weights
        
    def forward(self, y_pred, y_true):
        # Calculate dice loss per class
        loss = self.dice(y_pred, y_true)  # returns [loss_class0, loss_class1, ...]
        
        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.to(loss.device)
            loss = loss * weights
            
        # Return mean loss
        return loss.mean()
    

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


# Step 2: Choose a loss function
def get_loss_function(loss_type="iou", class_weights=None):
    if loss_type == "dice":
        return WeightedDiceLoss(weights=class_weights)
    elif loss_type == "iou":
        return MultiClassIoULoss(weights=class_weights)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
def calculate_metrics(y_true, y_pred, num_classes=4):
    """
    Calculate precision, recall, and F1 score for each class.
    
    Args:
        y_true: Ground truth labels (tensor)
        y_pred: Predicted labels (tensor)
        num_classes: Number of classes
        
    Returns:
        precision, recall, f1 arrays
    """
    # Move tensors to CPU and convert to numpy arrays
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy().flatten()
    
    # Initialize arrays to store metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    
    # Calculate metrics for each class
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Calculate precision, recall, and F1 score
        precision[cls] = tp / (tp + fp + 1e-8)
        recall[cls] = tp / (tp + fn + 1e-8)
        f1[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + 1e-8)
    
    return precision, recall, f1

def visualize_prediction(image, mask, pred, class_names):
    """
    Create a visualization of the original image, ground truth mask, and predicted mask.
    Returns a matplotlib figure for logging to wandb.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image.squeeze().cpu().numpy(), cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Define colormap for visualizing masks
    cmap = plt.cm.get_cmap('viridis', 4)
    
    # Ground truth mask
    axes[1].imshow(mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Predicted mask
    axes[2].imshow(pred.cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 3)), 
                       ax=axes, orientation='horizontal', fraction=0.046, pad=0.04)
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    return fig

def train_one_epoch(model, loader, criterion, optimizer, device, class_names=None,epoch=0):
    model.train()
    epoch_loss = 0
    print("Starting training for one epoch...")
    print("Memory before training:")
    print_memory_stats()

     # Initialize metrics
    batch_count = 0
    class_iou_totals = torch.zeros(4).to(device)  # For 4 classes
    confusion_matrix = torch.zeros((4, 4)).to(device)  # For 4 classes

    # Initialize precision, recall, and F1 accumulators
    precision_totals = torch.zeros(4).to(device)
    recall_totals = torch.zeros(4).to(device)
    f1_totals = torch.zeros(4).to(device)

    if class_names is None:
        class_names = ["Background", "Rock", "Glacier", "Ocean&Ice"]
    
    batch_losses = []
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (images, masks) in pbar:
        
        images = images.to(device)  # Shape: [batch_size, 1, height, width]
        masks = masks.to(device).squeeze(1).long()  # Shape: [batch_size, height, width]
        
        optimizer.zero_grad()
        outputs = model(images)
        #calculate the loss
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        epoch_loss += loss.item()

         # Calculate per-class IoU for monitoring
        with torch.no_grad():
            # Get predicted class - argmax gives the index of the max value along the specified dimension
            preds = torch.argmax(outputs, dim=1)  # [batch_size, H, W]
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=4)
            
            # Accumulate metrics
            precision_totals += torch.tensor(batch_precision).to(device)
            recall_totals += torch.tensor(batch_recall).to(device)
            f1_totals += torch.tensor(batch_f1).to(device)
            
            # Calculate IoU for each class
            for cls in range(4):
                pred_cls = (preds == cls)
                target_cls = (masks == cls)
                
                intersection = (pred_cls & target_cls).sum().float()
                union = (pred_cls | target_cls).sum().float()
                
                iou = intersection / (union + 1e-8)
                class_iou_totals[cls] += iou
                
                # Update confusion matrix
                for true_cls in range(4):
                    true_positive = ((masks == true_cls) & (preds == cls)).sum().item()
                    confusion_matrix[true_cls, cls] += true_positive

            if i % 50 == 0:
                # Select first image in batch for visualization
                vis_fig = visualize_prediction(
                    images[0], 
                    masks[0], 
                    preds[0], 
                    class_names
                )
                wandb.log({
                    f"prediction_vis_epoch_{epoch}_batch_{i}": wandb.Image(vis_fig),
                    "epoch": epoch,
                    "batch": i
                })
                plt.close(vis_fig)

        # Log batch-level metrics (less frequently to avoid too many logs)
        if i % 10 == 0:  # Log every 10 batches
            wandb.log({
                "batch_loss": loss.item(),
                "batch": i + len(loader) * epoch,
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]['lr']
            })    

    # Calculate epoch-level metrics
    class_ious = class_iou_totals / batch_count
    mean_iou = class_ious.mean().item()
    
    avg_precision = precision_totals / batch_count
    avg_recall = recall_totals / batch_count
    avg_f1 = f1_totals / batch_count
    
    # Plot batch losses within epoch
    batch_loss_fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(batch_losses)), batch_losses)
    plt.title(f"Batch Losses for Epoch {epoch}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    # Normalize confusion matrix by row (true labels)
    norm_confusion_matrix = confusion_matrix.clone()
    for i in range(4):
        row_sum = confusion_matrix[i].sum()
        if row_sum > 0:
            norm_confusion_matrix[i] = confusion_matrix[i] / row_sum
    
    # Create confusion matrix figure
    conf_fig, ax = plt.subplots(figsize=(8, 7))
    cax = ax.matshow(norm_confusion_matrix.cpu().numpy(), cmap='Blues')
    conf_fig.colorbar(cax)
    
    # Set labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Rotate x labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            value = norm_confusion_matrix[i, j].item()
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", 
                    color="white" if value > 0.5 else "black")
    
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    print("Finished training for one epoch.")
    print("Memory after training:")
    print_memory_stats()
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    
    # Log epoch-level metrics
    metrics = {
        "train_loss": epoch_loss / len(loader),
        "mean_iou": mean_iou,
        "confusion_matrix": wandb.Image(conf_fig),
        "batch_losses": wandb.Image(batch_loss_fig),
        "mean_precision": avg_precision.mean().item(),
        "mean_recall": avg_recall.mean().item(),
        "mean_f1": avg_f1.mean().item()
    }
    
    # Add per-class IoUs
    for i, class_name in enumerate(class_names):
        metrics[f"iou_{class_name}"] = class_ious[i].item()
    
    wandb.log(metrics)
    plt.close(conf_fig)
    plt.close(batch_loss_fig)
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    plt.close('all')
    
    return epoch_loss / len(loader)


def main():
    class_names = ["Background", "Rock", "Glacier", "Ocean&Ice"]

    wandb.init(
        project="CAFFE",  # Choose a name visible in your W&B dashboard
        name="seg_weighted_dice",       # Optional: a custom name for this run
        entity= "amy-morgan-university-of-oxford",
        config={                      # Optional: hyperparameters to log
            "learning_rate": 0.0001,
            "epochs": 40,
            "batch_size": 4,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "model": "Unet_resnet34",
            "encoder": "resnet50",
            "classes": 4,
            "class_names": class_names,
            "loss_function": "WeightedDiceLoss",  # Updated loss function
        }
    )
    train_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/train'
    val_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/val'
    test_data_path = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/test'
    parent_dir = "/gws/nopw/j04/iecdt/amorgan/data_copy"
    
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

    def calculate_class_weights(dataset, max_samples=None, method='balanced'):
        print("Calculating class weights...")
        class_counts = torch.zeros(4)
        total_pixels = 0
        
        # Use a small subset to speed up calculation
        if max_samples is None:
            # Use all samples
            indices = range(len(dataset))
            sample_count = len(dataset)
        else:
            # Use a subset for efficiency
            sample_count = min(max_samples, len(dataset))
            indices = torch.randperm(len(dataset))[:sample_count]
        
        for idx in tqdm.tqdm(indices):
            _, mask = dataset[idx]
            if torch.is_tensor(mask):
                if len(mask.shape) == 4:  # [1, 4, H, W]
                    mask = mask.squeeze(0)
                if mask.shape[0] == 4:  # One-hot encoded
                    # Convert to class indices
                    mask_indices = torch.argmax(mask, axis=0)
                else:
                    mask_indices = mask.squeeze()
                
                # Count pixels per class
                for cls in range(4):
                    class_counts[cls] += torch.sum(mask_indices == cls).item()
                
                total_pixels += mask_indices.numel()
        
        # Calculate frequencies
        class_freq = class_counts / total_pixels
        print(f"Class counts: {class_counts}")
        print(f"Class frequencies: {class_freq}")
        
        if method == 'inverse':
            # Original method - inverse frequency (can be extreme)
            class_weights = 1.0 / (class_freq + 1e-8)
            
        elif method == 'balanced':
            # Balanced method - more moderate weights
            class_weights = 1.0 / (torch.sqrt(class_freq) + 1e-8)
            
        elif method == 'effective':
            # Effective number of samples (better for extreme imbalance)
            # From "Class-Balanced Loss Based on Effective Number of Samples"
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, class_counts)
            class_weights = (1.0 - beta) / (effective_num + 1e-8)
            
        elif method == 'capped':
            # Capped inverse frequency with maximum ratio limit
            max_ratio = 5.0  # Maximum weight ratio between classes
            class_weights = 1.0 / (class_freq + 1e-8)
            max_weight = torch.max(class_weights)
            min_weight = torch.min(class_weights)
            
            if max_weight / min_weight > max_ratio:
                # Scale down the highest weights
                scale_factor = max_ratio * min_weight / max_weight
                class_weights = torch.where(
                    class_weights > max_ratio * min_weight,
                    max_ratio * min_weight,
                    class_weights
                )
        
        # Normalize weights so they sum to number of classes
        class_weights = class_weights / class_weights.sum() * 4
        
        print(f"Class weights: {class_weights}")
        return class_weights


    class_weights = calculate_class_weights(train_dataset_mask, max_samples=1000, method='balanced')
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_weights.numpy())
    plt.title("Class Weights")
    plt.ylabel("Weight Value")
    plt.tight_layout()
    wandb.log({"class_weights": wandb.Image(plt)})
    plt.close()

    model = smp.Unet(  
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,                      # model output channels (number of classes in your dataset), # class ids area=0, stone=1, glacier=2, ocean with ice melange=3
    )
    # Loss and optimizer
    criterion = MultiClassIoULoss(weights=class_weights)  # Use the custom IoULoss
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        model.to(device)
    except RuntimeError as e:
        print(f"Error moving model to GPU: {e}")
        print("Falling back to CPU...")
        device = torch.device('cpu')
        model.to(device)
    if torch.is_tensor(class_weights):
        class_weights = class_weights.to(device)

    train_loader = DataLoader(train_dataset_mask, batch_size=4, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

    print("Starting training loop...")

    epoch_losses = []  # To store loss values for visualization
    patience = 5  # Number of epochs to wait before early stopping
    counter = 0
    best_epoch = 0
    
    for epoch in range(20):
        print(f"Epoch {epoch + 1} starting...")
        best_train_loss = float('inf')
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, class_names)
        epoch_losses.append(train_loss)
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss
        })
        
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}")
        
        # Check if this is the best model so far
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            counter = 0
            best_epoch = epoch
            # Save the best model
            # torch.save(model.state_dict(), "best_model_MultiClassIoULoss_256.pth")
            # print(f"New best model saved at epoch {epoch+1}")
        else:
            counter += 1
            print(f"EarlyStopping counter: {counter} out of {patience}")
            
        # If we've waited for 'patience' epochs without improvement, stop training
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}. Best epoch was {best_epoch+1}.")
            break
    # Save the model
    torch.save(model.state_dict(), "MultiClassIoULoss_trained_model_256.pth")
    wandb.save("MultiClassIoULoss_trained_model_256.pth")

    wandb.finish()
    print("Training loop completed.")

if __name__ == "__main__":
    print("about to go!")
    main()



#archive
# A class to handle the segmentation loss
# class IoULoss(nn.Module):
#     """
#     A class for the Jaccard index loss function.
#     """
#     def __init__(self, smooth=1e-6):
#         """
#         Initializes the Jaccard index loss function.

#         Args:
#             smooth (float): A small constant to avoid division by zero.
#         """
#         super(IoULoss, self).__init__()
#         self.smooth = smooth

#     def forward(self, y_pred, y_true):
#         """
#         Computes the Jaccard index loss.

#         Args:
#             y_pred (torch.Tensor): The predicted output.
#             y_true (torch.Tensor): The ground truth labels.

#         Returns:
#             torch.Tensor: The computed Jaccard index loss.
#         """
#         intersection = torch.sum(y_pred * y_true)
#         union = torch.sum(y_pred) + torch.sum(y_true) - intersection
#         return 1 - (intersection + self.smooth) / (union + self.smooth)