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
import numpy as np
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import random
import cv2
from sklearn.metrics import confusion_matrix, jaccard_score

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Config dictionary - centralize all hyperparameters
CONFIG = {
    "model": {
        "encoder_name": "resnet50",  # Upgraded from resnet34
        "encoder_weights": "imagenet",
        "in_channels": 1,
        "classes": 4
    },
    "training": {
        "batch_size": 4,  # Increased from 4
        "learning_rate": 1e-4,  # Slightly higher initial LR with scheduler
        "weight_decay": 1e-4,
        "epochs": 50,  # Increased max epochs
        "patience": 8,  # Early stopping patience
        "class_weight_method": "balanced",  # Options: inverse, balanced, effective, capped
        "mixed_precision": True,  # Enable mixed precision training
        "clip_grad_norm": 1.0
    },
    "data": {
        "img_size": 256,  # Image size for training
        "augmentation_prob": 0.5,  # Probability of applying augmentations
        "parent_dir": "/gws/nopw/j04/iecdt/amorgan/data_copy"
    },
    "class_names": ["Background", "Rock", "Glacier", "Ocean&Ice"]
}

# Loss - Multiclass IoU with gradient stability improvements
class MultiClassIoULoss(nn.Module):
    def __init__(self, smooth=1e-5, weights=None):
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
            print(f"y_true_one_hot shape: {y_true_one_hot.shape}")
        else:  # Assume it's already one-hot [B, C, H, W]
            y_true_one_hot = y_true
            
        # Initialize loss
        class_iou = []
        
        # Calculate IoU for each class
        for cls in range(num_classes):
            pred_cls = y_pred[:, cls]  # [B, H, W]
            true_cls = y_true_one_hot[:, cls]  # [B, H, W]
            
            # Calculate intersection and union
            intersection = torch.sum(pred_cls * true_cls, dim=(1, 2))
            pred_sum = torch.sum(pred_cls, dim=(1, 2))
            true_sum = torch.sum(true_cls, dim=(1, 2))
            union = pred_sum + true_sum - intersection
            
            # Calculate batch IoU for this class - use batch mean for stability
            batch_iou = (intersection + self.smooth) / (union + self.smooth)
            iou = torch.mean(batch_iou)
            class_iou.append(iou)
        
        # Convert to tensor
        class_iou = torch.stack(class_iou)
        
        # Apply weights if provided
        if self.weights is not None:
            weights = self.weights.to(y_pred.device)
            class_iou = class_iou * weights
            
        # Return 1 - mean IoU as the loss
        return 1 - torch.mean(class_iou)

# Combined loss function for better performance
class CombinedLoss(nn.Module):
    def __init__(self, weights=None, dice_weight=0.5, focal_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=False,
            softmax=True,
            squared_pred=False,
            smooth_nr=1e-5,
            smooth_dr=1e-5
        )
        self.focal_loss = FocalLoss(
            include_background=True,
            to_onehot_y=False,
            gamma=2.0
        )
        self.weights = weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
    def forward(self, y_pred, y_true):
        # Calculate losses
        print(f"y_pred shape: {y_pred.shape}, y_true shape: {y_true.shape}")
        dice_loss = self.dice_loss(y_pred, y_true)
        focal_loss = self.focal_loss(y_pred, y_true)
        
        # Apply class weights if provided
        if self.weights is not None:
            weights = self.weights.to(dice_loss.device)
            dice_loss = dice_loss * weights
            focal_loss = focal_loss * weights
            
        # Calculate weighted sum
        total_loss = (self.dice_weight * dice_loss.mean() + 
                      self.focal_weight * focal_loss.mean())
        
        return total_loss

# Check memory status
def print_memory_stats():
    process = psutil.Process(os.getpid())
    print(f"Process memory usage: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / (1024 * 1024):.2f} MB")
            # print(f"GPU {i} memory cached: {torch.cuda.memory_cached(i) / (1024 * 1024):.2f} MB")

# Get loss function based on selection
def get_loss_function(loss_type="combined", class_weights=None):
    if loss_type == "dice":
        return DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=False
        )
    elif loss_type == "dicece":
        return DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=0.5,
            lambda_ce=0.5
        )
    elif loss_type == "iou":
        return MultiClassIoULoss(weights=class_weights)
    elif loss_type == "combined":
        return CombinedLoss(weights=class_weights, dice_weight=0.5, focal_weight=0.5)
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
    print(f"prreeee y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

    # Move tensors to CPU and convert to numpy arrays
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy().flatten()
    print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
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

# Enhanced visualization function
def visualize_prediction(image, mask, pred, class_names, sample_info=None):
    """
    Create a visualization of the original image, ground truth mask, and predicted mask.
    Returns a matplotlib figure for logging to wandb.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image - with proper SAR normalization
    img_display = image.squeeze().cpu().numpy()
    # Log scale visualization for SAR images (optional)
    img_display = np.clip(img_display, 0.01, np.percentile(img_display, 99))
    img_display = 20 * np.log10(img_display)
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
    
    axes[0].imshow(img_display, cmap='gray')
    axes[0].set_title('Original SAR Image')
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
    
    # Add sample info if provided
    if sample_info:
        plt.suptitle(f"Sample: {sample_info}", fontsize=14)
    
    plt.tight_layout()
    return fig

def train_one_epoch(model, loader, criterion, optimizer, device, class_names=None, epoch=0, scaler=None):
    model.train()
    epoch_loss = 0
    batch_losses = []
    print(f"Starting training for epoch {epoch+1}...")
    
    # Initialize metrics
    batch_count = len(loader)
    class_iou_totals = torch.zeros(4).to(device)  # For 4 classes
    confusion_matrix = torch.zeros((4, 4)).to(device)  # For 4 classes

    # Initialize precision, recall, and F1 accumulators
    precision_totals = torch.zeros(4).to(device)
    recall_totals = torch.zeros(4).to(device)
    f1_totals = torch.zeros(4).to(device)

    if class_names is None:
        class_names = CONFIG["class_names"]
    
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (images, masks) in pbar:
        print(f"input images shape: {images.shape},\n input masks shape: {masks.shape}")
        images = images.to(device)  # Shape: [batch_size, 1, height, width]
        masks = masks.to(device).long()  #squeeze 1 was causing issues
        
        print("masks shape:", masks.shape)
        print("masks unique:", torch.unique(masks))
        print(f"images shape: {images.shape}")
        
        optimizer.zero_grad()
        
        # Mixed precision training if enabled
        if CONFIG["training"]["mixed_precision"] and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
                
            # Scaled backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if CONFIG["training"]["clip_grad_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              max_norm=CONFIG["training"]["clip_grad_norm"])
            
            # Update weights with gradient scaling
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(images)
            print(f"pre masks: {masks}")
            masks = torch.nn.functional.one_hot(masks, num_classes=4).permute(0, 3, 1, 2)
            print("outputs shape:", outputs.shape)
            print("masks shape (after forward):", masks.shape)
            print("masks unique (after forward):", torch.unique(masks))
            loss = criterion(outputs, masks) 
            loss.backward()
            
            if CONFIG["training"]["clip_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                              max_norm=CONFIG["training"]["clip_grad_norm"])
            
            optimizer.step()
        
        # Record the loss
        current_loss = loss.item()
        epoch_loss += current_loss
        batch_losses.append(current_loss)
        
        # Update progress bar
        pbar.set_description(f"Epoch {epoch+1} | Loss: {current_loss:.4f}")
     

        # Calculate per-class metrics for monitoring
        with torch.no_grad():
            # Get predicted class
            preds = torch.argmax(outputs, dim=1)  # [batch_size, H, W]
            masks_flattened = torch.argmax(masks, dim=1)
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks_flattened, preds, num_classes=4)
            
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

            # Visualize predictions (less frequently to save resources)
            if i % 50 == 0:
                # Select first image in batch for visualization
                print(f"images shape: {images.shape}, masks shape: {masks.shape}, mask flattened: {masks_flattened.shape}, preds shape: {preds.shape}")
                vis_fig = visualize_prediction(
                    images[0], 
                    masks_flattened[0], 
                    preds[0], 
                    class_names,
                    sample_info=f"Epoch {epoch+1}, Batch {i}"
                )
                plt.show()
                #plt.close(vis_fig)

        # Log batch-level metrics (less frequently to avoid too many logs)
     

    # Calculate epoch-level metrics
    class_ious = class_iou_totals / batch_count
    mean_iou = class_ious.mean().item()
    
    avg_precision = precision_totals / batch_count
    avg_recall = recall_totals / batch_count
    avg_f1 = f1_totals / batch_count
    
    # Plot batch losses within epoch
    batch_loss_fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(batch_losses)), batch_losses)
    plt.title(f"Batch Losses for Epoch {epoch+1}")
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

  
    plt.close(conf_fig)
    plt.close(batch_loss_fig)
    
    # Clear cache and collect garbage
    torch.cuda.empty_cache()
    gc.collect()
    plt.close('all')
    
    return epoch_loss / len(loader)

def validate(model, loader, criterion, device, class_names=None, epoch=0):
    model.eval()
    val_loss = 0
    
    # Initialize metrics
    batch_count = len(loader)
    class_iou_totals = torch.zeros(4).to(device)
    precision_totals = torch.zeros(4).to(device)
    recall_totals = torch.zeros(4).to(device)
    f1_totals = torch.zeros(4).to(device)
    
    if class_names is None:
        class_names = CONFIG["class_names"]
        
    print(f"Running validation for epoch {epoch+1}...")
    
    with torch.no_grad():
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        for i, (images, masks) in pbar:
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            # val_loss += loss.item()
            
            # Get predicted class
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics for this batch
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=4)
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
            
            # Visualize some predictions
            if i == 0:  # Just the first batch
                for j in range(min(4, images.size(0))):
                    vis_fig = visualize_prediction(
                        images[j],
                        masks[j],
                        preds[j],
                        class_names,
                        sample_info=f"Validation - Epoch {epoch+1}"
                    )
            
                    plt.close(vis_fig)
    
    # Calculate average metrics
    # avg_val_loss = val_loss / batch_count
    class_ious = class_iou_totals / batch_count
    mean_iou = class_ious.mean().item()
    
    avg_precision = precision_totals / batch_count
    avg_recall = recall_totals / batch_count
    avg_f1 = f1_totals / batch_count
    
    # # Log validation metrics
    # metrics = {
    #     "val_loss": avg_val_loss,
    #     "val_mean_iou": mean_iou,
    #     "val_mean_precision": avg_precision.mean().item(),
    #     "val_mean_recall": avg_recall.mean().item(),
    #     "val_mean_f1": avg_f1.mean().item(),
    #     "epoch": epoch
    # }
    
    # # Add per-class metrics
    # for i, class_name in enumerate(class_names):
    #     metrics[f"val_iou_{class_name}"] = class_ious[i].item()
    #     metrics[f"val_precision_{class_name}"] = avg_precision[i].item() 
    #     metrics[f"val_recall_{class_name}"] = avg_recall[i].item()
    #     metrics[f"val_f1_{class_name}"] = avg_f1[i].item()
    
    # wandb.log(metrics)
    
    return mean_iou

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
            class_weights = torch.where(
                class_weights > max_ratio * min_weight,
                max_ratio * min_weight,
                class_weights
            )
    
    # Normalize weights so they sum to number of classes
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    print(f"Class weights: {class_weights}")
    return class_weights

def main():
    # Initialize W&B
    
    class_names = CONFIG["class_names"]
    parent_dir = CONFIG["data"]["parent_dir"]
    
    print("making dataset...")
    # Load datasets
    train_dataset = GlacierSegDataset(mode='train', parent_dir=parent_dir, label_type="mask")
    # val_dataset = GlacierSegDataset(mode='val', parent_dir=parent_dir, label_type="mask")
    print("dataset made")
    # Calculate class weights
    # class_weights = calculate_class_weights(
    #     train_dataset, 
    #     max_samples=1000, 
    #     method=CONFIG["training"]["class_weight_method"]
    # )
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0])  # Placeholder for testing
    print(f"class weights calculated: {class_weights}")
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG["training"]["batch_size"], 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    print("train loader made")
    # Visualize class weights
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, class_weights.numpy())
    plt.title("Class Weights")
    plt.ylabel("Weight Value")
    plt.tight_layout()
    plt.close()
    print("making model...")
    # Create model
    model = smp.Unet(
        encoder_name=CONFIG["model"]["encoder_name"],
        encoder_weights=CONFIG["model"]["encoder_weights"],
        in_channels=CONFIG["model"]["in_channels"],
        classes=CONFIG["model"]["classes"]
    )
    print("model made")
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    if torch.is_tensor(class_weights):
        class_weights = class_weights.to(device)
    
    # Loss function and optimizer
    criterion = get_loss_function("combined", class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=CONFIG["training"]["learning_rate"], 
        weight_decay=CONFIG["training"]["weight_decay"]
    )
    
    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=1,  # Multiply period by 1 (no change) after each restart
        eta_min=1e-6  # Minimum learning rate
    )
      # Training loop
    

    patience_counter = 0
    print("Starting training...")
    for epoch in range(CONFIG["training"]["epochs"]):
        # Train one epoch
        train_loss = train_one_epoch(
            model, 
            train_loader, 
            criterion, 
            optimizer, 
            device, 
            class_names=class_names,
            epoch=epoch
        )
        
        # Update scheduler
        scheduler.step()
        # current_lr = optimizer.param_groups[0]['lr']
        # print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
        
        # Check for early stopping
        if patience_counter >= CONFIG["training"]["patience"]:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
            
        # Print memory stats
        if epoch % 5 == 0:
            print_memory_stats()
            
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
    print("Training completed!")

def inference(model_path, image_path, device=None):
    """
    Run inference on a single image using a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        image_path: Path to the input image
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        prediction: Segmentation mask (numpy array)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = smp.Unet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,  # Don't load pretrained weights
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = np.array(image)
    
    # Resize to match model's expected input size
    img_size = config["data"]["img_size"]
    image = cv2.resize(image, (img_size, img_size))
    
    # Normalize and convert to tensor
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    image = image.to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    return prediction

def visualize_inference(image_path, prediction, class_names=None):
    """
    Visualize inference results.
    
    Args:
        image_path: Path to the original image
        prediction: Predicted segmentation mask (numpy array)
        class_names: List of class names
    
    Returns:
        fig: Matplotlib figure
    """
    if class_names is None:
        class_names = CONFIG["class_names"]
    
    # Load original image
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    
    # Resize original image to match prediction size
    image = cv2.resize(image, (prediction.shape[1], prediction.shape[0]))
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Display prediction
    cmap = plt.cm.get_cmap('viridis', len(class_names))
    im = axes[1].imshow(prediction, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title('Segmentation Prediction')
    axes[1].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_ticks(np.arange(len(class_names)) + 0.5)
    cbar.set_ticklabels(class_names)
    
    plt.tight_layout()
    return fig

def evaluate_model(model_path, dataset, device=None):
    """
    Evaluate model on a dataset.
    
    Args:
        model_path: Path to the trained model checkpoint
        dataset: Dataset to evaluate on
        device: Device to run evaluation on ('cuda' or 'cpu')
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = smp.Unet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Setup data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize metrics
    num_classes = config["model"]["classes"]
    class_iou_totals = torch.zeros(num_classes).to(device)
    precision_totals = torch.zeros(num_classes).to(device)
    recall_totals = torch.zeros(num_classes).to(device)
    f1_totals = torch.zeros(num_classes).to(device)
    
    # Run evaluation
    with torch.no_grad():
        for images, masks in tqdm.tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device).long()
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate metrics
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=num_classes)
            precision_totals += torch.tensor(batch_precision).to(device)
            recall_totals += torch.tensor(batch_recall).to(device)
            f1_totals += torch.tensor(batch_f1).to(device)
            
            # Calculate IoU for each class
            for cls in range(num_classes):
                pred_cls = (preds == cls)
                target_cls = (masks == cls)
                
                intersection = (pred_cls & target_cls).sum().float()
                union = (pred_cls | target_cls).sum().float()
                
                iou = intersection / (union + 1e-8)
                class_iou_totals[cls] += iou
    
    # Calculate final metrics
    class_ious = class_iou_totals / len(dataloader)
    mean_iou = class_ious.mean().item()
    
    avg_precision = precision_totals / len(dataloader)
    avg_recall = recall_totals / len(dataloader)
    avg_f1 = f1_totals / len(dataloader)
    
    # Create metrics dictionary
    metrics = {
        "mean_iou": mean_iou,
        "class_ious": class_ious.cpu().numpy(),
        "mean_precision": avg_precision.mean().item(),
        "mean_recall": avg_recall.mean().item(),
        "mean_f1": avg_f1.mean().item(),
        "precision_per_class": avg_precision.cpu().numpy(),
        "recall_per_class": avg_recall.cpu().numpy(),
        "f1_per_class": avg_f1.cpu().numpy()
    }
    
    return metrics



# Main execution entry point
if __name__ == "__main__":
    # Call the main training function
    main()