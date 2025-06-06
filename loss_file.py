
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, DiceCELoss, FocalLoss

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
