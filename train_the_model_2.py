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
from monai.losses import DiceLoss, DiceCELoss, FocalLoss
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import random
import cv2
from sklearn.metrics import confusion_matrix
import hydra
import logging
from omegaconf import DictConfig, OmegaConf
from loss_file import get_loss_function
from epoch_train import train_one_epoch, validate, calculate_metrics
from utils_AM import print_gpu_usage

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

# # Config dictionary - centralize all hyperparameters
# CONFIG = {
#     "model": {
#         "encoder_name": "resnet50",  # Upgraded from resnet34
#         "encoder_weights": "imagenet",
#         "in_channels": 1,
#         "classes": 4
#     },
#     "training": {
#         "batch_size": 4,  # Increased from 4
#         "learning_rate": 1e-4,  # Slightly higher initial LR with scheduler
#         "weight_decay": 1e-4,
#         "epochs": 50,  # Increased max epochs
#         "patience": 8,  # Early stopping patience
#         "class_weight_method": "balanced",  # Options: inverse, balanced, effective, capped
#         "mixed_precision": True,  # Enable mixed precision training
#         "clip_grad_norm": 1.0
#     },
#     "data": {
#         "img_size": 256,  # Image size for training
#         "augmentation_prob": 0.5,  # Probability of applying augmentations
#         "parent_dir": "/gws/nopw/j04/iecdt/amorgan/data_copy"
#     },
#     "class_names": ["Background", "Rock", "Glacier", "Ocean&Ice"]
# }

def calculate_class_weights(dataset, max_samples=500, method="balanced"):
    log.info("Calculating class weights...")
    class_counts = torch.zeros(4)

    # Use a subset of the dataset for efficiency
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

    total_pixels = class_counts.sum()
    class_freq = class_counts / total_pixels
    log.info(f"Class frequencies: {class_freq}")

    if method == "inverse":
        # Original method - inverse frequency (can be extreme)
        class_weights = 1.0 / (class_freq + 1e-8)

    elif method == "balanced":
        # Balanced method - more moderate weights
        class_weights = 1.0 / (torch.sqrt(class_freq) + 1e-8)

    elif method == "effective":
        # Effective number of samples (better for extreme imbalance)
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, class_counts)
        class_weights = (1.0 - beta) / (effective_num + 1e-8)

    elif method == "capped":
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
                class_weights,
            )

    # Normalize weights so they sum to number of classes
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    log.info(f"Class weights: {class_weights}")
    return class_weights


""" 
HYDRA IMPLEMENTATION
"""
log = logging.getLogger(__name__)


@hydra.main(
    version_base=None, config_path="conf", config_name="config"
)  # TODO: could change to train and test
def main(cfg: DictConfig):  # before was def main():  ## NOT COMPLETED YET

    print("MAIN FUNCTION STARTED")
    print("Config keys:", list(cfg.keys()))
    print("Training config:", cfg.training)
    print("Model config:", cfg.model)

    # ALL NEW HYDRA STUFF
    set_seed(cfg["seed"])

    # if cfg.use_wandb:
    #     wandb.init(
    #         project=cfg.name,
    #         config=OmegaConf.to_container(cfg, resolve=True),
    #         name=cfg.name
    #     )
    print("seed set")
    wandb.init(

        #usual slurm setup!
        project="CAFFE",
        name="seg_improved_glacier",
        entity="amy-morgan-university-of-oxford",
        settings=wandb.Settings(start_method="thread"),
        job_type="training",
        save_code=False,
        #config=cfg, 
        config=OmegaConf.to_container(cfg, resolve=True), 
      
        # config=cfg#OmegaConf.to_container(cfg, resolve=True),



        # for wandb hyperparameter sweeps
        #config = wandb.config
    )
#debug
    # print(config)

    print("wandb init done")

    class_names = cfg["class_names"]
    parent_dir = cfg["data"]["parent_dir"]
    filtered_csv_path = os.path.join(parent_dir, "train_cleaned_patches.csv")

    # save the models to gws
    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Load datasets
    train_dataset = GlacierSegDataset(
        mode="train",
        parent_dir=parent_dir,
        label_type="mask",
        filtered_csv_path=filtered_csv_path,
    )
    val_dataset = GlacierSegDataset(
        mode="val", parent_dir=parent_dir, label_type="mask"
    )
    class_weights = calculate_class_weights(
        train_dataset, method=cfg["training"]["class_weight_method"]
    )
    print(f"Class weights: {class_weights}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        # persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        # persistent_workers=True,
    )

    log.info("After DataLoader creation")

    # TODO: Load trained model if the weights exist
    # Add batch size to the model name
    model_name_prefix = f"{cfg["name"]}_bs{cfg['training']['batch_size']}"
    best_model_loss_path = os.path.join(
        cfg["save_dir"], f"{model_name_prefix}_best_loss.pth"
    )
    best_model_iou_path = os.path.join(
        cfg["save_dir"], f"{model_name_prefix}_best_iou.pth"
    )
    checkpoint_path = os.path.join(
        cfg["save_dir"], f"{model_name_prefix}_latest_epoch.pth"
    )

    # Create model
    model = smp.Unet(
        encoder_name=cfg["model"]["encoder_name"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=cfg["model"]["in_channels"],
        classes=cfg["model"]["classes"],
    )

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    log.info(f"Using device: {device}")
    log.info(print_gpu_usage("After model.to(device)"))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Learning rate scheduler - cosine annealing with warm restarts
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # Restart every 10 epochs
        T_mult=1,  # Multiply period by 1 (no change) after each restart
        eta_min=1e-6,  # Minimum learning rate
    )

    # Check if a model load path is specified and exists
    # We also need to load the optimizer and scheduler to continue training
    if cfg["load_path"] and os.path.exists(checkpoint_path):
        log.info(f"Loading model weights from {checkpoint_path}")
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        log.info("Model weights loaded successfully.")
        print("Model weights loaded successfully.")
    else:
        log.info(
            "No valid load_path specified or file does not exist. Training from scratch."
        )

    if torch.is_tensor(class_weights):
        class_weights = class_weights.to(device)

    # Loss function and optimizer
    criterion = get_loss_function("combined", class_weights)
    best_val_loss = float("inf")
    best_val_iou = 0.0

    for epoch in range(cfg["training"]["epochs"]):

        log.info(print_gpu_usage(f"Start of epoch {epoch}"))
        # Train one epoch
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            cfg,  # used to be class_names
            log,
            epoch=epoch,
        )
        log.info(print_gpu_usage(f"After train_one_epoch {epoch}"))
        val_loss, val_iou = validate(
            model, val_loader, criterion, device, cfg, log, epoch=epoch
        )
        log.info(print_gpu_usage(f"After validate {epoch}"))
        # Update scheduler
        scheduler.step()
        if cfg.get("use_wandb", False):
            wandb.log({"train_loss": train_loss, "epoch": epoch})

        if val_loss > best_val_loss:
            best_val_loss = val_loss
            # Save model checkpoint with more information
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "config": cfg,
                },
                best_model_loss_path,
            )

        if val_iou > best_val_iou:
            best_val_iou = val_iou

            # Save model checkpoint with more information
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "config": cfg,
                },
                best_model_iou_path,
            )

        if epoch % cfg.save_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler else None
                    ),
                    "val_loss": val_loss,
                    "val_iou": val_iou,
                    "config": cfg,
                },
                checkpoint_path,
            )

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        log.info(print_gpu_usage(f"After empty_cache/collect {epoch}"))

    # Final evaluation on validation set using best model
    print(f"\n{'='*20} Final Evaluation {'='*20}")

    # Final validation
    final_val_loss, final_val_iou = validate(
        model,
        val_loader,
        criterion,
        device,
        class_names=class_names,
        epoch=cfg["training"]["epochs"],  # Just for logging purposes
    )

    # Log final results
    final_metrics = {
        "final_val_loss": final_val_loss,
        "final_val_iou": final_val_iou,
        "best_val_loss": best_val_loss,
        "best_val_iou": best_val_iou,
        "total_epochs_trained": epoch + 1,
    }
    wandb.log(final_metrics)

    # Print summary
    print("\nTraining Summary:")
    print(f"Total epochs trained: {epoch + 1}")
    print(f"Best validation IoU: {best_val_iou:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final validation IoU: {final_val_iou:.4f}")
    print(f"Final validation loss: {final_val_loss:.4f}")

    wandb.finish()


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
    config = checkpoint["config"]

    model = smp.Unet(
        encoder_name=config["model"]["encoder_name"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Setup data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
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
            batch_precision, batch_recall, batch_f1 = calculate_metrics(
                masks, preds, num_classes=num_classes
            )
            precision_totals += torch.tensor(batch_precision).to(device)
            recall_totals += torch.tensor(batch_recall).to(device)
            f1_totals += torch.tensor(batch_f1).to(device)

            # Calculate IoU for each class
            for cls in range(num_classes):
                pred_cls = preds == cls
                target_cls = masks == cls

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
        "f1_per_class": avg_f1.cpu().numpy(),
    }

    return metrics


# Main execution entry point
if __name__ == "__main__":
    import pickle
    import cloudpickle
    # Test pickling of main function
    try:
        cloudpickle.dumps(main)
        print("Main function is picklable")
    except Exception as e:
        print(f"Main function pickling failed: {e}")
    main()
