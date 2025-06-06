import os
import torch
import tqdm
import gc
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wandb
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from utils_AM import print_gpu_usage

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
        f1[cls] = (
            2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls] + 1e-8)
        )

    return precision, recall, f1


def train_one_epoch(
    model, loader, criterion, optimizer, device, cfg, log, epoch=0, scaler=None
):  # for hydra, remove class_names and replace w cfg
    model.train()
    epoch_loss = 0
    batch_losses = []
    # print(f"Starting training for epoch {epoch+1}...")
    log.info(f"Starting training for epoch {epoch+1}...")

    # Initialize metrics
    batch_count = len(loader)
    running_precision = np.zeros(cfg.model.classes)
    running_recall = np.zeros(cfg.model.classes)
    running_f1 = np.zeros(cfg.model.classes)
    num_batches = 0
    # all_preds = []
    # all_masks = []
    # class_iou_totals = torch.zeros(4).to(device)  # For 4 classes
    # confusion_matrix = torch.zeros((4, 4)).to(device)  # For 4 classes

    # # Initialize precision, recall, and F1 accumulators
    # precision_totals = torch.zeros(4).to(device)
    # recall_totals = torch.zeros(4).to(device)
    # f1_totals = torch.zeros(4).to(device)
    is_slurm = "SLURM_JOB_ID" in os.environ
    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))

    # HYDRA
    class_names = cfg.class_names
    log_freq = cfg.log_freq

    # if class_names is None:
    #     class_names = CONFIG["class_names"]

    # if is_slurm:
    #     loader_iter = enumerate(loader)
    #     print(f"Running in SLURM environment. Total batches: {batch_count}")

    # else:
    #     loader_iter = tqdm.tqdm(enumerate(loader), total=batch_count)

    vis_interval = max(len(loader) - 1, 1)

    for i, (images, masks) in pbar:
        images = images.to(device)  # Shape: [batch_size, 1, height, width]
        masks = masks.to(device).long()  # squeeze 1 was causing issues
        optimizer.zero_grad()

        # Apply gradient accumulation if configured
        is_accumulation_step = (i + 1) % cfg.training.gradient_accumulation_steps != 0
        if i % cfg.training.gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        # Mixed precision training if enabled
        if cfg.training.mixed_precision and scaler is not None:
            with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                outputs = model(images)
                masks_one_hot = (
                    F.one_hot(masks, num_classes=cfg.model.classes)
                    .permute(0, 3, 1, 2)
                    .float()
                )
                loss = criterion(outputs, masks_one_hot)

                # Scale loss by gradient accumulation steps
                if cfg.training.gradient_accumulation_steps > 1:
                    loss = loss / cfg.training.gradient_accumulation_steps

            # Scaled backward pass
            scaler.scale(loss).backward()

            if not is_accumulation_step:
                # Gradient clipping
                if cfg.training.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cfg.training.clip_grad_norm
                    )

                # Update weights with gradient scaling
                scaler.step(optimizer)
                scaler.update()
        else:
            # Standard training
            outputs = model(images)
            masks_one_hot = (
                F.one_hot(masks, num_classes=cfg.model.classes)
                .permute(0, 3, 1, 2)
                .float()
            )
            loss = criterion(outputs, masks_one_hot)

            # Scale loss by gradient accumulation steps
            if cfg.training.gradient_accumulation_steps > 1:
                loss = loss / cfg.training.gradient_accumulation_steps

            loss.backward()

            if not is_accumulation_step:
                if cfg.training.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=cfg.training.clip_grad_norm
                    )
                optimizer.step()

        # Record the loss
        # current_loss = loss.item()
        current_loss = loss.item() * (
            cfg.training.gradient_accumulation_steps if is_accumulation_step else 1
        )
        epoch_loss += current_loss
        batch_losses.append(current_loss)

        # Accumulate predictions and masks for metrics at the end of the epoch
        with torch.no_grad():
            # Get predicted class
            preds = torch.argmax(outputs, dim=1)  # [batch_size, H, W]
            # all_preds.append(preds.cpu())
            # all_masks.append(masks.cpu())
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=cfg.model.classes)
            running_precision += batch_precision
            running_recall += batch_recall
            running_f1 += batch_f1
            num_batches += 1

            # batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=4)
            # precision_totals += torch.tensor(batch_precision).to(device)
            # recall_totals += torch.tensor(batch_recall).to(device)
            # f1_totals += torch.tensor(batch_f1).to(device)

        # Log batch-level metrics (less frequently to avoid too many logs)
        if i % log_freq == 0:  # Log every 10 batches
            if cfg.use_wandb:
                wandb.log(
                    {
                        "batch_loss": current_loss,
                        "batch": i + len(loader) * epoch,
                        "epoch": epoch,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    }
                )

        if i == vis_interval:
            vis_fig = visualize_prediction(
                images[0],
                masks[0],
                preds[0],
                class_names,
                sample_info=f"Epoch {epoch+1}, Final Batch",
            )
            if cfg.use_wandb:
                wandb.log(
                    {
                        f"prediction_vis_epoch_{epoch}": wandb.Image(vis_fig),
                        "epoch": epoch,
                    }
                )
            plt.close(vis_fig)

        if is_slurm and (i % log_freq == 0 or i == batch_count - 1):
            print(
                f"Epoch {epoch+1} | Batch {i+1}/{batch_count} | Loss: {loss.item():.4f}"
            )

        if i % 500 == 0:
            log.info(print_gpu_usage(f"Epoch {epoch} Batch {i}"))

    # all_preds = torch.cat(all_preds, dim=0)
    # all_masks = torch.cat(all_masks, dim=0)

    # Calculate metrics ONCE at the end of the epoch
    # precision, recall, f1 = calculate_metrics(
    #     all_masks, all_preds, num_classes=cfg.model.classes
    # )

    precision = running_precision / num_batches
    recall = running_recall / num_batches
    f1 = running_f1 / num_batches

    # Calculate IoU for each class
    class_ious = []
    for cls in range(cfg.model.classes):
        pred_cls = preds == cls
        target_cls = masks == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        iou = intersection / (union + 1e-8)
        # class_iou_totals[cls] += iou
        class_ious.append(iou.item())
    class_ious = np.array(class_ious)
    mean_iou = class_ious.mean()
    # # Update confusion matrix
    # for true_cls in range(4):
    #     true_positive = ((masks == true_cls) & (preds == cls)).sum().item()
    #     confusion_matrix[true_cls, cls] += true_positive

    # # Calculate epoch-level metrics
    # class_ious = class_iou_totals / batch_count
    # mean_iou = class_ious.mean().item()

    # avg_precision = precision_totals / batch_count
    # avg_recall = recall_totals / batch_count
    # avg_f1 = f1_totals / batch_count

    # Plot batch losses within epoch
    batch_loss_fig = plt.figure(figsize=(10, 5))
    plt.plot(range(len(batch_losses)), batch_losses)
    plt.title(f"Batch Losses for Epoch {epoch+1}")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()

    # Log epoch-level metrics
    metrics = {
        "train_loss": epoch_loss / len(loader),
        "mean_iou": mean_iou,
        "batch_losses": wandb.Image(batch_loss_fig),
        "mean_precision": precision.mean(),
        "mean_recall": recall.mean(),
        "mean_f1": f1.mean(),
        "epoch": epoch,
    }
    if cfg.use_wandb:
        metrics["batch_losses"] = wandb.Image(batch_loss_fig)

    # Add per-class IoUs
    for i, class_name in enumerate(class_names):
        metrics[f"train_iou_{class_name}"] = class_ious[i]
        metrics[f"train_precision_{class_name}"] = precision[i]
        metrics[f"train_recall_{class_name}"] = recall[i]
        metrics[f"train_f1_{class_name}"] = f1[i]

    if cfg.use_wandb:
        wandb.log(metrics)

    plt.close(batch_loss_fig)
    plt.close("all")

    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss / len(loader)

def validate(
    model, loader, criterion, device, cfg, log, epoch=0
):  # remove class_names for cfg
    model.eval()
    val_loss = 0
    vis_batch = len(loader) - 1

    # Initialize metrics
    batch_count = len(loader)

    running_precision = np.zeros(cfg.model.classes)
    running_recall = np.zeros(cfg.model.classes)
    running_f1 = np.zeros(cfg.model.classes)
    num_batches = 0
    # all_preds = []
    # all_masks = []
    # class_iou_totals = torch.zeros(4).to(device)
    # precision_totals = torch.zeros(4).to(device)
    # recall_totals = torch.zeros(4).to(device)
    # f1_totals = torch.zeros(4).to(device)
    # confusion_matrix = torch.zeros((4, 4)).to(device)  # For 4 classes

    class_names = cfg.class_names
    log_freq = cfg.log_freq
    log.info(f"Running validation for epoch {epoch+1}...")

    if class_names is None:
        class_names = cfg["class_names"]

    # print(f"Running validation for epoch {epoch+1}...")

    is_slurm = "SLURM_JOB_ID" in os.environ
    if is_slurm:
        loader_iter = enumerate(loader)
        total_batches = len(loader)
        # print(f"Running validation in SLURM environment. Total batches: {total_batches}")
    else:
        pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
        loader_iter = pbar

    with torch.no_grad():
        for i, (images, masks) in loader_iter:
            images = images.to(device)
            masks = masks.to(device).long()

            outputs = model(images)
            # loss = criterion(outputs, masks)
            masks_one_hot = torch.nn.functional.one_hot(
                masks, num_classes=cfg.model.classes
            ).permute(0, 3, 1, 2)
            loss = criterion(outputs, masks_one_hot)
            val_loss += loss.item()

            # Get predicted class
            preds = torch.argmax(outputs, dim=1)
            # all_preds.append(preds.cpu())
            # all_masks.append(masks.cpu())

            # Calculate metrics for this batch
            # batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=4)
            # precision_totals += torch.tensor(batch_precision).to(device)
            # recall_totals += torch.tensor(batch_recall).to(device)
            # f1_totals += torch.tensor(batch_f1).to(device)
            batch_precision, batch_recall, batch_f1 = calculate_metrics(masks, preds, num_classes=cfg.model.classes)
            running_precision += batch_precision
            running_recall += batch_recall
            running_f1 += batch_f1
            num_batches += 1

            # Visualize some predictions
            if i == vis_batch:
                sample_idx = 0
                vis_fig = visualize_prediction(
                    images[sample_idx],
                    masks[sample_idx],
                    preds[sample_idx],
                    class_names,
                    sample_info=f"Validation - Epoch {epoch+1}, Final",
                )
                if cfg.use_wandb:
                    wandb.log(
                        {
                            f"val_pred_epoch_{epoch}": wandb.Image(vis_fig),
                            "epoch": epoch,
                        }
                    )
                plt.close(vis_fig)

            if is_slurm and i % log_freq == 0:
                # print(f"Validation Epoch {epoch+1} | Batch {i}/{total_batches} | Loss: {loss.item():.4f}")
                log.info(
                    f"Validation Epoch {epoch+1} | Batch {i}/{total_batches} | Loss: {loss.item():.4f}"
                )

    # all_preds = torch.cat(all_preds, dim=0)
    # all_masks = torch.cat(all_masks, dim=0)
    precision = running_precision / num_batches
    recall = running_recall / num_batches
    f1 = running_f1 / num_batches
    class_ious = []

    # Calculate IoU for each class
    for cls in range(cfg.model.classes):
        pred_cls = preds == cls
        target_cls = masks == cls
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        iou = intersection / (union + 1e-8)
        class_ious.append(iou.item())
    class_ious = np.array(class_ious)
    mean_iou = class_ious.mean()
    avg_val_loss = val_loss / batch_count

    # for true_cls in range(4):
    #     true_positive = ((masks == true_cls) & (preds == cls)).sum().item()
    #     confusion_matrix[true_cls, cls] += true_positive

    # Code for plotting confusion matrix

    # conf_matrix_fig = plt.figure(figsize=(10, 8))
    # conf_matrix_np = confusion_matrix.cpu().numpy()
    # plt.imshow(conf_matrix_np, cmap='Blues')
    # plt.colorbar()
    # plt.title('Validation Confusion Matrix')
    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.xticks(range(4), class_names, rotation=45)
    # plt.yticks(range(4), class_names)

    # # Add text annotations to the confusion matrix
    # for i in range(4):
    #     for j in range(4):
    #         # Normalize by row (true class)
    #         row_sum = conf_matrix_np[i].sum()
    #         percentage = (conf_matrix_np[i, j] / row_sum) * 100 if row_sum > 0 else 0
    #         plt.text(j, i, f'{percentage:.1f}%', ha='center', va='center',
    #                  color='white' if conf_matrix_np[i, j] > conf_matrix_np.max() / 2 else 'black')
    # plt.tight_layout()
    # Log validation metrics
    metrics = {
        "val_loss": avg_val_loss,
        "val_mean_iou": mean_iou,
        "val_mean_precision": precision.mean(),
        "val_mean_recall": recall.mean(),
        "val_mean_f1": f1.mean(),
        # "val_confusion_matrix": wandb.Image(conf_matrix_fig),
        "epoch": epoch,
    }
    # # Add per-class metrics
    for i, class_name in enumerate(class_names):
        metrics[f"val_iou_{class_name}"] = class_ious[i]
        metrics[f"val_precision_{class_name}"] = precision[i]
        metrics[f"val_recall_{class_name}"] = recall[i]
        metrics[f"val_f1_{class_name}"] = f1[i]

    if cfg.use_wandb:
        wandb.log(metrics)
    # plt.close(conf_matrix_fig)
    plt.close("all")

    torch.cuda.empty_cache()
    gc.collect()

    return avg_val_loss, mean_iou

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
    img_display = (img_display - img_display.min()) / (
        img_display.max() - img_display.min()
    )

    axes[0].imshow(img_display, cmap="gray")
    axes[0].set_title("Original SAR Image")
    axes[0].axis("off")

    # Define colormap for visualizing masks
    cmap = plt.cm.get_cmap("viridis", 4)

    # Ground truth mask
    axes[1].imshow(mask.cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Predicted mask
    axes[2].imshow(pred.cpu().numpy(), cmap=cmap, vmin=0, vmax=3)
    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # Add colorbar
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 3)),
        ax=axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.04,
    )
    cbar.set_ticks([0.375, 1.125, 1.875, 2.625])
    cbar.set_ticklabels(class_names)

    # Add sample info if provided
    if sample_info:
        plt.suptitle(f"Sample: {sample_info}", fontsize=14)

    plt.tight_layout()
    plt.close('all')
    return fig
