from PIL import Image
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torchvision
from dataset_class import GlacierSegDataset
import segmentation_models_pytorch as smp
import tqdm
import torch.nn as nn
import torch.optim as optim

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    print("Starting training for one epoch...")

    pbar = tqdm.tqdm(enumerate(loader), total=len(loader))
    for i, (images, masks) in pbar:

        images = images.to(device)  # Shape: [batch_size, 1, height, width]
        masks = masks.to(device).squeeze(1).long()  # Shape: [batch_size, height, width]
        
        optimizer.zero_grad()
        outputs = model(images)
        #print(f"for batch {i}, input shape: {images.shape} outputs shape: {outputs.shape}, masks shape: {masks.shape}")
        loss = criterion(outputs, masks)
        pbar.set_postfix(loss=loss.item())
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print("Finished training for one epoch.")

    return epoch_loss / len(loader)


def main():


    # # image size
    path_to_image = '/gws/nopw/j04/iecdt/amorgan/data_copy/sar_images/train/Crane_2002-11-18_ERS_20_3_195__93_102_0_0_0.png'
    image = Image.open(path_to_image)
    print(image.size)

    # print some of the values from the matrix itself
    #plot this image with the labels
    import numpy as np
    image = Image.open(path_to_image)
    image = np.array(image)
    




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

    image, label = train_dataset_mask[0]  # Get first sample

    print(f"Image shape: {image.shape}") 
    print(f"Label shape: {label.shape}") # label can be either mask or front depending on the label_type


    model = smp.Unet(  
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=4,                      # model output channels (number of classes in your dataset), # class ids area=0, stone=1, glacier=2, ocean with ice melange=3
    )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    train_loader = DataLoader(train_dataset_mask, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


    print("Starting training loop...")

    epoch_losses = []  # To store loss values for visualization
    for epoch in range(1):
        print(f"Epoch {epoch + 1} starting...")
    
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        epoch_losses.append(train_loss)  # Save loss for visualization
    
        print(f"Epoch {epoch+1}, Loss: {train_loss:.4f}")
    print("Training loop completed.")

    # Save the model
    torch.save(model.state_dict(), "amy_trained_model.pth")

if __name__ == "__main__":
    print("about to go!")
    main()