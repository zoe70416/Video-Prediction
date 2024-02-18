#imports 
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm
import numpy as np
import os
import logging
from unet_utils import *
from unet_model import UNET
from torchmetrics import JaccardIndex
from tqdm import tqdm

MODEL_PATH = "/scratch/ki2130/mask_models/unet_multi_class_segmentation.pth"
# MODEL_PATH = "/mnt/home/kinchoco/ceph/mask_models/unet_multi_class_segmentation.pth"
LOAD_MODEL = False
Learning_Rate = 1e-06
BATCH_SIZE = 16
NUM_WORKERS = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize Jaccard Index for evaluation
jaccard_evaluator = JaccardIndex(task="multiclass", num_classes=49).to(device)
train_root_dir = '/scratch/ki2130/train'
# train_root_dir = '/mnt/home/kinchoco/dlFinalProj/train'

def train_function(model, train_loader, criterion, optimizer, device):
    model.train()
    for batch in tqdm(train_loader, desc="Training"):
        inputs, masks = batch["image"].to(device), batch["mask"].to(device)

        # Forward pass
        outputs = model(inputs)
        # Compute loss
        loss = criterion(outputs, masks) 

        # Apply softmax along the channel dimension
        outputs_probs = torch.softmax(outputs, dim=1)

        # Apply argmax to get which channel has the highest softmax value among the 49 channels
        _, predicted_classes = torch.max(outputs_probs, dim=1)

        # Compute Jaccard Index
        jaccard_val = jaccard_evaluator(predicted_classes, masks)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item(), jaccard_val.item()
    

def main(args):
    logging.info(msg="Started unet_train.py. ")
    #Get Dataloader 
    logging.info(msg="Loading the data ")
    train_loader = get_trainval_dataloader(root_dir=train_root_dir, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    print('Data Loaded Successfully!')
    logging.info(msg="Defining the model, optimizer and loss function ")
    print('Defining the model, optimizer and loss function')
    # Initialize U-Net and other necessary components for multi-class segmentation
    num_classes = 49  # Including the background
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNET(in_channels=3, classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    jaccard = JaccardIndex(task="multiclass", num_classes=49)
    optimizer = optim.AdamW(unet.parameters(), lr=Learning_Rate)
    
    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH)
        unet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']+1
        LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    

    logging.info(msg="Starting the training loop ")
    
    LOSS_VALS = [] # Defining a list to store loss values after every epoch 
    JACCARD_VALS = [] # Defining a list to store jaccard values after every epoch
    #Training the model for each epoch. 
    for epoch in range(args.epochs):
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            loss_val, jaccard_val = train_function(unet, train_loader, criterion, optimizer, device)
            LOSS_VALS.append(np.mean(loss_val))
            JACCARD_VALS.append(np.mean(jaccard_val))
            torch.save({
            'model_state_dict': unet.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss_values': LOSS_VALS
        }, MODEL_PATH)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss_val}")
        plot_and_save_learning_curves(LOSS_VALS)
        plot_and_save_jaccard_idx(JACCARD_VALS)
    # Save the trained model for multi-class segmentation
    torch.save(unet.state_dict(), "/scratch/ki2130/mask_models/unet_multi_class_segmentation_v1.pth")
    # torch.save(unet.state_dict(), "/mnt/home/kinchoco/ceph/mask_models/unet_multi_class_segmentation_v1.pth")
    print("Epoch completed and model successfully saved!")   

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, help="number of epochs to train the model for")
    args = parser.parse_args()
    main(args)


