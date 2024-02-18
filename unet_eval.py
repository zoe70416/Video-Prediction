"""
This script performs evaluation on the val dataset. 

Example command: python3 unet_eval.py 

"""

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

MODEL_PATH = "/mnt/home/kinchoco/ceph/dl-project/mask/unet_multi_class_segmentation_v7.pth"
LOAD_MODEL = True
Learning_Rate = 1e-06
val_dir = "/mnt/home/kinchoco/ceph/val"

def main():
    #Get Dataloader 
    logging.info(msg="Loading the data ")
    # val_loader = get_trainval_dataloader(root_dir=val_dir, batch_size=4, num_workers=0, shuffle=True)
    val_loader = get_trainval_dataloader(root_dir=val_dir, batch_size=64, num_workers=0, shuffle=True)
    print('Data Loaded Successfully!')
    logging.info(msg="Defining the model, optimizer and loss function ")
    print('Defining the model, optimizer and loss function')
    # Load the trained model
    # Initialize U-Net and other necessary components for multi-class segmentation
    num_classes = 49  # Including the background
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet = UNET(in_channels=3, classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=Learning_Rate)

    # Initialize Jaccard Index for evaluation
    # jaccard_evaluator = JaccardIndex(task="multiclass", num_classes=49)
    jaccard_evaluator = JaccardIndex(task="multiclass", num_classes=49).to(device)
    
    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        checkpoint = torch.load(MODEL_PATH,map_location=torch.device('cpu'))
        # unet.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optim_state_dict'])
        # print(checkpoint.keys())
        # epoch = checkpoint['epoch']+1
        # LOSS_VALS = checkpoint['loss_values']
        print("Model successfully loaded!")    
    
    logging.info(msg="Starting the evaluation ")
    print('Starting the evaluation')
    unet.eval()


    # Lists to store per-batch losses and Jaccard Indices
    val_losses = []
    val_jaccards = []

    # Evaluate on the validation dataset
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            inputs, masks = batch["image"].to(device), batch["mask"].to(device)

            # Forward pass
            outputs = unet(inputs)

            # Apply softmax along the channel dimension
            outputs_probs = torch.softmax(outputs, dim=1)

            # Apply argmax to get which channel has the highest softmax value among the 49 channels
            _, predicted_classes = torch.max(outputs_probs, dim=1)

            print(outputs.shape)
            print(predicted_classes.shape)

            # Compute loss
            loss = criterion(outputs, masks)
            val_losses.append(loss.item())

            # Compute Jaccard Index
            jaccard_val = jaccard_evaluator(predicted_classes, masks)
            val_jaccards.append(jaccard_val.item())
    
    # Calculate average loss and Jaccard Index
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_val_jaccard = sum(val_jaccards) / len(val_jaccards)

    print(f"Validation Loss: {avg_val_loss}, Validation Jaccard Index: {avg_val_jaccard}")

if __name__ == "__main__":
    main()