"""
This script get in the path of prednet's output folder: "prednet_output"
And then predict/output the mask of each image (.png) in the directory with .npy format

Please make sure the files in prednet's output folder are in the following format:
"0001.png", "0002.png", "0003.png", ..., "2000.png"
 
Example command: python3 unet_pred.py --dir hidden-practice2-sun5pm

By Zoe
"""

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
from tqdm import tqdm
import re

MODEL_PATH = "/scratch/ki2130/unet_multi_class_segmentation_v6.pth" #Change to the path where you saved the model
LOAD_MODEL = True 
num_classes = 49 # Including the background
save_output_array = '/scratch/ki2130/mask_preds/output_array.npy'

inference_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Function for inference on a single image
def infer_single_image(image_path, model):
    # Load and preprocess the input image
    input_image = Image.open(image_path).convert("RGB")
    input_tensor = inference_transform(input_image).unsqueeze(0)
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
    # Post-process the output (apply softmax, argmax, etc.)
    output_probs = torch.softmax(output, dim=1)
    _, predicted_classes = torch.max(output_probs, dim=1)
    return predicted_classes.squeeze().cpu().numpy()

def extract_numeric_part(filename):
    '''
    This function is used to extract the numeric part of the filename
    The file name should have this structure: "test_video{numeric part}_9y_11.png"
    '''
    match = re.search(r'video(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        return float('inf')

def get_all_file_names_sorted(root_dir):
    folder = os.path.join(root_dir)
    if not os.path.exists(folder) or not os.path.isdir(folder):
        print(f"Error: The  {folder} does not exist at {root_dir}")
        return []
    file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    sorted_filenames = sorted(file_names, key=extract_numeric_part)
    return sorted_filenames


def main(args):

    # Load the trained model
    # Initialize U-Net and other necessary components for multi-class segmentation
    num_classes = 49  
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    unet = UNET(in_channels=3, classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet.parameters(), lr=0.001)
    
    print('Defining the model, optimizer and loss function')
    # Loading a previous stored model from MODEL_PATH variable
    if LOAD_MODEL == True:
        # checkpoint = torch.load(MODEL_PATH)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        unet.load_state_dict(checkpoint)
        print("Model successfully loaded!")    

    unet.eval()
    # Get the list of all the files in the directory
    file_ls = get_all_file_names_sorted(args.dir)

    output = []
    print('Predicting the masks for each image in the directory')
    # Use tqdm to add a progress bar
    for file in tqdm(file_ls, desc='Predicting masks', unit='image'):
        image_path = os.path.join(args.dir, file)
        print(f"Predicting {image_path}")
        predicted_mask = infer_single_image(image_path, unet)
        output.append(predicted_mask)
    print('Finish predicting the masks for each image in the directory')

    output = np.array(output)
    # np.save('/mnt/home/kinchoco/ceph/hidden-practice2-mask/output_array.npy', output)
    # np.save('/mnt/home/kinchoco/ceph/dl-project/mask/output_array.npy', output)
    np.save(save_output_array, output)
    print("Output is saved as output_array.npy")
    print("The shape of the output array",output.shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=str, help="The directory of the prednet's output folder")
    args = parser.parse_args()
    main(args)
