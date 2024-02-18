'''
This script takes in the mask.npy file and convert them to png file in the output directory.

Example Command: python3 data_prep.py Dataset_Student/train mask 

Author: Zoe updated  updated by 12.01.2023
'''

#Use the fastaiV2 version
from fastai.vision.all import *
from zipfile import ZipFile
import numpy as np 
import sys
import os 


def get_mask(folder_name, output_dir):
    """Function to get the mask from the numpy file"""
    # # Assuming your numpy array is named masks_array
    # Assuming your numpy array is named masks_array
    masks_array = np.load(os.path.join(folder_name, 'mask.npy'))
    for i, mask_array in enumerate(masks_array):
        # Convert the numpy array to a PIL Image
        mask_pil = PILMask.create(mask_array)

        # Save the PIL Image to a file in the specified output directory
        output_path = os.path.join(output_dir, f'{os.path.basename(folder_name)}_mask_{i}.png')
        mask_pil.save(output_path)



if __name__ == "__main__":
    # Check if the base folder is provided as a command-line argument
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 data_prep.py <base_folder> <output_dir>")
        sys.exit(1)

    # Get the base folder from the command-line argument
    base_folder = sys.argv[1]
    output_dir = sys.argv[2]

    # Loop through folders in the specified directory
    for folder_name in os.listdir(base_folder):
        # Skip .DS_Store file
        if folder_name == '.DS_Store':
            continue

        folder_path = os.path.join(base_folder, folder_name)

        # Check if the folder contains 'mask.npy'
        mask_file_path = os.path.join(folder_path, 'mask.npy')
        if os.path.exists(mask_file_path):
            # Load masks_array only if 'mask.npy' is present
            masks_array = np.load(mask_file_path)

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Assuming masks_array is defined or loaded from somewhere
            get_mask(folder_path,output_dir)
        else:
            print(f"Warning: 'mask.npy' not found in {folder_path}")