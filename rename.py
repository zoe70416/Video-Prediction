'''
This script takes in the folder name, and rename the individual frame images in the output directory.

Example Command: python3 rename.py Dataset_Student/train  train/img


Author: Zoe updated by 12.01.2023
'''

import numpy as np 
import sys
import os 
import shutil


def rename_image(folder_name, output_dir):
    """Function to take in a video folder and rename each image with the folder name and index"""
    for i, image in enumerate(os.listdir(folder_name)):
        if image.endswith(".png"):
            image_path = os.path.join(folder_name, image)
            new_name = f'{os.path.basename(folder_name)}_{image}'
            # os.rename(image_path, os.path.join(output_dir, new_name))
            shutil.copy(image_path, os.path.join(output_dir, new_name))


if __name__ == "__main__":
    # Check if the base folder is provided as a command-line argument
    # # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python3 rename.py <base_folder> <output_dir>")
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

        if os.path.exists(folder_path):

            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Assuming masks_array is defined or loaded from somewhere
            rename_image(folder_path,output_dir)
        else:
            print(f"Can't find {folder_path}")

