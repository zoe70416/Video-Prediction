import os 
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.io import read_image
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


def get_all_file_names(root_dir):
    # Get all file names in the root_dir, which is 'mask'
    folder = os.path.join(root_dir, 'mask')
    
    if not os.path.exists(folder) or not os.path.isdir(folder):
        print(f"Error: The 'mask' folder does not exist at {root_dir}")
        return []

    file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    return file_names

# Custom Dataset for multi-class segmentation
class CustomDataset(Dataset):
    def __init__(self, root_dir, mask_name_list, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.img_folder = "img"
        self.mask_folder = "mask"
        self.mask_name_list = mask_name_list

    def __len__(self):
        return len(self.mask_name_list)

    def __getitem__(self, idx):

        mask_name = self.mask_name_list[idx]
        img_name = mask_name.replace("mask", f"image")

        img_path = os.path.join(self.root_dir, self.img_folder, img_name)
        mask_path = os.path.join(self.root_dir, self.mask_folder, mask_name)

        try:
            image = Image.open(img_path).convert("RGB")
            mask = Image.open(mask_path)

            if self.transform:
                image = self.transform(image)
                # mask = torch.tensor(np.array(mask))  #convert to tensor while preserving the original value
                mask = torch.tensor(np.array(mask), dtype=torch.long)  # Convert to tensor with long data type
                
            return {"image": image, "mask": mask}
        except FileNotFoundError:
            print(f"File not found: {img_path} or {mask_path}")
            return None  # Skip this data point

# Transformations
data_transform = transforms.Compose([
    transforms.ToTensor(), # Convert the image to tensor and automatically normalizes the pixel value to [0,1]
])

def get_trainval_dataloader(root_dir, batch_size=4, num_workers=0, shuffle=True):
    """
    Get train and validation dataloader given the root directory of the dataset
    For train set: root_directory = 'train', in the train directory, there should be two folders: 'img' and 'mask'
    For train set: root_directory = 'val', same as val directory, there should be two folders: 'img' and 'mask'
    """
    mask_name_list = get_all_file_names(root_dir)
    dataset = CustomDataset(root_dir, mask_name_list, transform=data_transform)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)

def plot_and_save_learning_curves(train_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    # plot_path = os.path.join(os.path.dirname(model_path), 'training_loss_plot.png')
    plot_path = "/mnt/home/kinchoco/ceph/dl-project/mask_metrics/training_loss_plot.png"
    plt.savefig(plot_path)
    plt.close()

def plot_and_save_jaccard_idx(train_jaccard):
    epochs = range(1, len(train_jaccard) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_jaccard, label='Training Jaccard Index')
    plt.title('Training Jaccard Index Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot
    # plot_path = os.path.join(os.path.dirname(model_path), 'training_jaccard_plot.png')
    plot_path = "/mnt/home/kinchoco/ceph/dl-project/mask_metrics/training_jaccard_plot.png"
    plt.savefig(plot_path)
    plt.close()

# import os 
# import numpy as np
# import torch
# from PIL import Image
# from torchvision import transforms
# from torchvision.io import read_image
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset


# def get_all_file_names(root_dir):
#     # Get all file names in the root_dir, which is 'mask'
#     folder = os.path.join(root_dir, 'mask')
    
#     if not os.path.exists(folder) or not os.path.isdir(folder):
#         print(f"Error: The 'mask' folder does not exist at {root_dir}")
#         return []

#     file_names = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
#     return file_names

# # Custom Dataset for multi-class segmentation
# class CustomDataset(Dataset):
#     def __init__(self, root_dir, mask_name_list, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.img_folder = "img"
#         self.mask_folder = "mask"
#         self.mask_name_list = mask_name_list

#     def __len__(self):
#         return len(self.mask_name_list)

#     def __getitem__(self, idx):

#         mask_name = self.mask_name_list[idx]
#         img_name = mask_name.replace("mask", f"image")

#         img_path = os.path.join(self.root_dir, self.img_folder, img_name)
#         mask_path = os.path.join(self.root_dir, self.mask_folder, mask_name)

#         try:
#             image = Image.open(img_path).convert("RGB")
#             mask = Image.open(mask_path)

#             if self.transform:
#                 image = self.transform(image)
#                 # mask = torch.tensor(np.array(mask))  #convert to tensor while preserving the original value
#                 mask = torch.tensor(np.array(mask), dtype=torch.long)  # Convert to tensor with long data type
                
#             return {"image": image, "mask": mask}
#         except FileNotFoundError:
#             print(f"File not found: {img_path} or {mask_path}")
#             return None  # Skip this data point

# # Transformations
# data_transform = transforms.Compose([
#     transforms.ToTensor(), # Convert the image to tensor and automatically normalizes the pixel value to [0,1]
# ])

# def get_trainval_dataloader(root_dir, batch_size=4, num_workers=0, shuffle=True):
#     """
#     Get train and validation dataloader given the root directory of the dataset
#     For train set: root_directory = 'train', in the train directory, there should be two folders: 'img' and 'mask'
#     For train set: root_directory = 'val', same as val directory, there should be two folders: 'img' and 'mask'
#     """
#     mask_name_list = get_all_file_names(root_dir)
#     dataset = CustomDataset(root_dir, mask_name_list, transform=data_transform)
#     return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)