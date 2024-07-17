#########################################################################################################
# Dataset should have the following format:                                                             #
#                                                                                                       #
# - Dataset's dir folder                                                                                #
#     - train (folder)                                                                                  #
#         - images (folder)                                                                             #
#             - images (images for train)                                                               #
#         - labels (folder)                                                                             #
#             - masks (masks for train)                                                                 #
#     - valid (folder)                                                                                  #
#         - images (folder)                                                                             #
#             - images (images for validation)                                                          #
#         - labels (folder)                                                                             #
#             - masks (masks for validation)                                                            #
#     - test (folder)                                                                                   #
#         - images (folder)                                                                             #
#             - images (images for test)                                                                #
#         - labels (folder)                                                                             #
#             - masks (masks for test)                                                                  #
#                                                                                                       #
# Ensure that images and their corresponding masks have identical filenames,                            #
# with only the directory or extension differing to indicate whether the file is an image or a mask.    #
#                                                                                                       #
# For example, if you have an image file named image1.png in the images directory,                      #
# its corresponding mask should be named image1.png in the masks directory.                             #
#########################################################################################################

import os
import cv2
import torch
import glob
from torch.utils.data.dataset import Dataset


class MultiClassSegDataset(Dataset):
    """Custom dataset class for multi-class segmentation."""


    def __init__(self, image_list, mask_list, classes=None, transform=None):
        """
        Initialize the dataset.

        Args:
            image_list (list): List of image file paths.
            mask_list (list): List of mask file paths corresponding to the images.
            classes (list): List of class names.
            transform (callable, optional): Optional transform to be applied to the images and masks.
        """
        self.image_list = image_list
        self.mask_list = mask_list
        self.classes = classes
        self.transform = transform


    def __getitem__(self, idx):
        """
        Get an item (image and mask) from the dataset by index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and its corresponding mask.
        """
        # Load image and mask
        image_name = self.image_list[idx]
        mask_name = self.mask_list[idx]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)

        # Ensure mask has only one channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]

        # Apply transformation if available
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        # Convert to PyTorch tensors
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)

        # Permute dimensions and normalize image
        image = image.permute(2, 0, 1).float() / 255

        # Convert mask to long tensor and add batch dimension
        mask = mask.long().unsqueeze(0)

        return image, mask  


    def __len__(self):
        """Get the length of the dataset."""
        return len(self.image_list)


def create_dataset(dataset_path, transform, classes, extensions):
    """
    Create a MultiClassSegDataset given the dataset path, transformation function, classes, and file extensions.

    Args:
        dataset_path (str): Path to the dataset directory.
        transform (callable): Transformation function to be applied to the images and masks.
        classes (list): List of class names.
        extensions (list): List of file extensions for images and masks.

    Returns:
        dataset: Instance of MultiClassSegDataset containing the dataset.
    """
    # Define the directories for images and masks
    imgs_dir = os.path.join(dataset_path, 'images') 
    masks_dir = os.path.join(dataset_path, 'labels')
    
    # Get the paths of all images and masks in the directories
    img_paths = sorted(glob.glob(os.path.join(imgs_dir, f"*{extensions[0]}")))
    mask_paths = sorted(glob.glob(os.path.join(masks_dir, f"*{extensions[1]}")))

    # Create the dataset using the MultiClassSegDataset class
    dataset = MultiClassSegDataset(img_paths, mask_paths, classes=classes, transform=transform)
    
    return dataset
