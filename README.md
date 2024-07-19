# Segmentation_with_Unet_Plus_Plus

This repository represents a segmentation project training and evaluation for unett++ models

## Installation
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
2. Instead of using the [Segmentation Models Pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) repository, install [the forked version from this repository (which includes Focal Tversky Loss and Panoptic Quality metrics)](https://github.com/chatzianagnostis/segmentation_models.pytorch):
    ```bash
    git clone https://github.com/chatzianagnostis/segmentation_models.pytorch.git
    cd segmentation_models.pytorch
    pip install .


## Quick start
- Train a model
  To train a model, use:
  ```bash
  python train.py --config.yaml
  
- Evaluate the model
  To evaluate the model, use:
  ```bash
  python test.py --config.yaml

> config.yaml is for training and evaluation

## Configure data and model

### Dataset Format:
   
Dataset should have the following format:       

Ensure your dataset has the following structure:                                                                             

![Screenshot 2024-07-19 151656](https://github.com/user-attachments/assets/adb7d363-de71-4f68-9888-630c5da39f99)

>Ensure that images and their corresponding masks have identical filenames,                            
>with only the directory or extension differing to indicate whether the file is an image or a mask.    
                                                                                                       
>For example, if you have an image file named image1.png in the images directory,                      
>its corresponding mask should be named image1.png in the masks directory.


### Config.yaml

Modify the `config.yaml` file as needed:
- Paths: Specify the paths to your data and where to save models.
- Image Properties: Set image width and height (must be divisible by 32). Provide the values for reshape, not the actual width and height.
- Model Parameters: Configure model-specific parameters.
- Class Names: Ensure class-0 is the background.
- Augmentations: Define any data augmentations.

> We provide an example for `config.yaml`


