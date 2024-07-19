import argparse
import os
import torch
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from dataset import create_dataset
from utils import load_config, PanopticQuality


def main(config_file):
    # Read from config.yaml====================================================================
    config = load_config(config_file)
    # Directory paths:
    DIR = config["DIR"]
    DATASET_PATH = config["DATASET_PATH"]
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    VALID_PATH = os.path.join(DATASET_PATH, 'valid')
    IMAGE_EXTENSION = config["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]
    IMAGE_WIDTH,IMAGE_HEIGHT = config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"]
    # Model hyperparameters:
    MULTICLASS_MODE = config["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config["MODEL"]["EXP_NAME"]
    ENCODER = config["MODEL"]["ENCODER"]
    ENCODER_WEIGHTS = config["MODEL"]["ENCODER_WEIGHTS"]
    CLASSES = config["MODEL"]["CLASSES"]
    ACTIVATION = config["MODEL"]["ACTIVATION"]
    BATCH_SIZE = config["MODEL"]["BATCH_SIZE"]
    LR0 = config["MODEL"]["LR0"]
    LRF = config["MODEL"]["LRF"]
    EPOCHS = config["MODEL"]["EPOCHS"]
    CHANNELS = config["MODEL"]["CHANNELS"]
    # Augmentations
    HORIZONTAL_FLIP = config["HORIZONTAL_FLIP"]
    VERTICAL_FLIP = config["VERTICAL_FLIP"]
    TRANSLATE = config["TRANSLATE"]
    MIXUP = config["MIXUP"]

    # Create folder for train resuts =========================================================
    OUTPUT_FOLDER = os.path.join(DIR, 'runs','train', EXP_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    SAVE_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'model')
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    # Define tranforms using Albumations =====================================================
    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH)
    ])

    train_transform = A.Compose(
        [
            A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
            A.HorizontalFlip(p=HORIZONTAL_FLIP),
            A.VerticalFlip(p=VERTICAL_FLIP),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=TRANSLATE),
            A.MixUp(p=MIXUP)
        ]
    )

    # Create datasets and define dataloaders =================================================
    train_dataset = create_dataset(
        dataset_path=TRAIN_PATH,
        transform = train_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    valid_dataset = create_dataset(
        dataset_path=VALID_PATH,
        transform = test_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    train_set = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    valid_set = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    # Initiate UNet++ Model ==================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.get_device_properties(0))

    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=CHANNELS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    ).to(DEVICE)

    torch.save(model, f'{SAVE_MODEL_PATH}/init_model.pth')

    #summary(model, input_size=(3, 1376, 800), device=DEVICE.type)
    print(f'Dataset stats:\n Training Set: {len(train_dataset)} images\n Validation Set: {len(valid_dataset)} images')

    # Define Loss and Metrics to Monitor =====================================================
    loss = smp.losses.TverskyLoss(mode=MULTICLASS_MODE)
    loss.__name__ = 'TverskyLoss'

    metrics = [
        smp.utils.metrics.PanopticQuality(smp.utils.metrics.SegmentationQuality(),smp.utils.metrics.RecognitionQuality()),
        smp.utils.metrics.SegmentationQuality(),
        smp.utils.metrics.RecognitionQuality()
    ]

    # OneCycleLR parameters
    lr0 = LR0  # initial learning rate
    lrf = LRF  # final learning rate factor

    # Define the optimizer with the initial learning rate
    optimizer = optim.Adam(params=model.parameters(), lr=lr0)

    # Define the OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr0, steps_per_epoch=len(train_set), epochs=EPOCHS, final_div_factor=1/lrf)

    # Define epochs ==========================================================================
    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss,
        metrics= metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, 
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )

    # Train model ============================================================================
    best_loss = float('inf')
    best_pq = -float('inf')
    best_sq = -float('inf')
    best_rq = -float('inf')
    writer = SummaryWriter(OUTPUT_FOLDER)

    print('Starting Training ...')

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch+1}/{EPOCHS}')
        
        train_logs = train_epoch.run(train_set)
        valid_logs = valid_epoch.run(valid_set)
        
        # Log training and validation loss
        writer.add_scalar('TverskyLoss/train', train_logs['TverskyLoss'], epoch)
        writer.add_scalar('TverskyLoss/val', valid_logs['TverskyLoss'], epoch)
        
        # Log the learning rate
        current_lr = scheduler.get_last_lr()[0]
        writer.add_scalar('Learning Rate', current_lr, epoch)

        # Log the PQ metric
        writer.add_scalar('PQ/train', train_logs['panoptic_quality'], epoch)
        writer.add_scalar('PQ/val', valid_logs['panoptic_quality'], epoch)
        # Log the SQ metric
        writer.add_scalar('SQ/train', train_logs['segmentation_quality'], epoch)
        writer.add_scalar('SQ/val', valid_logs['segmentation_quality'], epoch)
        # Log the RQ metric
        writer.add_scalar('RQ/train', train_logs['recognition_quality'], epoch)
        writer.add_scalar('RQ/val', valid_logs['recognition_quality'], epoch)


        # Save the model after each epoch
        torch.save(model, f'{SAVE_MODEL_PATH}/last_model.pth')


        # Save the model with the best loss
        if valid_logs['TverskyLoss'] < best_loss:
            best_loss = valid_logs['TverskyLoss']
            torch.save(model, f'{SAVE_MODEL_PATH}/best_loss_model.pth')
        # Save the model with the best panoptic qualty
        if valid_logs['panoptic_quality'] > best_pq:
            best_pq = valid_logs['panoptic_quality']
            torch.save(model, f'{SAVE_MODEL_PATH}/best_pq_model.pth')
        # Save the model with the best segmentation qualty
        if valid_logs['segmentation_quality'] > best_sq:
            best_sq = valid_logs['segmentation_quality']
            torch.save(model, f'{SAVE_MODEL_PATH}/best_sq_model.pth')
        # Save the model with the best recognition qualty
        if valid_logs['recognition_quality'] > best_rq:
            best_rq = valid_logs['recognition_quality']
            torch.save(model, f'{SAVE_MODEL_PATH}/best_rq_model.pth')


        # Step the scheduler
        scheduler.step()

    writer.close()

    print(f'TverskyLoss loss from best model in validation set: {best_loss}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet++ with custom dataset")
    parser.add_argument("--config", dest="config_file", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config_file)
