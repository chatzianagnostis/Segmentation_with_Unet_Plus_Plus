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
import optuna
from dataset import create_dataset
from utils import load_config

ssl._create_default_https_context = ssl._create_unverified_context


def objective(trial, config, optimize_lr=False, optimize_loss_params=False):
    if optimize_loss_params:
        alpha = trial.suggest_categorical('alpha', [0.5, 0.27, 0.3, 0.6, 0.4, 0.8, 0.2, 1.0])
        beta = 1.0 - alpha
        gamma = trial.suggest_categorical('gamma', [1.0, 0.75, 1.33, 1.5, 2.0, 2.5, 3.0, 0.5])
    else:
        alpha = 0.5
        beta = 0.5
        gamma = 1.0
    
    if optimize_lr:
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    else:
        lr = config["MODEL"]["LEARNING_RATE"]

    return train_and_evaluate(config, alpha, beta, gamma, lr)

def train_and_evaluate(config, alpha, beta, gamma,lr):
    # Directory paths and model hyperparameters
    DIR = config["DIR"]
    DATASET_PATH = config["DATASET_PATH"]
    TRAIN_PATH = os.path.join(DATASET_PATH, 'train')
    VALID_PATH = os.path.join(DATASET_PATH, 'valid')
    IMAGE_EXTENSION = config["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]
    IMAGE_WIDTH, IMAGE_HEIGHT = config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"]
    MULTICLASS_MODE = config["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config["MODEL"]["EXP_NAME"]
    ENCODER = config["MODEL"]["ENCODER"]
    ENCODER_WEIGHTS = config["MODEL"]["ENCODER_WEIGHTS"]
    CLASSES = config["MODEL"]["CLASSES"]
    ACTIVATION = config["MODEL"]["ACTIVATION"]
    BATCH_SIZE = config["MODEL"]["BATCH_SIZE"]
    EPOCHS =  config["MODEL"]["EPOCHS"]
    CHANNELS = config["MODEL"]["CHANNELS"]
    HORIZONTAL_FLIP = config["HORIZONTAL_FLIP"]
    VERTICAL_FLIP = config["VERTICAL_FLIP"]
    TRANSLATE = config["TRANSLATE"]
    MIXUP = config["MIXUP"]

    OUTPUT_FOLDER = os.path.join(DIR, 'runs', 'train', EXP_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    SAVE_MODEL_PATH = os.path.join(OUTPUT_FOLDER, 'model')
    if not os.path.exists(SAVE_MODEL_PATH):
        os.makedirs(SAVE_MODEL_PATH)

    test_transform = A.Compose([A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH)])
    train_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH),
        A.HorizontalFlip(p=HORIZONTAL_FLIP),
        A.VerticalFlip(p=VERTICAL_FLIP),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=TRANSLATE),
        A.MixUp(p=MIXUP)
    ])

    train_dataset = create_dataset(
        dataset_path=TRAIN_PATH,
        transform=train_transform,
        classes=CLASSES,
        extensions=EXTENSIONS
    )

    valid_dataset = create_dataset(
        dataset_path=VALID_PATH,
        transform=test_transform,
        classes=CLASSES,
        extensions=EXTENSIONS
    )

    train_set = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_set = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=CHANNELS,
        classes=len(CLASSES),
        activation=ACTIVATION,
    ).to(DEVICE)

    loss = smp.losses.TverskyLoss(mode=MULTICLASS_MODE, alpha=alpha, beta=beta, gamma=gamma)
    loss.__name__ = 'TverskyLoss'

    metrics = [
        smp.utils.metrics.PanopticQuality(smp.utils.metrics.SegmentationQuality(), smp.utils.metrics.RecognitionQuality()),
        smp.utils.metrics.SegmentationQuality(),
        smp.utils.metrics.RecognitionQuality()
    ]

    optimizer = optim.Adam(params=model.parameters(), lr=lr)

    train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss,
        metrics=metrics,
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

    best_pq = -float('inf')
    for epoch in range(EPOCHS):
        train_logs = train_epoch.run(train_set)
        valid_logs = valid_epoch.run(valid_set)
        pq = valid_logs['panoptic_quality']
        if pq > best_pq:
            best_pq = pq
            torch.save(model, f'{SAVE_MODEL_PATH}/best_model.pth')

    return best_pq

def main(config_file, search_best_loss_params, search_opt_lr):
    config = load_config(config_file)

    if search_best_loss_params:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, config, optimize_loss_params=True), n_trials=50)
        best_params = study.best_trial.params
        alpha = best_params['alpha']
        beta = 1.0 - alpha
        gamma = best_params['gamma']
        lr = config["MODEL"]["LEARNING_RATE"]

        print(f'Best hyperparameters for loss: {best_params}')
    elif search_opt_lr:
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, config, optimize_lr=True), n_trials=50)
        best_params = study.best_trial.params
        alpha = 0.5
        beta = 0.5
        gamma = 1.0
        lr = best_params['lr']
        print(f'Best learning rate: {best_params}')
    else:
        alpha = 0.5
        beta = 0.5
        gamma = 1.0
        lr = config["MODEL"]["LEARNING_RATE"]

    train_and_evaluate(config, alpha, beta, gamma, lr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Unet++ with custom dataset")
    parser.add_argument("--config", dest="config_file", default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--search_best_loss_params", action="store_true", help="Flag to perform hyperparameter search for Tversky loss")
    parser.add_argument("--search_opt_lr", action="store_true", help="Flag to perform learning rate optimization")
    args = parser.parse_args()
    main(args.config_file, args.search_best_loss_params, args.search_opt_lr)
