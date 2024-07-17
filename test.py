import argparse
import os
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torchsummary import summary

import albumentations as A
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import utils

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import create_dataset
from utils import load_config, seg2bbox, process_batch, create_single_masks, iou, conf_mat_bboxes, conf_mat_pixel, compine_conf_matrix, calculate_tp_fp_fn, calculate_metrics, visualize_segments, visualize_instances


def main(config_file):
    # Read from config.yaml===================================================================
    config = load_config(config_file)
    # Directory paths:
    DIR = config["DIR"]
    DATASET_PATH = config["DATASET_PATH"]
    TEST_PATH = os.path.join(DATASET_PATH, 'test')
    IMAGE_EXTENSION = config["IMAGE_EXTENSION"]
    MAKS_EXTENSION = config["MAKS_EXTENSION"]
    EXTENSIONS = [IMAGE_EXTENSION, MAKS_EXTENSION]
    IMAGE_WIDTH,IMAGE_HEIGHT = config["IMAGE_WIDTH"], config["IMAGE_HEIGHT"]

    # Model hyperparameters:
    MULTICLASS_MODE = config["MODEL"]["MULTICLASS_MODE"]
    EXP_NAME = config["MODEL"]["EXP_NAME"]
    CLASSES = config["MODEL"]["CLASSES"]
    MODEL_PATH = os.path.join(DIR,'runs','train', EXP_NAME, 'model', 'best_model.pth')
    
    # Create folder for train resuts =========================================================
    OUTPUT_FOLDER = os.path.join(DIR, 'runs','train', EXP_NAME)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    DETECT_FOLDER = os.path.join(OUTPUT_FOLDER,'detect')
    if not os.path.exists(DETECT_FOLDER):
        os.makedirs(DETECT_FOLDER)
    if not os.path.exists(os.path.join(DETECT_FOLDER,'segments')):
        os.makedirs(os.path.join(DETECT_FOLDER,'segments'))
    if not os.path.exists(os.path.join(DETECT_FOLDER,'bboxes')):
        os.makedirs(os.path.join(DETECT_FOLDER,'bboxes'))
        

    # Define tranforms using Albumations =====================================================
    test_transform = A.Compose([
        A.Resize(IMAGE_HEIGHT, IMAGE_WIDTH)
    ])

    # Create datasets and define dataloaders =================================================
    test_dataset = create_dataset(
        dataset_path=TEST_PATH,
        transform = test_transform,
        classes = CLASSES,
        extensions = EXTENSIONS
    )

    test_set = torch.utils.data.DataLoader(test_dataset, batch_size= 1, shuffle=True, sampler=None,
                batch_sampler=None, num_workers=0, collate_fn=None,
                pin_memory=False, drop_last=False, timeout=0,
                worker_init_fn=None)
        
    # Load model =============================================================================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(MODEL_PATH, map_location=torch.device('cuda'))
    #summary(model, input_size=(3, 1376, 800), device=DEVICE.type)
    print(f"Loading model : {MODEL_PATH}")

    # Define Loss and Metrics to Monitor =====================================================
    loss = smp.losses.TverskyLoss(mode=MULTICLASS_MODE, alpha=0.6, beta=0.4)
    loss.__name__ = 'FocalTverskyLoss'

    metrics=[] #TODO

    # Test Epoch =============================================================================
    test_epoch = smp.utils.train.ValidEpoch(
    model=model,
    loss=loss,
    metrics=metrics,
    device=DEVICE,
    )
    print('Testing ...')
    logs = test_epoch.run(test_set)
    print(logs)
    
    def predict_segments(model, image, batch_size=1, class_num=9, conf_thres=0.001, iou_thres=0.6 ):
        """
        Object segmentation model that returns the masks and classes for an image.
        model: Model for segmentation
        img: image
        :return: List of masks and class labels for image
        """
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float() / 255
        x_tensor = image.to('cuda').unsqueeze(0)
        pr_mask = model(x_tensor)
        m = nn.Softmax(dim=1)
        pr_probs = m(pr_mask)

        results = process_batch(pr_probs, batch_size, class_num, conf_thres, iou_thres)
        
        pr_masks=[]
        pr_classes=[]
        pr_confidences=[]
        for result in results:
            mask = result['mask']
            mask_resized = cv2.resize(mask.astype(np.float32), (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST).astype(np.bool_)
            class_id = result['class_id']
            confidence = result['confidence']
            pr_masks.append(mask_resized)
            pr_classes.append(class_id)
            pr_confidences.append(confidence)

        # Create the single_channel_mask
        single_channel_mask = create_single_masks(pr_masks, pr_classes, (IMAGE_HEIGHT, IMAGE_WIDTH))

        return pr_masks, pr_classes, pr_confidences, single_channel_mask

    conf_matrix_pixel = []
    conf_matrix_instance = []
    total_iou_matrix = []
    class_dict = {i: CLASSES[i] for i in range(len(CLASSES))}
    
    for i in range(len(test_dataset)):
        start_time = time.time()
        image_vis = test_dataset[i][0].permute(1,2,0)
        image_vis = image_vis.numpy()*255
        image_vis = image_vis.astype('uint8')
        image, gt_mask = test_dataset[i]
        gt_mask = (gt_mask.squeeze().cpu().numpy().round())
        gt_bboxes, gt_classes = seg2bbox(gt_mask)

        pr_mask, pr_label, pr_conf, single_channel_mask = predict_segments(model, image_vis)
        pr_bboxes, pr_classes = seg2bbox((single_channel_mask))

        conf_matrix_pixel.append(conf_mat_pixel(single_channel_mask, gt_mask))
        conf_matrix_instance.append(conf_mat_bboxes(pr_bboxes, pr_classes, gt_bboxes, gt_classes))

        # Calculate IoU matrix for the current image
        iou_matrix = np.zeros((len(pr_bboxes), len(gt_bboxes)))
        for p_idx, pr_bbox in enumerate(pr_bboxes):
            for g_idx, gt_bbox in enumerate(gt_bboxes):
                iou_matrix[p_idx, g_idx] = iou(pr_bbox, gt_bbox)
        
        total_iou_matrix.append(iou_matrix)

        save_path_pixel = f'{DETECT_FOLDER}/segments/image_{i}.png'
        visualize_segments(image_vis, pr_mask, pr_label, pr_conf, class_dict, save_path_pixel)
        pr_bboxes, pr_classes = seg2bbox((np.transpose(single_channel_mask)))
        save_path_instance = f'{DETECT_FOLDER}/bboxes/image_{i}.png'
        visualize_instances(image_vis, pr_bboxes, pr_classes, class_dict, save_path_instance)


    combined_conf_matrix_pixel=compine_conf_matrix(conf_matrix_pixel)
    combined_conf_matrix_instance=compine_conf_matrix(conf_matrix_instance)

    # Pixel Wise ==================================
    # Calculate and Print metric results
    true_positive, false_positive, false_negative = calculate_tp_fp_fn(combined_conf_matrix_pixel)
    precision, recall, f1_score = calculate_metrics(true_positive, false_positive, false_negative)
    # Display the Precision, Recall, and F1 Score for each class
    print(f"Pixel wise metrics")
    print(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}")
    for class_name, precision, recall, f1 in zip(CLASSES, precision, recall, f1_score):
        print(f"{class_name:<15}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}")

    # Plot conf
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_conf_matrix_pixel, annot=True, fmt="d", cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    filename = os.path.join(DIR,'runs','train', EXP_NAME,'confusion_martix_pixels.png')
    plt.savefig(filename)
    plt.show()
    plt.close()

    # Instance Wise ==================================
    # Calculate and Print metric results
    true_positive, false_positive, false_negative = calculate_tp_fp_fn(combined_conf_matrix_instance)
    precision, recall, f1_score = calculate_metrics(true_positive, false_positive, false_negative) 
    # Display the Precision, Recall, and F1 Score for each class
    print(f"Instance wise metrics")
    print(f"{'Class':<15}{'Precision':<15}{'Recall':<15}{'F1 Score':<15}")
    for class_name, precision, recall, f1 in zip(CLASSES, precision, recall, f1_score):
        print(f"{class_name:<15}{precision:<15.2f}{recall:<15.2f}{f1:<15.2f}")

    # Plot conf
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_conf_matrix_instance, annot=True, fmt="d", cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    filename = os.path.join(DIR,'runs','train', EXP_NAME,'confusion_martix_instance.png')
    plt.savefig(filename)
    plt.show()
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Unet++ with custom dataset")
    parser.add_argument("--config", dest="config_file", default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config_file)