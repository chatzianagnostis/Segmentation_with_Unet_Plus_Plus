import yaml
import numpy as np
import torch
from skimage.measure import label, regionprops
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import cv2


def load_config(config_file):
    """
    Load configuration settings from a YAML file.
    
    Parameters:
    config_file (str): Path to the YAML configuration file.
    
    Returns:
    dict: Configuration settings.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config

def seg2bbox(segm):
    """
    Convert segmentation masks to bounding boxes and corresponding classes.
    
    Parameters:
    segm (numpy.ndarray): Segmentation mask.
    
    Returns:
    tuple: Tuple containing lists of bounding boxes and corresponding classes.
    """
    labels = label(segm)
    props = regionprops(labels)
    bboxes = []
    classes = []
    
    for prop in props:
        x1, y1, x2, y2 = prop.bbox
        bboxes.append([x1, y1, x2, y2])
        class_in_bbox = np.argmax(np.bincount(segm[x1:x2, y1:y2].flatten())[1:])+1
        classes.append(class_in_bbox)
    return bboxes, classes

# =====================Prediction Process=====================

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

def iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1_max = max(box1[0], box2[0])
    y1_max = max(box1[1], box2[1])
    x2_min = min(box1[2], box2[2])
    y2_min = min(box1[3], box2[3])
    
    inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = inter_area / float(box1_area + box2_area - inter_area)
    
    return iou

def apply_nms(masks, iou_threshold=0.5):
    if len(masks) == 0:
        return []
    
    # Sort the masks by confidence score
    masks = sorted(masks, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    while masks:
        # Take the mask with the highest confidence and remove it from the list
        current_mask = masks.pop(0)
        keep.append(current_mask)
        
        # Compare the IoU of the current mask with the rest
        masks = [mask for mask in masks if compute_iou(current_mask['mask'], mask['mask']) < iou_threshold]
    
    return keep

def process_batch(pr_probs, batch_size, class_num, conf_thres=0.001, iou_thres=0.6):
    results = []
    #pdb.set_trace()
    for i in range(batch_size):
        image_pr_masks = []
    
        # Get the class with the highest probability for each pixel
        class_probs, class_indices = torch.max(pr_probs[i], dim=0)
        
        #pdb.set_trace()
        for class_idx in range(1, class_num):
            # Get the mask for the current class
            class_mask = (class_indices == class_idx)
            if class_mask.sum() == 0:
                continue  # Skip if no pixels are assigned to this class

            # Calculate the confidence for this class
            # for item in torch.sort(class_probs[class_mask]):
            #     print(item)
            class_confidence = class_probs[class_mask].mean().item()
            # Apply the confidence threshold filter
            if class_confidence >= conf_thres:
                # Save the mask and class information
                image_pr_masks.append({
                    "mask": class_mask.cpu().numpy(),
                    "confidence": class_confidence,
                    "class_id": class_idx
                })

        # Apply NMS to the image_pr_masks
        image_pr_masks = apply_nms(image_pr_masks, iou_thres)
        
        results.extend(image_pr_masks)

    return results

def create_single_masks(pr_masks, pr_classes, image_shape):
    single_channel_mask = np.zeros(image_shape, dtype=np.uint8)
    
    # Iterate through each mask and apply the corresponding class label
    for mask, class_id in zip(pr_masks, pr_classes):
        if mask.shape != single_channel_mask.shape:
            raise ValueError(f"Shape mismatch: mask shape {mask.shape} does not match single_channel_mask shape {single_channel_mask.shape}")
        single_channel_mask[mask > 0] = class_id  # Ensure we apply the class_id to the correct positions

    #resized_mask = cv2.resize(single_channel_mask, (1152, 2048), interpolation=cv2.INTER_NEAREST)
    return single_channel_mask


# =====================Confusion Matrixes=====================

def conf_mat_bboxes(pr_bboxes, pr_bclasses, gt_bboxes, gt_classes, iou_threshold=0.1, num_classes=9):
    """Compute confusion matrix for bounding boxes."""
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # List to keep track of matched predictions
    matched_pr = []

    # Iterate over all ground truth boxes
    for i, gt_bbox in enumerate(gt_bboxes):
        gt_class = gt_classes[i]
        
        best_iou = 0
        best_pred_idx = -1
        
        # Find the predicted box with the highest IoU
        for j, pr_bbox in enumerate(pr_bboxes):
            iou_score = iou(gt_bbox, pr_bbox)
            if iou_score > best_iou:
                best_iou = iou_score
                best_pred_idx = j
        
        # Determine the predicted class
        #(best_iou)
        if best_iou >= iou_threshold:
            pr_class = pr_bclasses[best_pred_idx]
            cm[gt_class, pr_class] += 1
            matched_pr.append(best_pred_idx)
        else:
            # No match found, count as a false negative
            cm[gt_class, 0] += 1

    # Any remaining predictions are false positives
    for idx, pr_class in enumerate(pr_bclasses):
        if idx not in matched_pr:
            cm[0, pr_class] += 1

    return cm

def conf_mat_pixel(pred_mask,gt_mask):
    # Flatten the masks
    pred_mask_flat = pred_mask.flatten()
    gt_mask_flat = gt_mask.flatten()
    # Compute the confusion matrix
    cm = confusion_matrix(gt_mask_flat, pred_mask_flat, labels=np.arange(9))
    return cm

def compine_conf_matrix(conf_matrixes):
    combined_conf_matrix = np.zeros_like(conf_matrixes[0])
    for conf_matrix in conf_matrixes:
        combined_conf_matrix += conf_matrix
    return combined_conf_matrix

def calculate_metrics(TP, FP, FN):
    # Calculate Precision, Recall, and F1 Score for each class
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Handle the case where Precision + Recall == 0 to avoid NaNs
    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)
    f1_score = np.nan_to_num(f1_score)
    return precision, recall, f1_score

def calculate_tp_fp_fn(conf_matrix):
    # Calculate TP, FP, FN for each class
    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    return TP, FP, FN

# =====================Panoptic Segmentation=====================

class PanopticQuality(smp.utils.base.Metric):
    def __init__(self):
        super(PanopticQuality, self).__init__()
        self.reset()

    def reset(self):
        self.true_positives = []
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, y_pred, y_true):
        # Assuming y_pred and y_true are binary masks
        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()
        
        iou = tp / (tp + fp + fn + 1e-6)  # Add small value to avoid division by zero
        self.true_positives.append(iou)
        self.false_positives += fp
        self.false_negatives += fn

    def compute(self):
        sq = compute_sq(self.true_positives)
        rq = compute_rq(sum(self.true_positives), self.false_positives, self.false_negatives)
        pq = compute_pq(sq, rq)
        return pq

    def forward(self, y_pred, y_true):
        self.update(y_pred, y_true)
        pq = self.compute()
        return torch.tensor(pq, device=y_pred.device)


def compute_sq(true_positives):
    """
    Calculate Segmentation Quality (SQ).
    
    Parameters:
    true_positives (list): A list with IoU values for True Positives (TP).
    
    Returns:
    float: Segmentation Quality (SQ).
    """
    if not true_positives:
        return 0.0
    return sum(true_positives) / len(true_positives)

def compute_rq(true_positives, false_positives, false_negatives):
    """
    Calculate Recognition Quality (RQ).
    
    Parameters:
    true_positives (int): Number of True Positives (TP).
    false_positives (int):Number of  False Positives (FP).
    false_negatives (int): Number of  False Negatives (FN).
    
    Returns:
    float: Recognition Quality (RQ).
    """
    if true_positives == 0:
        return 0.0
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def compute_pq(sq, rq):
    """
    Calculate Panoptic Quality (PQ).
    
    Parameters:
    sq (float): Segmentation Quality (SQ).
    rq (float): Recognition Quality (RQ).
    
    Returns:
    float: Panoptic Quality (PQ).
    """
    return sq * rq

def compute_tp_iou(pr_bboxes,pr_classes,gt_bboxes,gt_classes):
    ious = []
    for pred in zip(pr_bboxes,pr_classes):
            for gt in zip(gt_bboxes,gt_classes):
                io = iou(pred[0], gt[0])
                if io > 0.5 and pred[1]==gt[1]:
                    ious.append(io)
    return ious

# =====================Visualizations=====================

def visualize_segments(original_img, pr_masks, pr_classes, pr_confidences,class_dict, output_path):
    """
    Visualize predictions on the image and save to file.
    
    :param original_img: Original image without padding
    :param pr_masks: List of predicted masks
    :param pr_classes: List of predicted class IDs corresponding to the masks
    :param pr_confidences: List of predicted confidences corresponding to the masks
    :param output_path: Path to save the output image
    """

    # Fixed color map for each class
    colors = {
        1: (255, 0, 0),     # Red
        2: (0, 255, 0),     # Green
        3: (0, 0, 255),     # Blue
        4: (255, 255, 0),   # Cyan
        5: (255, 0, 255),   # Magenta
        6: (0, 255, 255),   # Yellow
        7: (128, 0, 128),   # Purple
        8: (0, 128, 128)    # Teal
    }
    
    # Create a copy of the original image to draw the predictions
    img_copy = original_img.copy()

    # Iterate through each mask, its corresponding class ID, and confidence
    for mask, class_id, confidence in zip(pr_masks, pr_classes, pr_confidences):
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            color = colors[class_id]
            cv2.drawContours(img_copy, [contour], -1, color, 2)
            x, y, w, h = cv2.boundingRect(contour)
            label = f"{class_dict[class_id]}: {confidence:.2f}"
            cv2.putText(img_copy, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # Save the image using OpenCV
    cv2.imwrite(output_path, img_copy)
    #print(f"Image saved to {output_path}")

def visualize_instances(original_img, pr_bboxes, pr_classes, class_dict, output_path):
    """
    Visualize predictions on the image and save to file.
    
    :param original_img: Original image without padding
    :param pr_bboxes: List of predicted bounding boxes, each bbox is in [x, y, width, height] format
    :param pr_classes: List of predicted class IDs corresponding to the bounding boxes
    :param class_dict: Dictionary mapping class IDs to class names
    :param output_path: Path to save the output image
    """

    # Fixed color map for each class
    colors = {
        1: (255, 0, 0),     # Red
        2: (0, 255, 0),     # Green
        3: (0, 0, 255),     # Blue
        4: (255, 255, 0),   # Cyan
        5: (255, 0, 255),   # Magenta
        6: (0, 255, 255),   # Yellow
        7: (128, 0, 128),   # Purple
        8: (0, 128, 128)    # Teal
    }
    
    # Create a copy of the original image to draw the predictions
    img_copy = original_img.copy()

    # Iterate through each bbox, its corresponding class ID,
    for bbox, class_id in zip(pr_bboxes, pr_classes):
        if class_id == 0:
            continue  # Skip if class_id is 0
        x1, y1, x2, y2 = bbox
        color = colors[class_id]
        # Draw the bounding box
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 1)
        # Prepare the label with class name
        label = f"{class_dict[class_id]}"
        # Calculate the position for the label
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = max(y1, label_size[1] + 10)
        # Draw the label background
        cv2.rectangle(img_copy, (x1, label_y - label_size[1] - 10), (x1 + label_size[0], label_y + base_line - 10), color, cv2.FILLED)
        # Draw the label text
        cv2.putText(img_copy, label, (x1, label_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Save the image using OpenCV
    cv2.imwrite(output_path, img_copy)
    #print(f"Image saved to {output_path}")

# =====================Other Function for specific use=====================
import os
def plot_histograms(counter, pr_probs):
    """
    Create histograms for each class in the 4D array pr_probs.
    
    Parameters:
    pr_probs (numpy.ndarray): 4D array with shape [q, class_num, width, height]
    """
    q, class_num, width, height = pr_probs.shape
    class_probs, class_indices = torch.max(pr_probs[0], dim=0)
    for class_index in range(class_num):
        # Extract the probabilities for the current class
        #class_probs = pr_probs[:, class_index, :, :].cpu().detach().flatten()
        class_mask = (class_indices == class_index)
        
        if class_mask.sum() == 0:
                continue
        
        # Create histogram
        plt.figure()
        plt.hist(class_probs[class_mask].cpu().detach().flatten(), bins=30, edgecolor='black')
        plt.title(f'Histogram for Class {class_index}')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.grid(True)
        filename = os.path.join(f'channel_{class_index}.png')
        plt.savefig(filename)
        plt.close()

def prob2img(pr_probs):
    pr_probs = pr_probs.cpu().detach().numpy().squeeze(0)
    for idx, pr_prob in enumerate(pr_probs):        
        # Save the image
        cv2.imwrite(f'op/output_image_{idx}.png', cv2.cvtColor(pr_prob, cv2.COLOR_RGB2BGR))
