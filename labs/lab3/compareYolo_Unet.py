import cv2
import numpy as np

# Load GT and predicted masks (grayscale)
gt = cv2.imread(
    "C:/university/241-353/T.Fern/program/labs/lab3/SegmentationClass/blue.png", cv2.IMREAD_GRAYSCALE)
yolo_pred = cv2.imread("yolo_pred.png", cv2.IMREAD_GRAYSCALE)
unet_pred = cv2.imread("unet_pred.png", cv2.IMREAD_GRAYSCALE)

# Convert to binary (threshold-based)
gt_bin = (gt > 0).astype(np.uint8)
yolo_bin_025 = (yolo_pred > 64).astype(np.uint8)   # for threshold 0.25
yolo_bin_075 = (yolo_pred > 191).astype(np.uint8)  # for threshold 0.75
unet_bin = (unet_pred > 127).astype(np.uint8)

# IoU function


def iou(y_true, y_pred):
    inter = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    return inter / union if union != 0 else 1.0

# Dice function


def dice(y_true, y_pred):
    inter = np.logical_and(y_true, y_pred).sum()
    return 2 * inter / (y_true.sum() + y_pred.sum())


# Print results
print("YOLOv5 @0.25 → IoU:", iou(gt_bin, yolo_bin_025),
      ", Dice:", dice(gt_bin, yolo_bin_025))
print("YOLOv5 @0.75 → IoU:", iou(gt_bin, yolo_bin_075),
      ", Dice:", dice(gt_bin, yolo_bin_075))
print("U-Net        → IoU:", iou(gt_bin, unet_bin),
      ", Dice:", dice(gt_bin, unet_bin))
