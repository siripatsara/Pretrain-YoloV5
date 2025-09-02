import numpy as np
import cv2


def iou_score(mask1, mask2):
    """
    คำนวณ IoU ระหว่าง mask1 กับ mask2
    mask1, mask2: numpy array หรือ torch tensor, binary mask (0 หรือ 1)
    return: IoU score (float)
    """
    # ถ้าเป็น torch tensor ให้แปลงเป็น numpy ก่อน
    if 'torch' in str(type(mask1)):
        mask1 = mask1.cpu().numpy()
    if 'torch' in str(type(mask2)):
        mask2 = mask2.cpu().numpy()

    # ทำให้เป็น binary 0/1 ชัดเจน
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # กรณีไม่มีอะไรเลยถือว่า IoU=1
    else:
        return intersection / union


def compare_masks_iou(yolov5_mask, unet_mask, gt_mask):
    """
    เปรียบเทียบ IoU ระหว่าง YOLOv5-seg, U-Net Carvana กับ GT mask
    รับ mask ทั้ง 3 เป็น binary numpy array หรือ torch tensor
    return dict ของ IoU score แต่ละคู่
    """
    iou_yolo_gt = iou_score(yolov5_mask, gt_mask)
    iou_unet_gt = iou_score(unet_mask, gt_mask)
    iou_yolo_unet = iou_score(yolov5_mask, unet_mask)

    return {
        "IoU YOLOv5 vs GT": iou_yolo_gt,
        "IoU U-Net vs GT": iou_unet_gt,
        "IoU YOLOv5 vs U-Net": iou_yolo_unet,
    }


def load_mask_from_image(path, target_size=None):
    # โหลดภาพ grayscale
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if target_size is not None:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 127).astype(np.uint8)


def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union


# โหลด GT เพียงครั้งเดียว
gt_mask = load_mask_from_image(
    'C:/university/241-353/T.Fern/program/labs/lab4/gt_bin.png')
gt_h, gt_w = gt_mask.shape  # เก็บขนาด

# โหลด YOLOv5 masks — resize ให้เท่ากับ GT
# yolov5_mask_025 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/myblue_mask_25.png', target_size=(gt_w, gt_h))
# yolov5_mask_075 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/myblue_mask_75.png', target_size=(gt_w, gt_h))
yolov5_mask_025 = load_mask_from_image(
    'C:/university/241-353/T.Fern/program/yolov5/runs/predict-seg/exp18/binary_masks_th0.25/myblue_binary_th0.25.png', target_size=(gt_w, gt_h))
yolov5_mask_075 = load_mask_from_image(
    'C:/university/241-353/T.Fern/program/yolov5/runs/predict-seg/exp19/binary_masks_th0.75/myblue_binary_th0.75.png', target_size=(gt_w, gt_h))

# โหลด U-Net masks — resize เช่นกัน
unet_mask_025 = load_mask_from_image(
    'C:/university/241-353/T.Fern/program/labs/lab4/unet05_25.png', target_size=(gt_w, gt_h))
unet_mask_075 = load_mask_from_image(
    'C:/university/241-353/T.Fern/program/labs/lab4/unet05_75.png', target_size=(gt_w, gt_h))

print("IoU YOLOv5 (0.25):", compute_iou(yolov5_mask_025, gt_mask))
print("IoU YOLOv5 (0.75):", compute_iou(yolov5_mask_075, gt_mask))
print("IoU U-Net (0.25):", compute_iou(unet_mask_025, gt_mask))
print("IoU U-Net (0.75):", compute_iou(unet_mask_075, gt_mask))


# # โหลด GT เพียงครั้งเดียว
# gt_mask = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/gt_bin.png')

# # โหลด YOLOv5 masks
# yolov5_mask_025 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/myblue_mask_25.png')
# yolov5_mask_075 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/myblue_mask_75.png')

# # โหลด U-Net masks
# unet_mask_025 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/unet05_25.png')
# unet_mask_075 = load_mask_from_image(
#     'C:/university/241-353/T.Fern/program/labs/lab4/unet05_75.png')

# # คำนวณ IoU
# print("IoU YOLOv5 (0.25):", compute_iou(yolov5_mask_025, gt_mask))
# print("IoU YOLOv5 (0.75):", compute_iou(yolov5_mask_075, gt_mask))
# print("IoU U-Net (0.25):", compute_iou(unet_mask_025, gt_mask))
# print("IoU U-Net (0.75):", compute_iou(unet_mask_075, gt_mask))
