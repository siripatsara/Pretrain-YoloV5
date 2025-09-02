import cv2
import numpy as np

# อ่าน SegmentationClass mask
mask = cv2.imread(
    'C:/university/241-353/T.Fern/program/labs/lab4/gt/SegmentationClass/myblue.png')

# แปลงเป็น BGR เป็น RGB
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

# กำหนดสีของ car class (42,125,209)
car_color = np.array([42, 125, 209])

# สร้าง binary mask โดยหาพิกเซลที่เป็นสี car
binary_mask = np.all(mask_rgb == car_color, axis=-1).astype(np.uint8) * 255

# บันทึกไฟล์
cv2.imwrite(
    'C:/university/241-353/T.Fern/program/labs/lab4/gt_bin.png', binary_mask)
print
