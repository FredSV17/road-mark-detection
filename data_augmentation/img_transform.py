import albumentations as A
from shared.bbox_loader import BBoxLoader
import cv2

from data_analysis.viz_config import CLASS_COLORS
import os
# Prototype function
def img_transform(bbl: BBoxLoader):
    
    img = bbl.dt_dict['images'][0]
    lbl_path = bbl.dt_dict['labels'][0]
    bboxes = bbl.get_bounding_boxes_by_label(lbl_path)
    
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
    
    image = cv2.imread(img)
    
    augmented = transform(image=image, keypoints=bboxes[0])
    new_image = augmented['image']
    # keypoints = 
    new_polygon = [list(map(int, kp)) for kp in augmented['keypoints']]
    img_with_poly = show_image_bbox_poly(new_image, [new_polygon])
    img_name, extension = os.path.splitext(os.path.basename(img))
    
    os.makedirs(f'data_augmentation/prototype_testing', exist_ok=True)
    cv2.imwrite(f'data_augmentation/prototype_testing/{img_name + extension}', img_with_poly)
    

def show_image_bbox_poly(img, lbl_values):
    new_img = img
    h, w, _ = img.shape
    for polygon in lbl_values:
        
        for i in range(len(polygon)):
            start = polygon[i]
            end = polygon[(i + 1) % len(polygon)]
            cv2.line(new_img, start, end, CLASS_COLORS[0])
    return new_img