import os
import cv2

from data_analysis.bbox_loader import BBoxLoader


from data_analysis.viz_config import CLASS_COLORS, PLOT_LABELS

def show_image_bbox_poly(img, lbl_values):
    new_img = img
    h, w, _ = img.shape
    for polygon in lbl_values:
        parameters = list(map(float, polygon.split()))
        values = parameters[1:]
        # Get polygon coordinates (x,y)
        points = [(int(values[i] * w), int(values[i + 1] * h)) 
                  for i in range(0, len(values), 2)]
        
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            cv2.line(new_img, start, end, CLASS_COLORS[parameters[0]], 2)
    return new_img

def save_imgs_with_bbox(dt_list : list[BBoxLoader], verbose=False):
    for dataset in dt_list:
        if verbose:
            print(f"Saving bounding boxes for {PLOT_LABELS[dataset.dt_type]} dataset ...")
        for img_path, lbl_path in zip(dataset.dt_dict["images"], dataset.dt_dict["labels"]):
            # Read the image
            image = cv2.imread(img_path)
            
            # Read the label file
            with open(lbl_path, 'r') as f:
                lbl_values = f.readlines()
            img_name, extension = os.path.splitext(os.path.basename(img_path))
            new_img = show_image_bbox_poly(image, lbl_values)
            # Create directory 
            os.makedirs(f'data_analysis/bounding_boxes/{dataset.dt_type}', exist_ok=True)
            
            cv2.imwrite(f'data_analysis/bounding_boxes/{dataset.dt_type}/{img_name + extension}', new_img)
            
