import os
import cv2

from shared.bbox_loader import BBoxLoader


from data_analysis.viz_config import CLASS_COLORS, PLOT_LABELS



def draw_bbox(img, points, color=(0, 255, 0)):
    for i in range(len(points)):
        start = points[i]
        end = points[(i + 1) % len(points)]
        cv2.line(img, start, end, color, 2)
    return img


def save_img_with_bboxes(img_path, bboxes, bbox_path):
    # Read the image
    image = cv2.imread(img_path)
    
    # # Read the label file
    # with open(lbl_path, 'r') as f:
    #     lbl_values = f.readlines()
    img_name, extension = os.path.splitext(os.path.basename(img_path))
    for bbox in bboxes:
        image = draw_bbox(image, bbox)
    
    cv2.imwrite(f'{bbox_path}/{img_name + extension}', image)
    
    
def get_image_bbox_by_label(img_path, lbl_path, img_shape):

    # Read the image
    image = cv2.imread(img_path)
    
    # Read the label file
    with open(lbl_path, 'r') as f:
        lbl_values = f.readlines()
        
    h, w = img_shape
    for polygon in lbl_values:
        parameters = list(map(float, polygon.split()))
        bboxes = parameters[1:]
        # Get polygon coordinates (x,y)
        points = [(int(bboxes[i] * h), int(bboxes[i + 1] * w))
                  for i in range(0, len(bboxes), 2)]
        
        image = draw_bbox(image, points, CLASS_COLORS[parameters[0]])

    return image

def save_dataset_with_bboxes(dt_list, verbose=False):
    for dataset in dt_list:
        if verbose:
            print(f"Saving bounding boxes for {PLOT_LABELS[dataset.dt_type]} dataset ...")
            
        path = f'data_analysis/bounding_boxes/{dataset.dt_type}'
        # Create directory 
        os.makedirs(path, exist_ok=True)
        for img_path, lbl_path in dataset.dt_paired_list:

            new_img = get_image_bbox_by_label(img_path, lbl_path, dataset.img_shape)
            
            img_name, extension = os.path.splitext(os.path.basename(img_path))
            cv2.imwrite(f'{path}/{img_name + extension}', new_img)

