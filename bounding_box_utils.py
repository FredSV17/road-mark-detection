import cv2
import os
import random
import numpy as np

class_colors = {
    0: (255, 0, 0),       # Red
    1: (0, 255, 0),       # Green
    2: (0, 0, 255),       # Blue
    3: (255, 255, 0),     # Yellow
    4: (255, 0, 255),     # Magenta
    5: (0, 255, 255),     # Cyan
    6: (255, 165, 0),     # Orange
    7: (128, 0, 128),     # Purple
    8: (0, 128, 128),     # Teal
    9: (128, 128, 0),     # Olive
    10: (255, 192, 203),  # Pink
    11: (0, 0, 128),      # Navy
    12: (128, 0, 0),      # Maroon
    13: (0, 128, 0),      # Dark Green
}

def show_image_bbox(img, bbox_list):
    new_img = img
    h_img, w_img, _ = img.shape
    colors = { 1 : (0, 0, 255), 0: (0, 255, 0)}
    for polygon in bbox_list:
        value_list = list(map(float, polygon.split()))
        x_c, y_c, h, w = value_list[1:]
        x_min = int((x_c - w/2) * w_img)
        x_max = int((x_c + w/2) * w_img)
        y_min = int((y_c - h/2) * h_img)
        y_max = int((y_c + h/2) * h_img)
        cv2.rectangle(new_img, (x_min,y_min), (x_max, y_max), colors[value_list[0]], 2)
    return new_img

def show_image_bbox_poly(img, lbl_values):
    new_img = img
    h, w, _ = img.shape
    for polygon in lbl_values:
        parameters = list(map(float, polygon.split()))
        values = parameters[1:]
        points = [(int(values[i] * w), int(values[i + 1] * h)) for i in range(0, len(values), 2)]
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            cv2.line(new_img, start, end, class_colors[parameters[0]], 2)
    return new_img

def show_bboxes(train, test):
    for img_path, lbl_path in zip(train[0], train[1]):
        # Read the image
        image = cv2.imread(img_path)
        
        # Read the label file
        with open(lbl_path, 'r') as f:
            lbl_values = f.readlines()
        img_name, extension = os.path.splitext(os.path.basename(img_path))
        new_img = show_image_bbox_poly(image, lbl_values)
        # Create directory 
        os.makedirs(f'bounding_boxes/train', exist_ok=True)
        
        cv2.imwrite(f'bounding_boxes/train/{img_name + extension}', new_img)
    for img_path, lbl_path in zip(test[0], test[1]):
        # Read the image
        image = cv2.imread(img_path)
        
        # Read the label file
        with open(lbl_path, 'r') as f:
            bbox_list = f.readlines()
        img_name, extension = os.path.splitext(os.path.basename(img_path))
        new_img = show_image_bbox_poly(image, bbox_list)
        # Create directory 
        os.makedirs(f'bounding_boxes/test', exist_ok=True)
                
        cv2.imwrite(f'bounding_boxes/test/{img_name + extension}', new_img)
    # Pick random examples in train and put it in the examples folder
    num_examples = 5
    image_list = [cv2.imread(img_path) for img_path in random.sample(train[0], num_examples)]
    os.makedirs(f'bounding_boxes/examples', exist_ok=True)
                
    cv2.imwrite(f'bounding_boxes/test/{img_name + extension}', new_img)
    
def create_heatmap(img_shape, labels, curr_class):
    h, w, _ = img_shape

    # Create an empty "map" of zeros (same size as image)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    for polygon in labels:
        parameters = list(map(float, polygon.split()))
        values = parameters[1:]
        points = [(int(values[i] * w), int(values[i + 1] * h)) for i in range(0, len(values), 2)]
        for i in range(len(points)):
            start = points[i]
            end = points[(i + 1) % len(points)]
            cv2.line(new_img, start, end, class_colors[parameters[0]], 2)
    # Define polygon points
    points = np.array([[100, 50], [200, 80], [150, 200]], np.int32)
    points = points.reshape((-1, 1, 2))

    # Fill polygon area with 1s
    cv2.fillPoly(mask, [points], color=1)