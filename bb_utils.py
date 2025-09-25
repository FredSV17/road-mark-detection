import os
import cv2

import random
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn  as sns

from data_loader import DataLoader
from shapely.geometry import Polygon

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

plot_labels = {
    'train' : 'training',
    'test' : 'testing',
    'valid' : 'validation'
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

def save_bboxes(dt_list : list[DataLoader]):
    for dataset in dt_list:
        for img_path, lbl_path in zip(dataset.dt_dict["images"], dataset.dt_dict["labels"]):
            # Read the image
            image = cv2.imread(img_path)
            
            # Read the label file
            with open(lbl_path, 'r') as f:
                lbl_values = f.readlines()
            img_name, extension = os.path.splitext(os.path.basename(img_path))
            new_img = show_image_bbox_poly(image, lbl_values)
            # Create directory 
            os.makedirs(f'bounding_boxes/{dataset.dt_type}', exist_ok=True)
            
            cv2.imwrite(f'bounding_boxes/{dataset.dt_type}/{img_name + extension}', new_img)
            
            
    # # Pick random examples in train and put it in the examples folder
    # num_examples = 5
    # image_list = [cv2.imread(img_path) for img_path in random.sample(train[0], num_examples)]
    
    # for img_path in image_list:
    #     # Read the image
    #     image = cv2.imread(img_path)
        
    #     os.makedirs(f'bounding_boxes/examples', exist_ok=True)
                    
    #     cv2.imwrite(f'bounding_boxes/examples/{img_path}', new_img)
    
def get_bounding_boxes(self, dataset='train'):
    bounding_box_dict = defaultdict(list)
    # Get list of labels
    labels = super().get_path_list(dataset)[1]
    for file in labels:
        with open(file, 'r') as f:
            values = f.readlines()
            f.close()
        for bounding_box in values:
            parameters = list(map(float, bounding_box.split()))
            points = parameters[1:]
            # Get points in bounding box
            points = [(int(points[i] * self.img_shape[0]), int(points[i + 1] * self.img_shape[1])) for i in range(0, len(points), 2)]
            # Save in a dictionary
            bounding_box_dict[parameters[0]] += [points]
    return bounding_box_dict
            
def make_heatmap(data : DataLoader, bb_class=0):
    heatmap = np.zeros(data.img_shape, dtype=np.uint8)
    for bounding_box in data.bb_dict[bb_class]:
        
        # Create an empty "map" of zeros (same size as image)
        mask = np.zeros(data.img_shape, dtype=np.uint8)

        # Define polygon points
        points = np.array(bounding_box, np.int32)

        # Fill polygon area with 1s
        cv2.fillPoly(mask, [points], color=1)
        
        # Add mask with heatmap
        heatmap += mask
    return heatmap
    
def all_classes_heatmap(dataset_list : list[DataLoader]):
    for dataset in dataset_list:
        fig, axes = plt.subplots(5, 3, figsize=(50, 50))
        for i in range(13):
            heatmap = make_heatmap(dataset, bb_class=i)

            # Compute row and column for this subplot
            row = i // 3
            col = i % 3
            # Plot the first heatmap on the first axis (axes[0])
            sns.heatmap(heatmap, cmap='hot', xticklabels=False, yticklabels=False, ax=axes[row][col])

        # # Adjust layout to prevent overlapping titles/labels
        # plt.tight_layout()
        fig.savefig(f"data_analysis/heatmaps_{dataset.dt_type}.png")
    
def get_bounding_box_areas(dataset_list : list[DataLoader]):
    area_list = []
    for dataset in dataset_list:
        for bb_class in list(dataset.bb_dict.values()):
            for bounding_box in bb_class:
                polygon = Polygon(bounding_box)
                # Get the area of the polygon
                area_list.append(polygon.area)
                # area = get_area(points)
                # area_list.append(area)
        # Create histogram for bounding box count in images
        sns.histplot(data=area_list, bins=10)
        plt.title(f"Bounding box area histogram ({plot_labels[dataset.dt_type]} dataset)")
        plt.savefig(f"data_analysis/bb_area_{plot_labels[dataset.dt_type]}")
        plt.clf()