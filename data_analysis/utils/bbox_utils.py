import os
import cv2

import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn  as sns

from shared.data_loader import DataLoader
from shapely.geometry import Polygon

from data_analysis.viz_config import CLASS_COLORS, PLOT_LABELS

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
            cv2.line(new_img, start, end, CLASS_COLORS[parameters[0]], 2)
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
            os.makedirs(f'data_analysis/bounding_boxes/{dataset.dt_type}', exist_ok=True)
            
            cv2.imwrite(f'data_analysis/bounding_boxes/{dataset.dt_type}/{img_name + extension}', new_img)
            

def get_bounding_boxes(self, dataset='train'):
    bbox_dict = defaultdict(list)
    # Get list of labels
    labels = super().get_path_list(dataset)[1]
    for file in labels:
        with open(file, 'r') as f:
            values = f.readlines()
            f.close()
        for bbox in values:
            parameters = list(map(float, bbox.split()))
            points = parameters[1:]
            # Get points in bounding box
            points = [(int(points[i] * self.img_shape[0]), int(points[i + 1] * self.img_shape[1])) for i in range(0, len(points), 2)]
            # Save in a dictionary
            bbox_dict[parameters[0]] += [points]
    return bbox_dict
            
def make_heatmap(data : DataLoader, bb_class=0):
    heatmap = np.zeros(data.img_shape, dtype=np.uint8)
    for bbox in data.bb_dict[bb_class]:
        
        # Create an empty "map" of zeros (same size as image)
        mask = np.zeros(data.img_shape, dtype=np.uint8)

        # Define polygon points
        points = np.array(bbox, np.int32)

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
            # Plot the heatmap
            sns.heatmap(heatmap, cmap='hot', xticklabels=False, yticklabels=False, ax=axes[row][col])

        fig.savefig(f"data_analysis/results/heatmaps_{dataset.dt_type}.png")
    
def get_bounding_box_areas(dataset_list : list[DataLoader]):
    area_list = []
    for dataset in dataset_list:
        for bb_class in list(dataset.bb_dict.values()):
            for bounding_box in bb_class:
                polygon = Polygon(bounding_box)
                # Get the area of the polygon
                area_list.append(polygon.area)
                
        # Create histogram for bounding box count in images
        sns.histplot(data=area_list, bins=10)
        plt.title(f"Bounding box area histogram ({PLOT_LABELS[dataset.dt_type]} dataset)")
        plt.savefig(f"data_analysis/results/bb_area_{PLOT_LABELS[dataset.dt_type]}")
        plt.clf()
        
        
# def show_image_bbox(img, bbox_list):
#     new_img = img
#     h_img, w_img, _ = img.shape
#     colors = { 1 : (0, 0, 255), 0: (0, 255, 0)}
#     for polygon in bbox_list:
#         value_list = list(map(float, polygon.split()))
#         x_c, y_c, h, w = value_list[1:]
#         x_min = int((x_c - w/2) * w_img)
#         x_max = int((x_c + w/2) * w_img)
#         y_min = int((y_c - h/2) * h_img)
#         y_max = int((y_c + h/2) * h_img)
#         cv2.rectangle(new_img, (x_min,y_min), (x_max, y_max), colors[value_list[0]], 2)
#     return new_img
