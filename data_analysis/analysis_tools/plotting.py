from typing import List

import pandas as pd
from data_analysis.bbox_loader import BBoxLoader
from shared.data_loader import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from shapely.geometry import Polygon
from data_analysis.viz_config import PLOT_LABELS

def check_class_representation(dl_list : List[DataLoader], verbose=False):
    for dl in dl_list:
        if verbose:
            print(f"Creating a graph of class representation for {PLOT_LABELS[dl.dt_type]} dataset...")
        # Count the amout of bounding boxes
        class_count = [[key, len(dl.bb_dict[key])] for key in dl.bb_dict.keys()]
        # Convert to a dataframe
        class_representation_df = pd.DataFrame(class_count, columns=["Class", "Count"])
        sns.barplot(class_representation_df, x='Class', y='Count')
        plt.title(f"Class representation in images ({PLOT_LABELS[dl.dt_type]} dataset)")
        plt.savefig(f"data_analysis/results/class_representation_{PLOT_LABELS[dl.dt_type]}")
        plt.clf()
        
        
def get_bbox_counts(dt):
        # Get list of labels
        labels = [item[1] for item in dt.dt_paired_list]
        bbox_count = []
        for file in labels:
            with open(file, 'r') as f:
                values = f.readlines()
                f.close()
            bbox_count.append(len(values))
        return bbox_count
    
def check_objects_per_image(dt_list : List[BBoxLoader], verbose=False):

    for dt in dt_list:
        if verbose:
            print(f"Creating a graph of objects per image for {PLOT_LABELS[dt.dt_type]} dataset...")
        # Create histogram for bounding box count in images
        sns.histplot(data=get_bbox_counts(dt))
        plt.title(f"Object count histogram ({PLOT_LABELS[dt.dt_type]} dataset)")
        plt.savefig(f"data_analysis/results/object_count_{PLOT_LABELS[dt.dt_type]}")
        plt.clf()
        
        
def make_heatmap(data : BBoxLoader, bb_class=0):
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
    
def all_classes_heatmap(dt_list, verbose=False):
    for dt in dt_list:
        if verbose:
            print(f"Creating heatmaps for {PLOT_LABELS[dt.dt_type]} dataset...")
        fig, axes = plt.subplots(5, 3, figsize=(50, 50))
        for i in range(13):
            heatmap = make_heatmap(dt, bb_class=i)

            # Compute row and column for this subplot
            row = i // 3
            col = i % 3
            # Plot the heatmap
            sns.heatmap(heatmap, cmap='hot', xticklabels=False, yticklabels=False, ax=axes[row][col])

        fig.savefig(f"data_analysis/results/heatmaps_{dt.dt_type}.png")
    
def get_bounding_box_areas(dt_list, verbose=False):
    
    area_list = []
    for dt in dt_list:
        if verbose:
            print(f"Creating bounding box area graph for {PLOT_LABELS[dt.dt_type]} dataset...")
        for bb_class in list(dt.bb_dict.values()):
            for bounding_box in bb_class:
                polygon = Polygon(bounding_box)
                # Get the area of the polygon
                area_list.append(polygon.area)
                
        # Create histogram for bounding box count in images
        sns.histplot(data=area_list, bins=10)
        plt.title(f"Bounding box area histogram ({PLOT_LABELS[dt.dt_type]} dataset)")
        plt.savefig(f"data_analysis/results/bb_area_{PLOT_LABELS[dt.dt_type]}")
        plt.clf()
        