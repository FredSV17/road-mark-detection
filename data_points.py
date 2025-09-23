from data_loader import DataLoader
import cv2
from collections import defaultdict
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import seaborn  as sns


class DataPointsPoly(DataLoader):
    def __init__(self):
        super().__init__()
        self.img_shape = (640,640)
        self.bounding_box_dict = self.get_bounding_boxes()
        
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
            
    def make_heatmap(self, dataset='train', bb_class=0):
        heatmap = np.zeros(self.img_shape, dtype=np.uint8)
        for bounding_box in self.bounding_box_dict[bb_class]:
            
            # Create an empty "map" of zeros (same size as image)
            mask = np.zeros(self.img_shape, dtype=np.uint8)

            # Define polygon points
            points = np.array(bounding_box, np.int32)

            # Fill polygon area with 1s
            cv2.fillPoly(mask, [points], color=1)
            
            # Add mask with heatmap
            heatmap += mask
        return heatmap
        
    def all_classes_heatmap(self, dataset='train'):
        fig, axes = plt.subplots(5, 3, figsize=(50, 50))
        for i in range(13):
            heatmap = self.make_heatmap(bb_class=i)

            # Compute row and column for this subplot
            row = i // 3
            col = i % 3
            # Plot the first heatmap on the first axis (axes[0])
            sns.heatmap(heatmap, cmap='hot', xticklabels=False, yticklabels=False, ax=axes[row][col])

        # # Adjust layout to prevent overlapping titles/labels
        # plt.tight_layout()

        fig.savefig("heatmaps.png")