from data_loader import DataLoader
import cv2
from collections import defaultdict
import numpy as np

class DataPointsPoly(DataLoader):
    def __init__(self):
        super().__init__()
        self.img_shape = (640,640)
        
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
        return "Not implemented"