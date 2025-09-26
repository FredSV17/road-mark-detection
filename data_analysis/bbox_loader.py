import os

from collections import defaultdict

from shared.data_loader import DataLoader

class BBoxLoader(DataLoader):
    def __init__(self, path, dataset='train'):
        super().__init__(path, dataset)
        self.bb_count = self.get_bbox_counts()
        self.bb_dict = self.define_bounding_boxes()
        
    def get_bbox_counts(self):
        # Get list of labels
        labels = self.dt_dict['labels']
        bbox_count = []
        for file in labels:
            with open(file, 'r') as f:
                values = f.readlines()
                f.close()
            bbox_count.append(len(values))
        return bbox_count
    
    def define_bounding_boxes(self):
        bbox_dict = defaultdict(list)
        # Get list of labels
        labels = self.dt_dict['labels']
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
                bbox_dict[int(parameters[0])] += [points]
        return bbox_dict
    