import os

from collections import defaultdict

from shared.data_loader import DataLoader

class BBoxLoader(DataLoader):
    def __init__(self, path, dataset='train'):
        super().__init__(path, dataset)
        self.bb_dict = self.group_bounding_boxes_by_class()
        
    
    def group_bounding_boxes_by_class(self):
        """
        Returns a dict mapping class names to lists of bounding boxes.
        Example:
        {
            "0": [(x1, y1, x2, y2), ...],
            "1": [(x1, y1, x2, y2), ...]
        }
        """
        bbox_dict = defaultdict(list)
        # Get list of labels
        labels = [item[1] for item in self.dt_paired_list]
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
    
    def get_bounding_boxes_by_label(self, label_path):
        """
        Returns a list of all bounding boxes contained in a txt label file.
        Example:
        [
            [2, [(x1, y1, x2, y2), ...]]
            [1, [(x1, y1, x2, y2), ...]]
        ]
        """
        bboxes = []
        with open(label_path, 'r') as f:
            values = f.readlines()
            f.close()
        for bbox in values:
            parameters = list(map(float, bbox.split()))
            points = parameters[1:]
            # Get points in bounding box
            points = [(int(points[i] * self.img_shape[0]), int(points[i + 1] * self.img_shape[1])) for i in range(0, len(points), 2)]
            # Save in a dictionary
            bboxes += [[int(parameters[0]),list(points)]]
        return bboxes