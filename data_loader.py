import os
import cv2

from collections import defaultdict

class DataLoader:
    def __init__(self, dataset='train'):
        base_path = "data"
        self.img_shape = (640,640)
        self.dt_type = dataset
        
        # TODO: Code only works with sagittal images for now
        images_path = os.path.join(base_path, f"{dataset}/images")
        labels_path = os.path.join(base_path, f"{dataset}/labels")
        self.dt_dict = {  
                        "images": sorted([f"{images_path}/{img}" for img in os.listdir(images_path) if img.endswith('.jpg')]),
                        "labels": sorted([f"{labels_path}/{lbl}" for lbl in os.listdir(labels_path) if lbl.endswith('.txt')])
                    }
        self.bb_dict = self.define_bounding_boxes()

    def define_bounding_boxes(self):
        bounding_box_dict = defaultdict(list)
        # Get list of labels
        labels = self.dt_dict['labels']
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
    