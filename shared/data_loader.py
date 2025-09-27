import os

from collections import defaultdict

class DataLoader:
    def __init__(self, path, dataset='train'):
        self.img_shape = (640,640)
        self.dt_type = dataset
        
        try:
            images_path = os.path.join(path, f"{dataset}/images")
            labels_path = os.path.join(path, f"{dataset}/labels")
        except Exception as e:
            raise ValueError(e)
        self.dt_dict = {  
                        "images": sorted([f"{images_path}/{img}" for img in os.listdir(images_path) if img.endswith('.jpg')]),
                        "labels": sorted([f"{labels_path}/{lbl}" for lbl in os.listdir(labels_path) if lbl.endswith('.txt')])
                    }

    