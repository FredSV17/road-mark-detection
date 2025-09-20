import os
import cv2

class DataLoader:
    def __init__(self):
        base_path = "data"
        # TODO: Code only works with sagittal images for now
        images_train_path = os.path.join(base_path, "train/images")
        labels_train_path = os.path.join(base_path, "train/labels")
        images_test_path = os.path.join(base_path, "test/images")
        labels_test_path = os.path.join(base_path, "test/labels")
        images_valid_path = os.path.join(base_path, "valid/images")
        labels_valid_path = os.path.join(base_path, "valid/labels")
        self.path_list = {  "train": [sorted([f"{images_train_path}/{img}" for img in os.listdir(images_train_path) if img.endswith('.jpg')]),
                                     sorted([f"{labels_train_path}/{lbl}" for lbl in os.listdir(labels_train_path) if lbl.endswith('.txt')])],
                            "valid": [sorted([f"{images_valid_path}/{img}" for img in os.listdir(images_valid_path) if img.endswith('.jpg')]),
                                      sorted([f"{labels_valid_path}/{lbl}" for lbl in os.listdir(labels_valid_path) if lbl.endswith('.txt')])],
                            "test": [sorted([f"{images_test_path}/{img}" for img in os.listdir(images_test_path) if img.endswith('.jpg')]),
                                     sorted([f"{labels_test_path}/{lbl}" for lbl in os.listdir(labels_test_path) if lbl.endswith('.txt')])]}

    
    def get_path_list(self, dataset="train"):
        return self.path_list[dataset]