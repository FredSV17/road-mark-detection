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
        
        self.image_train_path_list = sorted([f"{images_train_path}/{img}" for img in os.listdir(images_train_path) if img.endswith('.jpg')])
        self.label_train_path_list = sorted([f"{labels_train_path}/{lbl}" for lbl in os.listdir(labels_train_path) if lbl.endswith('.txt')])
        self.image_test_path_list = sorted([f"{images_test_path}/{img}" for img in os.listdir(images_test_path) if img.endswith('.jpg')])
        self.label_test_path_list = sorted([f"{labels_test_path}/{lbl}" for lbl in os.listdir(labels_test_path) if lbl.endswith('.txt')])
    
    def get_train_path_list(self):
        return [self.image_train_path_list, self.label_train_path_list],[self.image_test_path_list, self.label_test_path_list]