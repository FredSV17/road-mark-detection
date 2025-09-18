
from data_loader import DataLoader
from save_bounding_boxes import show_bboxes


if __name__ == "__main__":
    dl = DataLoader()
    train, test = dl.get_train_path_list()
    show_bboxes(train, test)