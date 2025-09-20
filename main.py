from data_loader import DataLoader
from bounding_box_utils import show_bboxes
from data_points import DataPointsPoly

if __name__ == "__main__":
    dl = DataPointsPoly()
    test = dl.get_bounding_boxes()
    print("debug here")
    # train, test = dl.get_train_path_list()
    # show_bboxes(train, test)