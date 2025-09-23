from data_loader import DataLoader
from bounding_box_utils import show_bboxes
from data_points import DataPointsPoly

if __name__ == "__main__":
    dl = DataPointsPoly()
    test = dl.all_classes_heatmap()
    train = dl.get_path_list("train")
    test = dl.get_path_list("train")
    show_bboxes(train, test)