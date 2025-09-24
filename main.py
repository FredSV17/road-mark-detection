from data_loader import DataLoader
from bb_utils import all_classes_heatmap, save_bboxes

if __name__ == "__main__":
    dl_list = [DataLoader('train'),DataLoader('test'),DataLoader('valid')]
    all_classes_heatmap(dl_list[0])
    save_bboxes(dl_list)