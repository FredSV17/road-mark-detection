from analysis_utils import check_class_representation, check_objects_per_image
from data_loader import DataLoader
from data_analysis.bbox_utils import all_classes_heatmap, get_bounding_box_areas, save_bboxes

if __name__ == "__main__":
    dl_list = [DataLoader('train'),DataLoader('test'),DataLoader('valid')]
    get_bounding_box_areas(dl_list)
    check_class_representation(dl_list)
    check_objects_per_image(dl_list)
    all_classes_heatmap(dl_list)
    save_bboxes(dl_list)