from shared.base_options import BaseOptions

from data_analysis.utils.bbox_utils import all_classes_heatmap, get_bounding_box_areas, save_imgs_with_bbox
from data_analysis.bbox_loader import BBoxLoader
from data_analysis.utils.analysis_utils import check_class_representation, check_objects_per_image


def run_analysis(path: str, dataset_list: list[str], save_bboxes: bool, verbose: bool):
    dl_list = []
    for dataset in dataset_list.split(','):
        try:
            dl_list.append(BBoxLoader(path, dataset))
        except:
            raise ValueError("Dataset not recognized (must be 'train', 'test', or 'valid')")
        
    get_bounding_box_areas(dl_list, verbose)
    check_class_representation(dl_list, verbose)
    check_objects_per_image(dl_list, verbose)
    all_classes_heatmap(dl_list, verbose)
    if save_bboxes:
        save_imgs_with_bbox(dl_list, verbose)


def main():
    args = BaseOptions().parser.parse_args()
    run_analysis(args.path, args.dataset, args.save_bboxes, args.verbose)


if __name__ == "__main__":
    main()
    