from data_analysis.analysis_options import AnalysisOptions
from shared.tools.bbox_save import save_dataset_with_bboxes
from data_analysis.analysis_tools.plotting import all_classes_heatmap, get_bounding_box_areas, check_class_representation, check_objects_per_image

from shared.bbox_loader import BBoxLoader



def run_analysis(path, dataset_list, save_bboxes, verbose):
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
        save_dataset_with_bboxes(dl_list, verbose)


def main():
    args = AnalysisOptions().parser.parse_args()
    run_analysis(args.path, args.dataset, args.save_bboxes, args.verbose)


if __name__ == "__main__":
    main()
    