from shared.base_options import BaseOptions


class AnalysisOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize_analysis_args()
        
    def initialize_analysis_args(self):
        self.parser.add_argument(
            "--dataset",
            type=str,
            default="train",
            help="Which dataset to analyze (Default: train)"
        )

        self.parser.add_argument(
            "--save_bboxes",
            action="store_true",
            help="Save images with the visual bounding box representation in a folder (Default: False)"
        )