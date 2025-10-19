from shared.base_options import BaseOptions


class AugOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize_augmentation_args()
        
    def initialize_augmentation_args(self):
        self.parser.add_argument(
            "--augment_type",
            type=str,
            default="random",
            help="Type of augmentation"
        )

        self.parser.add_argument(
            "--save_bboxes",
            action="store_true",
            help="Save images with the visual bounding box representation in a folder (Default: False)"
        )