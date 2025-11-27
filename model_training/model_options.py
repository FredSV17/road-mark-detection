from shared.base_options import BaseOptions


class ModelOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize_mdltraining_args()
        
    def initialize_mdltraining_args(self):
        self.parser.add_argument(
            "--model",
            type=str,
            default="yolo11n.pt",
            help="Data augmentation option - mosaic"
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Data augmentation option - mosaic"
        )
        self.parser.add_argument(
            "--mosaic",
            type=float,
            default=0,
            help="Data augmentation option - mosaic"
        )
        self.parser.add_argument(
            "--translate",
            type=float,
            default=0,
            help="Data augmentation option - translate"
        )
        self.parser.add_argument(
            "--scale",
            type=float,
            default=0,
            help="Data augmentation option - scale"
        )