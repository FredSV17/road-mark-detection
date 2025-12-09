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
            help="Model to load"
        )
        self.parser.add_argument(
            "--export_path",
            type=str,
            default="model_training/model/exported_model/model.onnx",
            help="Path to save exported model"
        )
        self.parser.add_argument(
            "--epochs",
            type=int,
            default=100,
            help="Number of training epochs"
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
        self.parser.add_argument(
            "--name",
            type=float,
            default=0,
            help="Data augmentation option - scale"
        )
