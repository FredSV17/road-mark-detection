from shared.base_options import BaseOptions


class ValOptions(BaseOptions):
    def __init__(self):
        super().__init__()
        self.initialize_validation_args()
        
    def initialize_validation_args(self):
        self.parser.add_argument(
            "--model_path",
            type=str,
            default="model_training/model/exported_model/model.onnx",
            help="Model to load"
        )
        self.parser.add_argument(
            "--test_imgs_path",
            type=str,
            default="model_validation/images/test",
            help="Path of images to be tested"
        )
        self.parser.add_argument(
            "--results_path",
            type=str,
            default="model_validation/images/result",
            help="Path to save result images"
        )
 