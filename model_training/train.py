from model_training.model.yolo import run_yolo
from model_training.model_options import ModelOptions


def main():
    args = ModelOptions().parser.parse_args()
    run_yolo(args)


if __name__ == "__main__":
    main()

