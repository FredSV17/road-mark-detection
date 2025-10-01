import albumentations as alb
from data_augmentation import img_transform
from data_augmentation.aug_options import AugOptions
from shared.bbox_loader import BBoxLoader

def run_data_augmentation(path, verbose):
    dl = BBoxLoader(path)
    img_transform(dl)
    

def main():
    args = AugOptions().parser.parse_args()
    run_data_augmentation(args.path, args.verbose)


if __name__ == "__main__":
    main()
    