from data_augmentation import img_transform
from data_augmentation.aug_options import AugOptions
from shared.data_loader import DataLoader
def run_data_augmentation(path, verbose):
    dl = DataLoader(path)
    img_transform(dl.dt_dict['images'][0],dl.dt_dict['labels'][0])
    

def main():
    args = AugOptions().parser.parse_args()
    run_data_augmentation(args.path, args.verbose)


if __name__ == "__main__":
    main()
    