
from data_augmentation.aug_options import AugOptions

from data_augmentation.augmentation_types.random import make_transform
# Prototype function

    
    
def main():
    args = AugOptions().parser.parse_args()
    make_transform()


if __name__ == "__main__":
    main()

