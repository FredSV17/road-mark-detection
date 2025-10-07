
from shapely import Polygon
from data_augmentation.aug_options import AugOptions
from shared.bbox_loader import BBoxLoader
import cv2

import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from shared.tools.bbox_save import save_img_with_bboxes

import os
# Prototype function
def make_transform():
    dl = BBoxLoader('data')

    img = dl.dt_paired_list[0][0]
    lbl = dl.dt_paired_list[0][1]
    
    bboxes = dl.get_bounding_boxes_by_label(lbl)
    image = cv2.imread(img)
    
    polygons = [
        Polygon(bbox) for bbox in bboxes
    ]

    polygons_on_image = PolygonsOnImage(polygons, shape=image.shape)

    seq = iaa.Sequential([
        iaa.Affine(rotate=20, scale=1.2)
    ])

    image_aug, polygons_aug = seq(image=image, polygons=polygons_on_image)
    aug_polygons = [[[int(x), int(y)] for x, y in p.coords] 
                    for p in polygons_aug]
    path = f'data_augmentation/augmentated_dataset'
    os.makedirs(path, exist_ok=True)
    save_img_with_bboxes(image_aug, img, aug_polygons, f'data_augmentation/augmentated_dataset')
    
    
def main():
    args = AugOptions().parser.parse_args()
    make_transform()


if __name__ == "__main__":
    main()

