
from shapely import Polygon


import cv2
import os
import imgaug.augmenters as iaa
from imgaug.augmentables.polys import Polygon, PolygonsOnImage

from shared.tools.bbox_save import save_img_with_bboxes
from shared.bbox_loader import BBoxLoader
from tqdm import tqdm
import os

def make_transform():
    dl = BBoxLoader('data')

    
    for img, lbl in tqdm(dl.dt_paired_list):
        bboxes_class = dl.get_bounding_boxes_by_label(lbl)
        image = cv2.imread(img)
        
        polygons = [
            Polygon(bbox) for bbox in [bboxes[1] for bboxes in bboxes_class]
        ]

        polygons_on_image = PolygonsOnImage(polygons, shape=image.shape)
        seq = iaa.Sequential([
            iaa.Affine(rotate=10, scale=1.5),
            iaa.Flipud(0.5),
            iaa.WithBrightnessChannels(iaa.Add((-50, 50))),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2*255)),
        ])
        classes = [bboxes[0] for bboxes in bboxes_class]
        image_aug, polygons_aug = seq(image=image, polygons=polygons_on_image)
        aug_polygons = [[c,[(int(x) / dl.img_shape[0], int(y) / dl.img_shape[0]) for x, y in p.coords]]
                        for p,c  in zip(polygons_aug, classes)]
        path = f'data_augmentation/augmented_dataset'
        os.makedirs(path, exist_ok=True)
        save_img_with_bboxes(image_aug, img, aug_polygons,lbl, path)