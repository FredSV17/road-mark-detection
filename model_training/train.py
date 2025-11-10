import albumentations as A

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11.pt")

# Training with custom augmentation parameters
model.train(data="data/data.yaml", epochs=100, hsv_h=0.03, hsv_s=0.6, hsv_v=0.5)

# Training without any augmentations (disabled values omitted for clarity)
model.train(
    data="data/data.yaml",
    epochs=100,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=1.0,
    erasing=0.0,
    auto_augment=None,
)

# Training with custom Albumentations transforms (Python API only)
custom_transforms = [
    A.Blur(blur_limit=7, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
]
model.train(data="data/data.yaml", epochs=100, augmentations=custom_transforms)