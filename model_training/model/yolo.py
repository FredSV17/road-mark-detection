import os
from ultralytics import YOLO


def run_yolo(args):
    # Load a model
    model = YOLO(args.model)

    model.train(
        data=f"{args.path}/data.yaml",
        epochs=args.epochs,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=args.translate,
        scale=args.scale,
        fliplr=0.0,
        mosaic=args.mosaic,
        erasing=0.0,
        auto_augment=None,
        patience=20,
        name=args.name,
        seed=64
    )
    target_path = args.export_path
    # # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
    os.rename(path, target_path)
