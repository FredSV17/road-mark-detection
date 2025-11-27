from ultralytics import YOLO

def run_yolo(args):
    # Load a model
    model = YOLO(args.model)

    model.train(
        data=f"{args.path}/data.yaml",
        epochs=100,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        translate=args.translate,
        scale=args.scale,
        fliplr=0.0,
        mosaic=args.mosaic,
        erasing=0.0,
        auto_augment=None,
    )

    # # Evaluate model performance on the validation set
    # metrics = model.val()

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model