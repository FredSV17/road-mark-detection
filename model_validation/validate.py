from ultralytics import YOLO
from model_validation.validation_options import ValOptions
import os
import cv2

def main():
    args = ValOptions().parser.parse_args()
    model = YOLO(args.model_path)


    onnx_path = args.model_path

    file_size = os.path.getsize(onnx_path)  # in bytes
    file_size_mb = file_size / (1024 * 1024)

    print(f"Model size: {file_size_mb:.2f} MB")

    model = YOLO(onnx_path)

    metrics = model.val(data=f"{args.path}/data.yaml", imgsz=640, device="cuda" or "cpu")


    # Create directory for saving images
    output_dir = args.results_path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the images in test_images
    test_images_dir = args.test_imgs_path

    # Loop through each image in the test_images directory
    for filename in os.listdir(test_images_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(test_images_dir, filename)
            
            # Resize image to 640x640
            # Load the image
            img = cv2.imread(image_path)
            img = cv2.resize(img, (640,640))
            # Save the resized image
            resized_image_path = os.path.join(test_images_dir, "resized_" + filename)
            cv2.imwrite(resized_image_path, img)
            

            # Perform inference
            results = model(resized_image_path)
            # Save the results
            output_path = os.path.join(output_dir, filename)
            results[0].save(output_path)
            if args.verbose:
                print(f"Processed {filename}, saved to {output_path}")

if __name__ == "__main__":
    main()

    
    

    # # Export the model to ONNX format
    # path = model.export(format="onnx")  # return path to exported model