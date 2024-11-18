import os
from ultralytics import YOLO

def detect_windows(input_folder, output_folder, model_path="modules/models/windowsPrediction_best.pt", confidence_threshold=0.5):
    """
    Detects windows in images using a trained YOLOv8 model and saves bounding boxes to text files.

    Args:
        input_folder (str): Path to the folder containing resized images.
        output_folder (str): Path to the folder where detection results will be saved.
        model_path (str): Path to the trained YOLOv8 model (e.g., "windowsPrediction_best.pt").
        confidence_threshold (float): Minimum confidence score to consider a detection valid.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load the trained YOLO model
    model = YOLO(model_path)

    # Iterate through all images in the input folder
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for image files
                img_path = os.path.join(root, file)

                # Perform window detection using the YOLO model
                results = model.predict(source=img_path, imgsz=640, conf=confidence_threshold)

                # Prepare the output text file
                base_name = os.path.splitext(file)[0]  # Get the file name without extension
                output_file = os.path.join(output_folder, f"{base_name}.txt")

                # Write bounding boxes to the text file
                with open(output_file, "w") as f:
                    f.write("Class Confidence xmin ymin xmax ymax\n")  # Header
                    for box in results[0].boxes:  # Iterate through detected boxes
                        class_id = int(box.cls)  # Class ID
                        confidence = float(box.conf)  # Confidence score
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()  # Bounding box coordinates
                        f.write(f"{class_id} {confidence:.2f} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n")

                print(f"Processed {file}, saved bounding boxes to {output_file}")

    print(f"Window detection complete. Results saved to {output_folder}")
