import os
from ultralytics import YOLO

def classify_images(resized_folder, classification_folder, model_path="modules/models/classification_best.pt", confidence_threshold=0.4):
    """
    Classifies images using a trained YOLOv8 model, including confidence scores.

    Args:
        resized_folder (str): Path to the folder containing resized images.
        classification_folder (str): Path to the folder where classification results will be saved.
        model_path (str): Path to the trained YOLOv8 model (e.g., "classification_best.pt").
        confidence_threshold (float): Minimum confidence score to consider a classification valid.
    """
    # Create classification folder if it doesn't exist
    os.makedirs(classification_folder, exist_ok=True)

    # Load the trained YOLO model
    model = YOLO(model_path)

    # Prepare the results file
    results_file = os.path.join(classification_folder, "classification_results_with_confidence.txt")
    with open(results_file, "w") as f:
        f.write("Filename,Predicted_Class,Confidence\n")  # Header for the results

    # Iterate through all images in the resized folder
    for root, _, files in os.walk(resized_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):  # Check for image files
                img_path = os.path.join(root, file)

                # Predict the class using the YOLO model
                results = model.predict(source=img_path, imgsz=640)
                probabilities = results[0].probs  # Classification probabilities object
                max_confidence = probabilities.top1conf.item()  # Confidence score of top class
                class_idx = probabilities.top1  # Index of the top class
                class_name = results[0].names[class_idx]  # Predicted class name

                # Handle low-confidence predictions
                if max_confidence < confidence_threshold:
                    class_name = "Other"

                # Print and save the classification result
                print(f"Image: {file} | Class: {class_name} | Confidence: {max_confidence:.2f}")
                with open(results_file, "a") as f:
                    f.write(f"{file},{class_name},{max_confidence:.2f}\n")

    print(f"Classification complete. Results saved to {results_file}")
