import os
import sys
from modules.resize_images import resize_images
from modules.classify_images import classify_images
from modules.detect_windows import detect_windows
from modules.facade_estimation import FacadeEstimation  # Import FacadeEstimation class
import shutil  # Import shutil for folder deletion


def main(input_folder):
    output_folder = "output"
    resized_folder = os.path.join(output_folder, "resized_images")
    classification_folder = os.path.join(output_folder, "classification_results")
    bounding_boxes_folder = os.path.join(output_folder, "bounding_boxes")
    facade_data_folder = os.path.join(output_folder, "facade_data")
    location_file = os.path.join(input_folder, "Locations.txt")  # Define the path to the Locations.txt file

    

    # Step 1: Resize images
    print("Resizing images...")
    resize_images(input_folder, resized_folder)
    
    # Step 2: Classify images
    print("Classifying images...")
    classify_images(resized_folder, classification_folder)
    
    # Step 3: Detect windows and bounding boxes
    print("Detecting windows and extracting bounding boxes...")
    detect_windows(resized_folder, bounding_boxes_folder)
    
    # Step 4: Estimate Facade
    print("Estimating facades...")
    buffer_factor = 0.1  # Set buffer factor
    facade_estimator = FacadeEstimation(buffer_factor=buffer_factor)  # Initialize the class
   # facade_estimator.process(resized_folder, bounding_boxes_folder, facade_data_folder)  # Process facades
    facade_estimator.process(resized_folder, bounding_boxes_folder, facade_data_folder, location_file)  # Process facades


    print(f"Pipeline complete! Output folder: {output_folder}")
    
        # Cleanup: Delete the resized_images folder
    if os.path.exists(resized_folder):
        print(f"Cleaning up")
        shutil.rmtree(resized_folder)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python BuildingAnalysis.py <input_folder>")
    else:
        input_folder = sys.argv[1]
        main(input_folder)
