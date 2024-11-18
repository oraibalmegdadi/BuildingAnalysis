import os
import sys
import numpy as np
from modules.resize_images import resize_images
from modules.classify_images import classify_images
from modules.detect_windows import detect_windows
#from modules.generate_stats import generate_stats
#from modules.annotate_images import annotate_images

def main(input_folder):
    output_folder = "output"
    resized_folder = os.path.join(output_folder, "resized_images")
    classification_folder = os.path.join(output_folder, "classification_results")
    bounding_boxes_folder = os.path.join(output_folder, "bounding_boxes")
  #  stats_folder = os.path.join(output_folder, "statistics")
   # annotated_folder = os.path.join(output_folder, "annotated_images")
    
    # Step 1: Resize images
    print("Resizing images...")
    resize_images(input_folder, resized_folder)
    
    # Step 2: Classify images
    print("Classifying images...")
    classify_images(resized_folder, classification_folder)
    
    # Step 3: Detect windows and bounding boxes
    print("Detecting windows and extracting bounding boxes...")
    detect_windows(resized_folder, bounding_boxes_folder)
    
    # Step 4: Generate statistics
  #  print("Generating statistics...")
    #generate_stats(resized_folder, stats_folder)
    
    # Step 5: Annotate images
 #   print("Annotating images...")
    #annotate_images(resized_folder, bounding_boxes_folder, annotated_folder)
    
    print(f"Pipeline complete! Output folder: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python BuildingAnalysis.py <input_folder>")
    else:
        input_folder = sys.argv[1]
        main(input_folder)
