import os
import cv2
import numpy as np

def resize_images(input_folder, output_folder, size=(640, 640)):
    os.makedirs(output_folder, exist_ok=True)
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                resized_img = cv2.resize(img, size)
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)
                cv2.imwrite(os.path.join(output_subfolder, file), resized_img)