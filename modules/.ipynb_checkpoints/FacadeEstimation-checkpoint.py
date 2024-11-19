import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.filters import gabor
from scipy.spatial import ConvexHull


class FacadeEstimation:
    def __init__(self, buffer_factor=0.1):
        """
        Initializes the FacadeEstimation class.
        :param buffer_factor: Factor for buffering the convex hull.
        """
        self.buffer_factor = buffer_factor

    @staticmethod
    def get_image_files(data_folder, extension=".jpg"):
        return [f for f in os.listdir(data_folder) if f.endswith(extension)]

    @staticmethod
    def read_bounding_boxes(txt_file):
        try:
            # Read the file, skipping the header
            df = pd.read_csv(txt_file, sep='\s+', names=['Class', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax'], skiprows=1)
            
            # Check if required columns are present
            if {'xmin', 'ymin', 'xmax', 'ymax'}.issubset(df.columns):
                return df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
        return []

    @staticmethod
    def calculate_convex_hull(bounding_boxes):
        points = []
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = box
            points.extend([[x_min, y_min], [x_max, y_min], [x_min, y_max], [x_max, y_max]])
        points = np.array(points)
        if len(points) > 2:
            try:
                hull = ConvexHull(points)
                return points[hull.vertices]
            except Exception as e:
                print(f"Error calculating convex hull: {e}")
        return None

    def buffer_convex_hull(self, hull_points):
        centroid = np.mean(hull_points, axis=0)
        buffered_hull = []
        for point in hull_points:
            direction = point - centroid
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
            buffered_point = point + direction * self.buffer_factor * norm
            buffered_hull.append(buffered_point)
        return np.array(buffered_hull)

    @staticmethod
    def apply_gabor_filters(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernels = [cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F) for theta in np.linspace(0, np.pi, 16, endpoint=False)]
        return [cv2.filter2D(gray_image, cv2.CV_8UC3, k) for k in kernels]

    @staticmethod
    def calculate_gabor_filter_densities(gabor_filtered_images, convex_hull):
        mask = np.zeros(gabor_filtered_images[0].shape, dtype=np.uint8)
        cv2.fillPoly(mask, [convex_hull.astype(int)], 255)
        return [np.mean(cv2.bitwise_and(img, img, mask=mask)[mask > 0]) for img in gabor_filtered_images]

   
    
    @staticmethod
    def calculate_facade_statistics(image, expanded_region_mask, bounding_boxes, rotated_bounding_box, output_path):
        """
        Calculate detailed facade statistics and save them to a file.
        """
        expanded_area = np.sum(expanded_region_mask == 255)
        bounding_box_area = cv2.contourArea(rotated_bounding_box)
        window_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bounding_boxes]
        total_window_area = sum(window_areas)
    
        # Statistics
        expanded_window_percentage = (total_window_area / expanded_area) * 100 if expanded_area else 0
        bounding_box_window_percentage = (total_window_area / bounding_box_area) * 100 if bounding_box_area else 0
        avg_window_width = np.mean([box[2] - box[0] for box in bounding_boxes]) if bounding_boxes else 0
        avg_window_height = np.mean([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0
        std_window_width = np.std([box[2] - box[0] for box in bounding_boxes]) if bounding_boxes else 0
        std_window_height = np.std([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0
    
        # Save to a JSON file
        data = {
            "expanded_region_area": expanded_area,
            "bounding_box_area": bounding_box_area,
            "total_window_area": total_window_area,
            "expanded_window_percentage": expanded_window_percentage,
            "bounding_box_window_percentage": bounding_box_window_percentage,
            "average_window_width": avg_window_width,
            "average_window_height": avg_window_height,
            "window_width_std_dev": std_window_width,
            "window_height_std_dev": std_window_height,
        }
    
        # Convert numpy data types to native Python types
        data = {key: (int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value) for key, value in data.items()}
    
        with open(output_path, "w") as file:
            json.dump(data, file, indent=4)
        print(f"Facade statistics saved to {output_path}")

    @staticmethod
    def visualize_expanded_region(image, region_mask, bounding_boxes, convex_hull, buffered_hull, rotated_box_points, output_path):
        """
        Visualize expanded regions and save the plot as an image file.
        """
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Prepare visualization images
        img_windows = image.copy()
        img_convex_hull = image.copy()
        img_buffered = image.copy()
        img_expanded = image.copy()

        # Draw bounding boxes, convex hulls, and regions
        for box in bounding_boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(img_windows, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if convex_hull is not None:
            cv2.polylines(img_convex_hull, [convex_hull.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)
        if buffered_hull is not None:
            cv2.polylines(img_buffered, [buffered_hull.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
        if contours:
            cv2.drawContours(img_expanded, contours, -1, (0, 255, 255), 2)

        # Plot images
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes[0, 0].imshow(cv2.cvtColor(img_windows, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original Image with Windows")
        axes[0, 1].imshow(cv2.cvtColor(img_convex_hull, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("Windows + Convex Hull")
        axes[1, 0].imshow(cv2.cvtColor(img_buffered, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Buffered Convex Hull")
        axes[1, 1].imshow(cv2.cvtColor(img_expanded, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title("Expanded Region")

        for ax in axes.ravel():
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(output_path)  # Save the plot to the specified path
        plt.close()
        print(f"Visualization saved to {output_path}")

    
  


   

    def process(self, image_folder, bbox_folder, output_folder):
        """
        Process images and bounding boxes from separate directories.
        """
        os.makedirs(output_folder, exist_ok=True)

        # Get image files from the folder
        image_files = self.get_image_files(image_folder)

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            txt_file = os.path.splitext(image_file)[0] + ".txt"
            txt_path = os.path.join(bbox_folder, txt_file)

            print(f"Processing {image_file}")

            # Read bounding boxes
            bounding_boxes = self.read_bounding_boxes(txt_path)

            # Skip processing if bounding boxes are missing
            if not bounding_boxes:
                print(f"No bounding boxes found for {image_file}. Skipping.")
                continue

            # Calculate convex hull and buffer it
            convex_hull = self.calculate_convex_hull(bounding_boxes)
            if convex_hull is None:
                print(f"Could not calculate convex hull for {image_file}. Skipping.")
                continue

            buffered_hull = self.buffer_convex_hull(convex_hull)

            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load image {image_file}. Skipping.")
                continue

            # Apply Gabor filters and calculate densities
            gabor_images = self.apply_gabor_filters(image)
            filter_densities = self.calculate_gabor_filter_densities(gabor_images, convex_hull)

            # Create expanded region mask
            expanded_region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(expanded_region_mask, [buffered_hull.astype(int)], 255)

            # Get rotated bounding box
            contours, _ = cv2.findContours(expanded_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea) if contours else None
            rotated_bounding_box = cv2.boxPoints(cv2.minAreaRect(largest_contour)) if largest_contour is not None else None
            rotated_bounding_box = rotated_bounding_box.astype(int) if rotated_bounding_box is not None else None

            # Save statistics to a JSON file
            output_json_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_facade_data.json")
            self.calculate_facade_statistics(image, expanded_region_mask, bounding_boxes, rotated_bounding_box, output_json_path)

            # Save visualization
            output_plot_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_visualization.png")
            self.visualize_expanded_region(image, expanded_region_mask, bounding_boxes, convex_hull, buffered_hull, rotated_bounding_box, output_plot_path)

            print(f"Processing completed for {image_file}.")
