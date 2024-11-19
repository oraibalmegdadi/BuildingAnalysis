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
            df = pd.read_csv(txt_file, sep='\s+', names=['Class', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax'], skiprows=1)
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
    def estimate_floor_count(rotated_bounding_box, bounding_boxes, avg_window_height):
        """
        Estimate the number of floors based on window height and vertical positions.
        """
        if not avg_window_height or not bounding_boxes:
            return 0, {}

        # Sort windows by their y-coordinate (top to bottom)
        bounding_boxes_sorted = sorted(bounding_boxes, key=lambda x: x[1])

        # Initialize floor grouping variables
        floors = {}
        current_floor = 1
        windows_per_floor = {current_floor: []}
        current_floor_y = bounding_boxes_sorted[0][1]  # Start with the y-coordinate of the first window
        floor_height_threshold = avg_window_height * 1.4  # Fixed threshold based on average window height

        # Maintain a set of already assigned windows to avoid re-assignment
        assigned_windows = set()

        # Assign windows to floors with proximity check
        for box in bounding_boxes_sorted:
            y_min = box[1]
            box_tuple = tuple(box)  # Convert box to a tuple to make it hashable

            # Check if the current window is already assigned
            if box_tuple in assigned_windows:
                # Find the floor it was previously assigned to
                for floor, windows in windows_per_floor.items():
                    if box in windows:
                        previous_floor = floor
                        break
                # Compare distances to decide if re-assignment is needed
                distance_to_current = abs(y_min - current_floor_y)
                distance_to_previous = abs(y_min - bounding_boxes_sorted[previous_floor - 1][1])

                if distance_to_current < distance_to_previous:
                    # Re-assign window to the current floor if closer
                    windows_per_floor[previous_floor].remove(box)
                    windows_per_floor[current_floor].append(box)
                continue  # Skip re-processing the already assigned window

            # If within threshold, assign to the current floor
            if abs(y_min - current_floor_y) < floor_height_threshold:
                windows_per_floor[current_floor].append(box)
            else:
                # Move to the next floor
                current_floor += 1
                windows_per_floor[current_floor] = [box]
                current_floor_y = y_min  # Reset y reference for the new floor

            # Mark this window as assigned
            assigned_windows.add(box_tuple)  # Add the tuple version of box to the set

        return current_floor, windows_per_floor

    @staticmethod
    def visualize_floors(image, expanded_region_mask, bounding_boxes, convex_hull, buffered_hull, rotated_box_points, num_floors, windows_per_floor, output_path):
        """
        Visualize expanded regions with floors and save the plot as an image file.
        """
        contours, _ = cv2.findContours(expanded_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for smoothed expanded region
        smoothed_mask = np.zeros_like(expanded_region_mask)
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.fillPoly(smoothed_mask, [approx], 255)

        # Prepare visualization image
        img_floors = image.copy()

        # Define floor colors
        floor_colors = [
            (255, 0, 0),     # Floor 1 - Red
            (0, 255, 0),     # Floor 2 - Green
            (0, 0, 255),     # Floor 3 - Blue
            (255, 255, 0),   # Floor 4 - Yellow
            (255, 0, 255),   # Floor 5 - Magenta
            (0, 255, 255)    # Floor 6 - Cyan
            (128, 0, 128),   # Floor 7 - Purple
            (128, 128, 0),   # Floor 8 - Olive
            (0, 128, 128),   # Floor 9 - Teal
            (128,128,128),  # Floor 10 - gray
        ]
        floor_colors += [(0, 0, 0)] * (num_floors - len(floor_colors))  # Add black for additional floors

        # Draw windows with floor colors
        for floor_idx, windows in windows_per_floor.items():
            color = floor_colors[floor_idx - 1]
            for box in windows:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(img_floors, (x_min, y_min), (x_max, y_max), color, 2)

        # Add rotated bounding box
        if rotated_box_points is not None:
            cv2.polylines(img_floors, [rotated_box_points], isClosed=True, color=(0, 165, 255), thickness=3)

        # Overlay smoothed expanded region
        smoothed_region_overlay = cv2.merge([smoothed_mask, smoothed_mask, smoothed_mask])
        img_floors = cv2.addWeighted(img_floors, 0.7, smoothed_region_overlay, 0.3, 0)

        # Save the visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(img_floors, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Floors with Different Colors + Rotated Box + Expanded Region (Smoothed)")
        plt.tight_layout()
        plt.savefig(output_path)
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

            # Apply Gabor filters
            gabor_images = self.apply_gabor_filters(image)

            # Create expanded region mask
            expanded_region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(expanded_region_mask, [buffered_hull.astype(int)], 255)

            # Get rotated bounding box
            contours, _ = cv2.findContours(expanded_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea) if contours else None
            rotated_bounding_box = cv2.boxPoints(cv2.minAreaRect(largest_contour)) if largest_contour is not None else None
            rotated_bounding_box = rotated_bounding_box.astype(int) if rotated_bounding_box is not None else None

            # Estimate floors
            avg_window_height = np.mean([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0
            num_floors, windows_per_floor = self.estimate_floor_count(rotated_bounding_box, bounding_boxes, avg_window_height)

            # Save visualization
            output_plot_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_visualization.png")
            self.visualize_floors(image, expanded_region_mask, bounding_boxes, convex_hull, buffered_hull, rotated_bounding_box, num_floors, windows_per_floor, output_plot_path)

            print(f"Processing completed for {image_file}.")
