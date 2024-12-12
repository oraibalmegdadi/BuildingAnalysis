import os
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import json


class FacadeEstimation:
    def __init__(self, buffer_factor=0.1):
        self.buffer_factor = buffer_factor

    
    @staticmethod
    def get_image_files(data_folder, extensions=("png", "jpg", "jpeg")):
        return [f for f in os.listdir(data_folder) if any(f.endswith(ext) for ext in extensions)]


    @staticmethod
    def read_bounding_boxes(txt_file):
        try:
            df = pd.read_csv(txt_file, sep='\s+', names=['Class', 'Confidence', 'xmin', 'ymin', 'xmax', 'ymax'], skiprows=1)
            return df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
        return []



    @staticmethod
    def read_location_file(location_file):
        """
        Reads a location file containing image names, latitudes, and longitudes.

        Parameters:
        - location_file: Path to the Location.txt file.

        Returns:
        - A dictionary with image names as keys and (latitude, longitude) as values.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            location_data = pd.read_csv(location_file)

            # Ensure required columns are present
            required_columns = {'ImageName', 'Latitude', 'Longitude'}
            if not required_columns.issubset(location_data.columns):
                raise ValueError(f"The file must contain the following columns: {', '.join(required_columns)}")

            # Convert the DataFrame into a dictionary
            location_dict = location_data.set_index('ImageName')[['Latitude', 'Longitude']].to_dict('index')

            # Format the dictionary to return latitude and longitude as tuples
            location_dict = {k: (v['Latitude'], v['Longitude']) for k, v in location_dict.items()}

            return location_dict

        except Exception as e:
            print(f"Error reading location file: {e}")
            return {}

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
        
        # Define 16 orientations for Gabor filters
        gabor_kernels = [
            cv2.getGaborKernel((21, 21), 5, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
            for theta in np.linspace(0, np.pi, 16, endpoint=False)
        ]
        
        # Apply each Gabor filter and store the result
        gabor_filtered_images = []
        for kernel in gabor_kernels:
            filtered_image = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
            gabor_filtered_images.append(filtered_image)
            #print(f"Gabor filter applied: {filtered_image.shape}")  # Debugging output
    
        # Check if the list is being populated
       # print(f"Number of Gabor-filtered images: {len(gabor_filtered_images)}")
        return gabor_filtered_images





   
    @staticmethod
    def calculate_gabor_filter_densities(gabor_filtered_images, convex_hull):
        """
        Calculate the average intensity (density) of Gabor filters in the convex hull area.
    
        Parameters:
        - gabor_filtered_images: List of Gabor-filtered images.
        - convex_hull: Points defining the convex hull.
    
        Returns:
        - filter_densities: List of average densities for each Gabor-filtered image.
        """
        mask = np.zeros(gabor_filtered_images[0].shape, dtype=np.uint8)
        cv2.fillPoly(mask, [convex_hull.astype(int)], 255)
    
        filter_densities = []
        for gabor_image in gabor_filtered_images:
            masked_image = cv2.bitwise_and(gabor_image, gabor_image, mask=mask)
            density = np.mean(masked_image[mask > 0])  # Mean intensity within the convex hull
            filter_densities.append(density)
        
        return filter_densities

    
    
    
    
    @staticmethod
    def calculate_average_color(image, mask_points):
        """
        Calculate the average color within a specified mask region of the image.
    
        Parameters:
        - image: The original image (BGR format).
        - mask_points: Points defining the mask region.
    
        Returns:
        - avg_color: A tuple representing the average color in (B, G, R) format.
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [mask_points.astype(int)], 255)
    
        # Mask the image and calculate the mean color of the masked area
        mean_color = cv2.mean(image, mask=mask)
        avg_color = mean_color[:3]  # Return the BGR color without the alpha component
        
        return avg_color

    
    
    #def region_growing(buffered_hull, gabor_filtered_images, filter_densities, image, average_color, threshold=20):
   # def region_growing(*, buffered_hull, gabor_filtered_images, filter_densities, image, average_color, image_height, min_y_threshold, max_y_threshold, threshold=20):
    def region_growing(self, *, buffered_hull, gabor_filtered_images, filter_densities, image, average_color, image_height, min_y_threshold, max_y_threshold, threshold=20):


        """
        Expand the region using Gabor filter similarity and color similarity.
    
        Parameters:
        - buffered_hull: Initial buffered convex hull points.
        - gabor_filtered_images: List of images filtered by Gabor kernels.
        - filter_densities: Densities of the Gabor-filtered regions.
        - image: The original input image (BGR format).
        - average_color: The average color in the initial buffered region.
        - threshold: Similarity threshold for expanding the region.
    
        Returns:
        - region_mask: A mask representing the expanded region.
        """
        height, width = gabor_filtered_images[0].shape
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(region_mask, [buffered_hull.astype(int)], 255)
    
        # Sort Gabor filters by density and select the top filters
        sorted_filters = sorted(range(len(filter_densities)), key=lambda i: filter_densities[i], reverse=False)
        best_gabor_images = [gabor_filtered_images[i] for i in sorted_filters[:5]]  # Top 5 Gabor filters
    
        # Initialize BFS queue and visited set
        queue = list(zip(*np.where(region_mask == 255)))
        visited = set(queue)
        added_pixels = 0  # Debugging counter for added pixels
    
        # Define 8-connectivity for neighboring pixels
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
        while queue:
            y, x = queue.pop(0)  # FIFO queue for breadth-first search
    
            for dy, dx in directions:
                ny, nx = y + dy, x + dx
    
                # Check if within bounds and not visited
                if 0 <= ny < height and 0 <= nx < width and (ny, nx) not in visited:
                    # Evaluate Gabor texture similarity
                    gabor_similarity = np.mean([
                        abs(best_gabor_image[ny, nx] - best_gabor_image[y, x])
                        for best_gabor_image in best_gabor_images
                    ])
    
                    # Evaluate color similarity
                    pixel_color = image[ny, nx]
                    color_similarity = np.linalg.norm(pixel_color - average_color)
    
                    # Combine similarities and expand region if below threshold
                    if (gabor_similarity < threshold) or (color_similarity < threshold * 1.3):
                        region_mask[ny, nx] = 255
                        queue.append((ny, nx))
                        visited.add((ny, nx))
                        added_pixels += 1  # Debugging counter for added pixels
    
        print(f"Region growing completed with {added_pixels} pixels added.")
        return region_mask


    
    def smooth_polygon(self, expanded_region_mask):
        contours, _ = cv2.findContours(expanded_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea) if contours else None
        if largest_contour is not None:
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            smoothed_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            return smoothed_polygon
        return None

    
    @staticmethod
    def visualize_results(image, bounding_boxes, rotated_box, floors, expanded_region):
        """
        Visualize the results:
        - Windows by floor (distinct colors).
        - Smoothed expanded region (outline).
        - Rotated bounding box (orange).
        """
        img = image.copy()
        
        # Define floor colors
        floor_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (128, 128, 0),
            (0, 128, 128), (128, 128, 128)
        ]
        floor_colors += [(0, 0, 0)] * 10  # Extra floors use black
        
        # Ensure expanded_region is valid
        if expanded_region is not None:
            if not isinstance(expanded_region, np.ndarray) or expanded_region.ndim != 2 or expanded_region.shape[1] != 2:
                raise ValueError("Expanded region must be a 2D NumPy array with shape (N, 2).")
            expanded_region = expanded_region.astype(np.int32)
        cv2.polylines(img, [expanded_region], isClosed=True, color=(255, 0, 255), thickness=2)  # Purple outline

        
        # Draw rotated bounding box
        if rotated_box is not None:
            cv2.polylines(img, [rotated_box.astype(int)], isClosed=True, color=(0, 165, 255), thickness=2)  # Orange outline

        # Draw windows by floor
        for floor_idx, windows in floors.items():
            color = floor_colors[floor_idx - 1] if floor_idx <= len(floor_colors) else (0, 0, 0)
            for box in windows:
                x_min, y_min, x_max, y_max = map(int, box)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
    
        return img



    
    @staticmethod
    def estimate_floor_count(bounding_boxes, avg_window_height):
        if not avg_window_height or not bounding_boxes:
            return 0, {}
    
        floors = {}
        current_floor = 1
        windows_per_floor = {current_floor: []}
        bounding_boxes_sorted = sorted(bounding_boxes, key=lambda x: x[1])
        current_floor_y = bounding_boxes_sorted[0][1]
    
        for box in bounding_boxes_sorted:
            y_min = box[1]
            if abs(y_min - current_floor_y) > avg_window_height * 1.4:
                current_floor += 1
                current_floor_y = y_min
                windows_per_floor[current_floor] = []
            windows_per_floor[current_floor].append(box)
    
        return len(windows_per_floor), windows_per_floor

    @staticmethod

    def calculate_and_save_statistics(output_path, bounding_boxes, expanded_region, rotated_box, image_name, location_dict):

        """
        Calculate and save statistics related to the facade in both text and JSON formats, including location data.
        """
        # Retrieve location data
        location = location_dict.get(image_name, "Unknown")
        location_str = f"Location: {location[0]}, {location[1]}" if location != "Unknown" else "Location: Unknown"
    
        # Calculate areas for expanded region and rotated bounding box
        expanded_region_area = cv2.contourArea(expanded_region) if expanded_region is not None else 0
        bounding_box_area = cv2.contourArea(rotated_box) if rotated_box is not None else 0
    
        # Calculate window areas and overall window coverage
        window_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in bounding_boxes]
        total_window_area = sum(window_areas)
    
        # Calculate facade coverage percentages
        expanded_window_percentage = (total_window_area / expanded_region_area) * 100 if expanded_region_area else 0
        bounding_box_window_percentage = (total_window_area / bounding_box_area) * 100 if bounding_box_area else 0
    
        # Calculate average window dimensions
        avg_window_width = np.mean([box[2] - box[0] for box in bounding_boxes]) if bounding_boxes else 0
        avg_window_height = np.mean([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0
        std_window_width = np.std([box[2] - box[0] for box in bounding_boxes]) if bounding_boxes else 0
        std_window_height = np.std([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0

    
        # Estimate number of floors and organize windows per floor
        avg_floor_height = avg_window_height * 1.1  # Adjusted floor height assumption
        num_floors, windows_per_floor = FacadeEstimation.estimate_floor_count(bounding_boxes, avg_floor_height)
    
        # Calculate floor-specific statistics
        floor_statistics = {}
        for floor, windows in windows_per_floor.items():
            floor_widths = [box[2] - box[0] for box in windows]
            floor_heights = [box[3] - box[1] for box in windows]
            avg_floor_width = np.mean(floor_widths) if floor_widths else 0
            avg_floor_height = np.mean(floor_heights) if floor_heights else 0
            avg_distance_between_windows = (
                np.mean([windows[i + 1][0] - windows[i][2] for i in range(len(windows) - 1)])
                if len(windows) > 1
                else 0
            )
            floor_statistics[floor] = {
                "num_windows": len(windows),
                "avg_width": avg_floor_width,
                "avg_height": avg_floor_height,
                "avg_distance": avg_distance_between_windows,
                "windows": windows,
            }
    
        # Prepare JSON data
        json_data = {
            "ImageName": image_name,
            "Location": {"Latitude": location[0], "Longitude": location[1]} if location != "Unknown" else "Unknown",
            "FacadeStatistics": {
                "ExpandedRegion": {
                    "AreaExcludingWindows": expanded_region_area - total_window_area,
                    "TotalArea": expanded_region_area,
                    "WindowCoveragePercentage": expanded_window_percentage,
                    "Vertices": expanded_region.tolist() if expanded_region is not None else None
                },
                "RotatedBoundingBox": {
                    "AreaExcludingWindows": bounding_box_area - total_window_area,
                    "TotalArea": bounding_box_area,
                    "WindowCoveragePercentage": bounding_box_window_percentage,
                    "Vertices": rotated_box.tolist() if rotated_box is not None else None
                },
                "Windows": {
                    "TotalArea": total_window_area,
                    "AverageWidth": avg_window_width,
                    "AverageHeight": avg_window_height,
                    "WidthStdDev": std_window_width,
                    "HeightStdDev": std_window_height,
                },
                "FloorEstimation": {
                    "EstimatedNumberOfFloors": num_floors,
                    "AverageFloorHeight": avg_floor_height,
                    "DetailedFloorStatistics": floor_statistics,
                }
            }
        }
    
        # Generate separate paths for text and JSON outputs
        text_file_path = f"{os.path.splitext(output_path)[0]}.txt"
        json_file_path = f"{os.path.splitext(output_path)[0]}.json"
    
        # Save results to the text file
        with open(text_file_path, 'w') as file:
            file.write(f"Facade Statistics of {image_name}, {location_str}\n")
            file.write("=================\n")
            file.write("Using Expanded Region:\n")
            file.write(f"- Area of Expanded Region (excluding windows): {expanded_region_area - total_window_area:.2f} pixels\n")
            file.write(f"- Total Window Area: {total_window_area:.2f} pixels\n")
            file.write(f"- Window Coverage Percentage: {expanded_window_percentage:.2f}%\n")
            if expanded_region is not None:
                file.write(f"- Vertices of Expanded Region: {expanded_region.tolist()}\n\n")
    
            file.write("Using Rotated Bounding Box:\n")
            file.write(f"- Area of Bounding Box (excluding windows): {bounding_box_area - total_window_area:.2f} pixels\n")
            file.write(f"- Total Window Area: {total_window_area:.2f} pixels\n")
            file.write(f"- Window Coverage Percentage: {bounding_box_window_percentage:.2f}%\n")
            if rotated_box is not None:
                file.write(f"- Vertices of Rotated Bounding Box: {rotated_box.tolist()}\n\n")
    
            file.write("Window Statistics:\n")
            file.write(f"- Average Window Width: {avg_window_width:.2f} pixels\n")
            file.write(f"- Average Window Height: {avg_window_height:.2f} pixels\n")
            file.write(f"- Window Width Std Dev: {std_window_width:.2f} pixels\n")
            file.write(f"- Window Height Std Dev: {std_window_height:.2f} pixels\n\n")
    
            file.write("Floor Estimation:\n")
            file.write(f"- Estimated Number of Floors: {num_floors}\n")
            file.write(f"- Average Floor Height: {avg_floor_height:.2f} pixels\n\n")
    
            file.write("Detailed Floor Statistics:\n")
            for floor, stats in floor_statistics.items():
                file.write(f"  Floor {floor}:\n")
                file.write(f"    - Number of Windows: {stats['num_windows']}\n")
                file.write(f"    - Average Width: {stats['avg_width']:.2f} pixels\n")
                file.write(f"    - Average Height: {stats['avg_height']:.2f} pixels\n")
                file.write(f"    - Average Distance Between Windows: {stats['avg_distance']:.2f} pixels\n")
                windows_str = ", ".join([f"[{', '.join(map(str, map(float, w)))}]" for w in stats["windows"]])
                file.write(f"    - Windows: {windows_str}\n")
            file.write("=================\n")
    
        # Save results to the JSON file
        with open(json_file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

            
    def process(self, image_folder, bbox_folder, output_folder, location_file):
        
        """
        Processes images and bounding boxes to estimate facades and generate results.
        """
        os.makedirs(output_folder, exist_ok=True)


        # Load location data
        location_dict = {}
        if os.path.exists(location_file):
            location_dict = self.read_location_file(location_file)
        else:
            print(f"Warning: Locations.txt not found in {os.path.dirname(location_file)}. Location data will not be included.")


        # Get image files
        image_files = self.get_image_files(image_folder)
    
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            txt_file = os.path.splitext(image_file)[0] + ".txt"
            txt_path = os.path.join(bbox_folder, txt_file)
    
            print(f"Processing {image_file}")
    
            # Read bounding boxes
            bounding_boxes = self.read_bounding_boxes(txt_path)
            if not bounding_boxes:
                print(f"No bounding boxes found for {image_file}. Skipping.")
                continue
    
            # Calculate convex hull
            convex_hull = self.calculate_convex_hull(bounding_boxes)
            if convex_hull is None:
                print(f"Could not calculate convex hull for {image_file}. Skipping.")
                continue
    
            # Buffer the convex hull
            buffered_hull = self.buffer_convex_hull(convex_hull)
    
            # Apply Gabor filters
            image = cv2.imread(image_path)
            if image is None:
                print(f"Image {image_file} could not be loaded. Skipping.")
                continue
    
            gabor_filtered_images = self.apply_gabor_filters(image)
            #print(f"Debug: Number of Gabor-filtered images: {len(gabor_filtered_images)}")
    
            # Ensure Gabor filters were successfully applied
            if not gabor_filtered_images:
                raise ValueError("Failed to apply Gabor filters. Gabor-filtered images list is empty.")
    
            # Calculate filter densities
            filter_densities = self.calculate_gabor_filter_densities(gabor_filtered_images, convex_hull)
    
            # Calculate average color
            average_color = self.calculate_average_color(image, buffered_hull)
    
            # Define vertical thresholds for region growing
            min_y_threshold = min(box[1] for box in bounding_boxes) + 150
            max_y_threshold = max(box[3] for box in bounding_boxes) + 150
    
            # Perform region growing


            try:

                expanded_region_mask = self.region_growing(
                buffered_hull=buffered_hull,
                gabor_filtered_images=gabor_filtered_images,
                filter_densities=filter_densities,
                image=image,
                average_color=average_color,
                image_height=image.shape[0],
                min_y_threshold=min_y_threshold,
                max_y_threshold=max_y_threshold,
                threshold=20 
                )


                
              #  expanded_region_mask = self.region_growing(
                  #  buffered_hull=buffered_hull,
                   # gabor_filtered_images=gabor_filtered_images,
                    #filter_densities=filter_densities,
                    #image=image,
                    #average_color=average_color,
                    #image_height=image.shape[0],
                    #min_y_threshold=min_y_threshold,
                    #max_y_threshold=max_y_threshold,
                    #threshold=20
              #  )
            except ValueError as e:
                print(f"Error during region growing: {e}")
                continue
    
            # Convert expanded_region_mask to polygon
            contours, _ = cv2.findContours(expanded_region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            largest_contour = max(contours, key=cv2.contourArea) if contours else None
            rotated_bounding_box = (
                cv2.boxPoints(cv2.minAreaRect(largest_contour)).astype(int)
                if largest_contour is not None
                else None
            )
            expanded_region = (
                largest_contour.reshape(-1, 2)
                if largest_contour is not None and len(largest_contour) > 2
                else None
            )
    
            # Validate expanded_region
            if expanded_region is None:
                print(f"Could not extract expanded region for {image_file}. Skipping.")
                continue
    
            # Estimate floor count and organize windows
            avg_window_height = np.mean([box[3] - box[1] for box in bounding_boxes]) if bounding_boxes else 0
            num_floors, windows_per_floor = self.estimate_floor_count(bounding_boxes, avg_window_height)
    
            # Visualize results
            visualization_image = self.visualize_results(
                image=image,
                bounding_boxes=bounding_boxes,
                rotated_box=rotated_bounding_box,
                floors=windows_per_floor,
                expanded_region=expanded_region
            )
    
            # Save visualization
            output_plot_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_visualization.png")
            cv2.imwrite(output_plot_path, visualization_image)
            print(f"Visualization saved to {output_plot_path}")



            # Pass the location_dict to calculate_and_save_statistics
            output_stats_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_statistics.txt")
            self.calculate_and_save_statistics(
                output_path=output_stats_path,
                bounding_boxes=bounding_boxes,
                expanded_region=expanded_region,
                rotated_box=rotated_bounding_box,
                image_name=image_file,  # Make sure this matches the key in location_dict
                location_dict=location_dict
            )
    
            print(f"Processing completed for {image_file}. Results saved.")
