## Overview

**BuildingAnalysis** is a Python-based project for detecting and analyzing facades of buildings using machine learning and image processing techniques. The pipeline includes image resizing, window detection, facade region growing, and output visualization. This repository demonstrates how to preprocess building images, extract facade information, and compute relevant statistics.

---

## Installation and Setup
---

Ensure [Python](https://www.python.org/) is installed and properly configured on your device. This project was developed using Python 3.9.

### Clone the Repository
#### 1. Windows

1. Open CMD:

- Press Win + R, type cmd, and press Enter.

2. Navigate to the Desired Directory:


Use the cd command to move to the folder where you want to clone the repository. For example:
```bash
cd  < Desired Directory > 
```
3. Run the git clone Command:
Execute the git clone command to clone the repository:
```bash

git clone https://github.com/oraibalmegdadi/BuildingAnalysis.git
```

4. Navigate to the Cloned Repository:

Once the repository is cloned, navigate to it using:
```bash
cd BuildingAnalysis
```

5.  Create a virtual environment:
```bash
python -m venv <virtual enviroment name>   // ex: python -m venv Venv1
```
6. Activate the virtual environment (also for future usage)
```bash
Venv1\Scripts\activate
```



7. Install Dependencies
Install the required Python libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```


#### 2. Linux/MacOS


1. Open Terminal:

Press Ctrl + Alt + T or search for "Terminal" in your system applications.

2. Navigate to the Desired Directory:

Use the cd command to navigate to the folder where you want to clone the repository. For example:

```bash
cd <Desired Directory>
```

3. Run the git clone Command:
Clone the repository by running:
```bash
git clone https://github.com/oraibalmegdadi/BuildingAnalysis.git
```

4. Navigate to the Cloned Repository:

Move into the cloned repository:

```bash
cd BuildingAnalysis
```

5. Create a virtual environment:
```bash
python3 -m venv <virtual enviroment name> //ex: python -m venv Venv1
```

6. Activate the virtual environment (for future usage):

```bash
source Venv1/bin/activate
```

7. Install Dependencies:

Install the required Python libraries from the requirements.txt file:
```bash
pip install -r requirements.txt
```

** Additional Notes for Linux/WSL Users**
If you encounter an error related to libGL.so.1 when running the code, it means that OpenCV requires additional system libraries to function correctly. You can resolve this issue by installing the necessary library using the following command:

```bash
sudo apt update
sudo apt install -y libgl1-mesa-glx
```

---

### Running the Code
After setting up the environment, you can run the pipeline as follows:

```bash
python BuildingAnalysis.py <input>
```
**Input Folder**

* Replace **<input_folder>** with the directory containing:
  * Building Images: Clear images of building facades where most windows and facade borders are visible.
  * Locations.txt (Optional): A file specifying the geographical location of each image.

**Notes on Input**
* Image Resizing:
   * All images will be resized to 640x640 pixels automatically during processing.
   * For the best results, ensure images focus on a building facade with most windows and the facade borders clearly visible.
* Location File (Optional):
    * The Locations.txt file is optional but can enhance output by providing geographical data for each image, If used, the file should follow this format:
```bash
ImageName,Latitude,Longitude
1.jpg,38.71010,-9.13735
2.jpg,38.71100,-9.13800
```

**Example:**
If your input folder is named input, run the pipeline like this:

```bash
python BuildingAnalysis.py input
```

** This project pipeline processes the images and location data provided in the input directory and saves the results in the output folder. this process include Image Resizing, image classification, Window Detection, Facade Estimation, Output Generation.  The output directory contains three main subfolders:**

1. bounding_boxes folder:
   * This folder includes a text file for each input image.
   * Each file contains the bounding boxes for all detected windows within the image in the following format:
```bash
Class Confidence xmin ymin xmax ymax
where:
**Class:** The class of the detected object.
**Confidence:** The detection confidence score.
**xmin, ymin, xmax, ymax:** Coordinates of the bounding box for each window.

```
2. classification_results folder:
 * A single text file summarizing the classification results for all input images.
 * The file uses the following format:
```bash
Filename, Predicted_Class, Confidence
Where:
**Filename:** Name of the input image.
**Predicted_Class:** The architectural class of the building (e.g., Century17and18 or After1950).
**Confidence:** The confidence score for the predicted class.

```

Facade Data:

For each input image, this folder contains:
Statistics:
Detailed facade detection parameters, including:
Facade area.
Number of floors.
Number of windows.
Window dimensions (average width, height, etc.).
Window coverage percentage.
The statistics are saved in both JSON and text formats.
Visualizations:
Annotated images that include:
Detected windows (each floor represented with a unique color for clarity).
The estimated facade area (outlined in magenta).
The bounding box of the facade.







---

## Output



## Explanation of Output Visualization

The image showcases the key steps in the **Building Analysis Pipeline**, as follows:
![Example Output](https://github.com/oraibalmegdadi/BuildingAnalysis/blob/main/output/facade_data/Untitled3_visualization.png)

1. **Classification:**
- The input building image is preprocessed and classified to determine its architectural category.
- The classification is based on a pre-trained YOLOv8 classification model trained on images from Lisbon, captured using Google Street View and Google Earth.
- Two architectural categories are identified:
  1. **Century17and18**: Buildings constructed in the 17th and 18th centuries.
  2. **After1950**: Modern buildings constructed after 1950.

2. **Window Detection:**
- The colored bounding boxes (e.g., red, green, blue, yellow) represent detected windows in the facade.
- Detection is performed using a pre-trained YOLOv8 model for object detection.
- The YOLOv8 detection model was trained on **20K images**, specifically annotated for window detection, using bounding boxes as ground truth.
- Detected bounding boxes are then extracted for further facade analysis and statistics.

3. **Texture Analysis for Facade Estimation:**
- The magenta outline represents the estimated facade region.
- The facade region is derived using a region-growing algorithm, combining:
  - **Gabor filter-based texture analysis** to detect texture patterns.
  - **Color similarity metrics** to ensure consistency in the facade's appearance.
- The facade area extends beyond the detected windows to cover the complete facade surface.

4. **Floor Estimation:**
- Windows are grouped into floors, with each floor visualized using distinct colors in the bounding boxes.
- Grouping is performed based on:
  - Vertical alignment of windows.
  - Distance-based calculations to estimate the boundaries between floors.

5. **Statistics:**
- **Text and JSON Outputs**: Each analyzed image generates detailed statistics in `.txt` and `.json` formats. These include:
  - Location information (if provided in metadata through `Locations.txt`).
  - Total number of detected windows.
  - Average dimensions of windows (width and height).
  - Estimated number of floors and average floor height.
  - Facade area and window coverage percentage.


For more detailed output, including additional exmples, visualizations and structured datd, visit the output directory:

[Detailed Outputs](https://github.com/oraibalmegdadi/BuildingAnalysis/tree/main/output)

