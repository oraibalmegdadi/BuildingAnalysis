# Building Analysis

## Overview

**BuildingAnalysis** is a Python-based project for detecting and analyzing facades of buildings using machine learning and image processing techniques. The pipeline includes image resizing, window detection, facade region growing, and output visualization. This repository demonstrates how to preprocess building images, extract facade information, and compute relevant statistics.

---

## Installation and Setup
---
Ensure **Python** (https://www.python.org/) is installed and properly configured on your device. This project was developed using Python 3.9.

### Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/oraibalmegdadi/BuildingAnalysis.git 
```
### Navigate to the BuildingAnalysis Directory
#### Windows

Use the following command in the Command Prompt (CMD):

```bash
cd BuildingAnalysis
```

#### Linux/MacOS
Use the following command in the Terminal:

```bash
cd BuildingAnalysis
```


### Activate the Virtual Environment

#### Windows

Create a virtual environment:
```bash
python -m venv venv
```

#### Linux/MacOS
Create a virtual environment:

```bash
python3 -m venv venv
```
Activate the virtual environment:

```bash
source venv/bin/activate

```
### Install Dependencies
Install the required Python libraries from the requirements.txt file:

```bash
pip install -r requirements.txt
```
---

### Running the Code
After setting up the environment, you can run the pipeline as follows:

```bash
python BuildingAnalysis.py <input>
```
Replace <input> with the directory containing your input images and the Locations.txt file.
Images will be resized to 640*640 pixels automaticlly. for best reutls we creommend  the images to focus on a building facade where most of windows and the facade borders are clear. 
location of each image is not mandotary. it could be added for more .... , the location represting each image could be included in  https://github.com/oraibalmegdadi/BuildingAnalysis/blob/main/input/Locations.txt   shuould be in the format: ImageName,Latitude,Longitude

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

