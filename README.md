# Building Analysis

## Overview

**BuildingAnalysis** is a Python-based project for detecting and analyzing facades of buildings using machine learning and image processing techniques. The pipeline includes image resizing, window detection, facade region growing, and output visualization. This repository demonstrates how to preprocess building images, extract facade information, and compute relevant statistics.

---

## Installation and Setup

### Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/oraibalmegdadi/BuildingAnalysis.git 
```
cd BuildingAnalysis


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
## Output

### Explanation of Output Visualization

The image showcases the key steps in the **Building Analysis Pipeline**, as follows:
![Example Output](https://github.com/oraibalmegdadi/BuildingAnalysis/blob/main/output/facade_data/Untitled3_visualization.png)


1. **Classification:**
   - The input building image is preprocessed and classified to identify whether it contains a facade suitable for analysis.
   - This ensures that only relevant images are processed further.

2. **Window Detection:**
   - The colored bounding boxes (e.g., red, green, blue, yellow) represent detected windows.
   - Detection is performed using a pre-trained YOLO model specifically trained to detect windows in building facades.

3. **Texture Analysis for Facade Estimation:**
   - The magenta outline represents the estimated facade region. 
   - This region is derived using **region-growing techniques** combined with **Gabor filter-based texture analysis**.
   - The facade area extends beyond the detected windows to cover the complete building facade.

4. **Floor Estimation:**
   - Windows are grouped into floors, with each floor visualized using distinct colors in the bounding boxes.
   - This grouping is based on vertical alignment and distance-based calculations to estimate floor boundaries.

5. **Statistics:**
   - ** Statistics:** A `.txt` and A `.json` files are generated containing detailed statistics, including:
     - Total number of windows.
     - Average window dimensions.
     - Floor height.
     - Window coverage percentage.
     - Facade area.
For more detailed output, including additional exmples, visualizations and structured datd, visit the output directory:

[Detailed Outputs](https://github.com/oraibalmegdadi/BuildingAnalysis/tree/main/output)

