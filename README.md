# Real-Time Indian Traffic Sign Detection System

This project is a real-time Indian Traffic Sign Detection and Classification system built using CNNs and OpenCV. It detects traffic signs through a live webcam feed and classifies them into 85 different Indian traffic sign categories.

The main goal of this project was to explore how deep learning and computer vision can be used in intelligent transportation systems and autonomous driving applications.


## What this project does

* Detects traffic signs in real time using a webcam
* Classifies 85 Indian traffic sign categories
* Draws bounding boxes with labels and confidence scores
* Uses HSV colour segmentation + CNN classification
* Works on normal systems without requiring a GPU


## Technologies Used

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* CNN (Convolutional Neural Networks)


## Dataset

The model was trained using the Indian Traffic Signs dataset from Kaggle.

Dataset details:

* 85 traffic sign classes
* Images in different lighting and angles
* Train/Test split provided


## How the system works

1. Webcam captures live frames
2. Frames are converted to HSV colour space
3. Red, blue, and yellow masks are created
4. Contours are detected to find possible traffic signs
5. Regions of interest are extracted
6. The CNN model predicts the traffic sign class
7. Bounding boxes and confidence scores are displayed on screen


## Model Details

The CNN model contains:

* 4 convolutional layers
* Global Average Pooling
* Dense layer with ReLU activation
* Dropout for regularization
* Softmax output layer for 85 classes

Input image size:
64 × 64


## Data Augmentation

To improve performance and reduce overfitting:

* Rotation
* Zoom
* Brightness variation
* Width/height shifts
* Shear transformations

Horizontal flipping was avoided because traffic signs are not symmetric.


## Results

* Test Accuracy: **89.52%**
* Test Loss: **0.6164**
* Real-time detection using webcam feed
* Runs on CPU without dedicated GPU support


## Challenges Faced

Some challenges during development:

* Similar-looking traffic signs caused confusion
* False detections from coloured objects
* Lower FPS on CPU systems
* Lighting conditions affecting detection quality


## Future Improvements

* Better optimization for FPS
* Improve low-light detection
* Use YOLO for better object detection
* Deploy on mobile or embedded systems


## Run the Project

Clone the repository:

```bash id="kq71sa"
git clone https://github.com/your-username/traffic-sign-detection.git
```

Install dependencies:

```bash id="i4tdv9"
pip install -r requirements.txt
```

Run the project:

```bash id="fgc2tl"
python main.py
```

---

## Author

Sameena B

Computer Science Student interested in AI/ML, Deep Learning, and Computer Vision.

