Vehicle Detection, Tracking, and Counting using YOLOv11

This repository implements a state-of-the-art vehicle detection, tracking, and counting system using YOLOv11 and the BDD100K dataset. It includes data preprocessing, model training, and real-time vehicle counting with ByteTrack. The project provides pretrained weights and results for easy reproduction and testing.

Dataset https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k

The BDD100K dataset is used for training and testing the model. The dataset contains three main folders:

Images

Labels

Segmentation (removed during project execution)

Images Folder

100K folder: Contains two subfolders—train (70,000 images), val (10,000 images with labels), and test (20,000 images without labels).

10K folder: Not used in this project and deleted for simplicity.

Labels Folder

train: Contains annotations for 70,000 images used for training.

val: Contains annotations for 10,000 images used for validation.

test: No annotations, contains 20,000 images used only for testing.

For more information, you can access and download the BDD100K dataset here
.

Features

Data Preprocessing: Convert BDD100K annotations (JSON format) into YOLO-compatible labels.

Vehicle Detection & Counting: Detect and count vehicles (cars, buses, trucks) in images and videos.

Tracking: Use ByteTrack for real-time vehicle tracking and counting.

Installation
Requirements


Install the dependencies:

pip install -r requirements.txt

Usage
Step 1: Dataset Conversion

To convert the BDD100K annotations to YOLO format labels, run the following:

from utils.data_utils import convert

convert(json_path="data/annotations/train.json", image_dir="data/train", output_label_dir="outputs/labels/train")

Step 2: Train the YOLO Model

Once the data is prepared, you can train the YOLOv5 model using the following script:

from ultralytics import YOLO

model = YOLO("models/yolo11s.pt")  # Use the pre-trained weights
model.train(data="configs/data.yaml", epochs=100, batch=16)

Step 3: Inference (Prediction)

For running inference on images or videos:

results = model.predict(source="path/to/image_or_video.jpg", conf=0.5, iou=0.5, save=True, save_txt=True)

Step 4: Vehicle Counting and Tracking (Real-time)

For real-time vehicle counting and tracking, use the following script:

import cv2
from ultralytics import YOLO

model = YOLO("models/best.pt")

cap = cv2.VideoCapture("path_to_video.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, classes=[0, 1, 2])
    # Additional code for drawing bounding boxes and tracking
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord("q"):
        break

Results

The model has been trained on the BDD100K dataset and evaluated using several metrics. The results, including training logs, metrics (precision, recall, mAP), and tracked outputs (images, videos), are stored in the results/ folder.

File Structure
vehicle-detection-yolo/
models/
    best.pt              # Pre-trained model weights
results/
    results.csv          # CSV file with performance metrics
    demo/                # Predicted outputs (images/videos)

configs/
    data.yaml            # YOLO dataset configuration
    args.yaml            # Training arguments

 notebooks/
    vehicle_detection.ipynb  # Jupyter notebook for experiments

outputs/                 
    demo/                # Demo images and videos with predictions
 README.md
 requirements.txt
 LICENSE

License

This project is licensed under the MIT License - see the LICENSE
 file for details.

Acknowledgments

YOLOv11: For providing an efficient and fast object detection framework.

ByteTrack: For robust object tracking implementation.

BDD100K: For providing a large-scale dataset for autonomous driving research.