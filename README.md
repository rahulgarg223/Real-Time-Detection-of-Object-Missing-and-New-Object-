# Real-Time Object Tracking System

This project implements a real-time video analytics pipeline that can detect:
1. Missing objects - when previously present objects are no longer visible
2. New object placement - when new objects appear in the scene

## Features

- Real-time object detection and tracking using YOLOv8
- FPS counter for performance monitoring
- Visual feedback with bounding boxes and object IDs
- Console output for missing and new object events
- Configurable confidence threshold

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv8

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python object_tracker.py
```

The script will:
- Open your default webcam (or you can modify the code to use a video file)
- Display the video feed with detected objects
- Print notifications when objects are missing or new objects appear
- Press 'q' to quit

## Performance Optimization

The system is optimized for real-time performance:
- Uses YOLOv8's efficient tracking capabilities
- Implements frame skipping for FPS calculation
- Avoids unnecessary video writing to disk
- Uses GPU acceleration when available

## Customization

You can modify the following parameters in the code:
- `model_path`: Change the YOLO model (default: 'yolov8n.pt')
- `conf_threshold`: Adjust detection confidence threshold (default: 0.5)
- Video source: Change `VideoCapture(0)` to use a different camera or video file 
=======
