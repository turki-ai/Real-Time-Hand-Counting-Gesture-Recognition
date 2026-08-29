# Real-Time Hand Counting Gestures Recognition Pipeline

## Overview
A lightweight, real-time hand gesture recognition system designed to classify dynamic hand gestures (0–5 finger counts) with zero perceptible latency. The project features an end-to-end pipeline: from automated data extraction and geometric augmentation using MediaPipe and OpenCV, to training a multi-class neural network in TensorFlow, and finally converting it to TensorFlow Lite for optimized inference.

## Key Features
* **Smart Gesture Counting:** Independently tracks both left and right hands, classifies gestures (0-5), and computes the total sum in real-time.
* **Automated Data Pipeline:** Extracts skeletal landmarks from raw `.mp4` videos and automatically generates geometric augmentations (rotation, scaling, flipping) to enrich the dataset.
* **Smart Training Trigger:** The system checks for the existence of the model; if missing, it automatically triggers the data collection and training pipeline before launching the live stream.
* **High Performance:** Achieves 30+ FPS real-time live video inference with >95% classification accuracy using a lightweight TFLite model to eliminate inference overhead.

## Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe Tasks API
* **Deep Learning:** TensorFlow, Keras, TensorFlow Lite
* **Data Manipulation:** Pandas, NumPy

## Project Structure
```text
├── main.py                        # Main execution file (Live stream & inference)
├── hand_video_data_collector.py   # Extracts landmarks from videos & applies augmentation
├── model_generator.py             # Builds, trains, and converts the Neural Network to TFLite
├── drawing.py                     # Utility functions for rendering landmarks and logic handling
├── csv_folder/                    # Auto-generated directory for extracted CSV datasets
├── videos/                        # Directory containing training gesture videos (.mp4)
└── counting_gestures_model.tflite # The finalized lightweight model (Auto-generated)
