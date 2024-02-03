# MiniProject_S3
This Repository Contain Mini Project files during my Semester 3 Accademics
# Vehicle Monitoring System

## Introduction
This project is a vehicle monitoring system developed as a Mini Project for Semester 3 Academics. It utilizes computer vision and OCR (Optical Character Recognition) to detect vehicles and read license plates from video feeds. The project employs a YOLO (You Only Look Once) object detection model and EasyOCR for text recognition.

## Features
- Vehicle detection using YOLO model.
- License plate recognition with EasyOCR.
- Virtual line crossing detection for vehicle entry/exit tracking.
- Logging of vehicle records to an Excel spreadsheet.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.6 or higher
- OpenCV library
- pandas library
- EasyOCR library
- YOLO model compatible with Ultralytics YOLO implementation

To install all required Python packages, you can use the `requirements.txt` file:

pip install -r requirements.txt

## To run the main script, navigate to the Project_Main_Folder directory and run:

python Main.py

## For the Streamlit application, navigate to the streamlit directory and run:

streamlit run app.py
