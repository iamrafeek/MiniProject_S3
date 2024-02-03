import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image
from ultralytics import YOLO  # Make sure to have ultralytics YOLO package installed

# Initialize YOLO model
model = YOLO('best.onnx')  # Replace 'best.onnx' with the path to your ONNX model

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# License Plate Regular Expression Pattern
plate_pattern = re.compile(r'KL\d{2}[A-Z]{1,2}\d{3,4}', re.IGNORECASE)

# Function to clean OCR results
def clean_text(text):
    clean_pattern = re.compile(r'[^A-Z0-9]')
    return clean_pattern.sub('', text)

# Load predefined images (put the images in the same folder as your Streamlit script)
predefined_images = {
    'Image 1': r'car_data\car0.pngath_to_image1.jpg',
    'Image 2': r'car_data\car1.pngath_to_image1.jpg',
    'Image 3': r'car_data\car2.pngath_to_image1.jpg'
}

# Streamlit App Layout
st.title("Automated Vehicle Entry and Exit Tracking System")

# Radio button for predefined images
choice = st.radio("Choose an image source:", ('Upload Image', 'Predefined Images'))
image = None
if choice == 'Upload Image':
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif choice == 'Predefined Images':
    selected_image = st.radio("Select an image:", list(predefined_images.keys()))
    image_path = predefined_images[selected_image]
    image = Image.open(image_path)

if image is not None:
    st.image(image, caption='Selected Image', use_column_width=True)
    if st.button('Submit'):
        # Process the selected or uploaded image
        opencv_image = np.array(image)

        # Use YOLO model for vehicle detection
        results = model(opencv_image)
        for result in results:
            boxes = result.boxes.xywh
            confidences = result.boxes.conf

            for box, confidence in zip(boxes, confidences):
                if confidence >= 0.80:
                    x_center, y_center, width, height = box
                    xmin, ymin = x_center - (width / 2), y_center - (height / 2)
                    xmax, ymax = x_center + (width / 2), y_center + (height / 2)

                    # Crop the detected vehicle image for OCR
                    crop_img = opencv_image[int(ymin):int(ymax), int(xmin):int(xmax)]

                    # Perform OCR on the cropped image
                    ocr_results = reader.readtext(crop_img)
                    for result in ocr_results:
                        detected_text = clean_text(result[1].upper())
                        ocr_confidence = result[2]
                        if ocr_confidence >= 0.80 and plate_pattern.match(detected_text):
                            st.write(f"Detected Number Plate: {detected_text}")
                            st.write(f"OCR Confidence: {ocr_confidence}")
