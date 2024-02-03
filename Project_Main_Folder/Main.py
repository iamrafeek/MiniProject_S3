import cv2
import pandas as pd
import re
from datetime import datetime, timedelta
from ultralytics import YOLO
import easyocr
import numpy as np
import time

# Initialize YOLO and EasyOCR
model = YOLO('best.onnx')
reader = easyocr.Reader(['en'])

# Define a virtual line for entry/exit detection
LINE_POSITION = 100

# Regular Expression Pattern for Number Plate
plate_pattern = re.compile(r'KL\d{2}[A-Z]{1,2}\d{3,4}', re.IGNORECASE)
clean_pattern = re.compile(r'[^A-Z0-9]')  # Pattern to clean the detected text

# Function to clean OCR results
def clean_text(text):
    return clean_pattern.sub('', text)

# Function to check if a car crosses the line
def check_line_crossing(y, line_position):
    return y > line_position

# Function to preprocess the image for OCR
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Initialize Video Capture
video_path = r"Vid\test1.MP4"
cap = cv2.VideoCapture(video_path)

# Set the desired FPS
# desired_fps = 60  # Adjust this value as needed

# Timing variables
prev_frame_time = 0

cv2.namedWindow("YOLOv8 Inference", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLOv8 Inference", 1920, 1080)

car_records = {}

entry_timestamps = {}  # Store entry timestamps for each detected car

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Frame not read successfully")
        break

    print("Processing new frame")
    current_time = datetime.now()  # Use current system time
    print(f"Current Frame Timestamp: {current_time}")

    cv2.line(frame, (0, LINE_POSITION), (frame.shape[1], LINE_POSITION), (255, 0, 0), 2)
    results = model(frame)

    for result in results:
        boxes = result.boxes.xywh
        confidences = result.boxes.conf

        for box, confidence in zip(boxes, confidences):
            if confidence >= 0.80:
                x_center, y_center, width, height = box
                xmin, ymin = x_center - (width / 2), y_center - (height / 2)
                xmax, ymax = x_center + (width / 2), y_center + (height / 2)

                cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
                cv2.putText(frame, f"Conf: {confidence:.2f}", (int(xmin), int(ymin - 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                crop_img = frame[int(ymin):int(ymax), int(xmin):int(xmax)]
                preprocessed_img = preprocess_image(crop_img)

                if check_line_crossing(y_center, LINE_POSITION):
                    print("Object passed the line")
                    ocr_results = reader.readtext(preprocessed_img)
                    number_plate_detected = False

                    for result in ocr_results:
                        detected_text = clean_text(result[1].upper())
                        ocr_confidence = result[2]

                        if ocr_confidence >= 0.50:  # Check if OCR confidence is 80% or higher
                            print("Object underwent OCR")
                            print(detected_text)
                            if plate_pattern.match(detected_text):
                                number_plate_detected = True
                                timestamp = current_time  # Use current system time

                                # Check if the car record exists and update accordingly
                                car_record = car_records.get(detected_text, {})
                                entry_time = entry_timestamps.get(detected_text)

                                if entry_time is None:
                                    entry_timestamps[detected_text] = timestamp
                                    car_record['Entry'] = timestamp
                                    car_record['OCR Confidence'] = ocr_confidence
                                elif timestamp - entry_time > timedelta(minutes=1):
                                    car_record['Exit'] = timestamp
                                    car_record['Exit OCR Confidence'] = ocr_confidence
                                    del entry_timestamps[detected_text]
                                car_records[detected_text] = car_record
                                display_text = f"{detected_text} (Conf: {confidence:.2f})"
                                cv2.putText(frame, display_text, (int(xmin), int(ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                                print(f"Car Number: {detected_text}, Confidence: {confidence:.2f}, OCR Confidence: {ocr_confidence:.2f}, Time: {timestamp}")

                    if not number_plate_detected:
                        print("OCR did not manage to detect the number plate with sufficient confidence.")

    cv2.imshow("YOLOv8 Inference", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Exiting loop")
        break


print("Saving car records to Excel")
# Create a DataFrame with the modified data structure
df = pd.DataFrame.from_dict(car_records, orient='index')
df.to_excel("car_records.xlsx")
print("Releasing resources")
cap.release()
cv2.destroyAllWindows()
