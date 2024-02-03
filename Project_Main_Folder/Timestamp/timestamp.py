import cv2
import datetime
import pytesseract 
# # Open the video file for reading
# video_capture = cv2.VideoCapture('test1.mp4')

# # Get the video's frames per second (fps) and frame size
# fps = int(video_capture.get(cv2.CAP_PROP_FPS))
# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

# # Calculate the duration of each frame in seconds
# frame_duration = 1 / fps



# # Define the codec and create a VideoWriter object to save the output video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))
# # Initialize the starting timestamp
# current_time = datetime.datetime.now()

# while True:
#     ret, frame = video_capture.read()
#     if not ret:
#         break

#     # Add the timestamp text to the frame
#     cv2.putText(frame, current_time.strftime("%Y-%m-%d %H:%M:%S"), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

#     # Increment the timestamp based on frame duration
#     current_time += datetime.timedelta(seconds=frame_duration)

#     # Write the frame to the output video
#     output_video.write(frame)

#     cv2.imshow('Video Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release video capture and writer objects
# video_capture.release()
# output_video.release()
# cv2.destroyAllWindows()

# Open the output video file for reading
# Open the output video file for reading
# Open the output video file for reading
output_video_capture = cv2.VideoCapture('output_video.mp4')

while True:
    ret, frame = output_video_capture.read()
    if not ret:
        break

    # Extract the timestamp text from the frame
    timestamp_region = frame[50:100, 10:260]  # Adjust the coordinates to match your timestamp region
    timestamp_text = cv2.cvtColor(timestamp_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for OCR

    # Use OCR (Optical Character Recognition) to extract text from the timestamp region
    try:
        timestamp_str = pytesseract.image_to_string(timestamp_text, config='--psm 6', lang='eng')
        timestamp = datetime.datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M:%S")
        print("Timestamp:", timestamp)
    except ValueError:
        print("Error parsing timestamp")

    cv2.imshow('Output Video Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the output video capture object
output_video_capture.release()
cv2.destroyAllWindows()


