import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# OpenCV setup for video capture
video_path = "/Users/lubr/Downloads/CLSFBI0911A_S020_B.mp4"
output_path = "/Users/lubr/Downloads/worked.mp4"
cap = cv2.VideoCapture(video_path)

# Buffer for storing recent bounding box coordinates for the moving average
buffer_size = 15  # half a second for 30 fps video
bounding_box_buffer = deque(maxlen=buffer_size)

# Output video setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, frame_rate // 2, (224, 224))


# Function to calculate the moving average of bounding box coordinates
def moving_average(bounding_box_buffer):
    # Calculate average for center_x, center_y, width, height
    avg_bbox = np.mean(bounding_box_buffer, axis=0)
    return avg_bbox


# Process the video frame by frame
i = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    i += 1

    if i % 4 == 0:
        continue  # only every second frame

    # Perform face detection
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        # Normalize bounding box coordinates
        center_x = bbox.xmin + bbox.width / 2
        center_y = bbox.ymin + bbox.height / 2
        width = bbox.width
        height = bbox.height

        bounding_box_buffer.append([center_x, center_y, width, height])

        # Get the moving average bounding box
        smoothed_bbox = moving_average(bounding_box_buffer)

        # Determine the cropping square size
        max_dimension = max(smoothed_bbox[2], smoothed_bbox[3]) * 3  # width or height
        crop_size = int(max_dimension * frame_width)  # proportional size in pixels

        # Calculate top-left corner of the crop area
        top_left_x = int((smoothed_bbox[0] - max_dimension / 2) * frame_width)
        top_left_y = int((smoothed_bbox[1] - max_dimension / 2) * frame_height)

        # Ensure the cropping area doesn't exceed the frame boundaries
        top_left_x = max(0, min(top_left_x, frame_width - crop_size))
        top_left_y = max(0, min(top_left_y, frame_height - crop_size))

        # Crop the square region and resize to frame size
        crop = frame[top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]
        crop = cv2.resize(crop, (224, 224))

        # Write the cropped frame to the output video
        out.write(crop)

# Release resources
cap.release()
out.release()
