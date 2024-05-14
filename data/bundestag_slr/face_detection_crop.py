from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np


def get_sample_aspect_ratio(video_path: str) -> float:
    # Open the video file with PyAV
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')

    return video_stream.sample_aspect_ratio.numerator / video_stream.sample_aspect_ratio.denominator


# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# OpenCV setup for video capture
video_path = "/Users/lubr/Downloads/CLSFBI0911A_S020_B_trim.mp4"
output_path = "/Users/lubr/Downloads/worked.mp4"
cap = cv2.VideoCapture(video_path)

# Output video setup
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

frame_rate_divisor = 4
frame_size_multiplier = get_sample_aspect_ratio(video_path)

# Buffer for storing recent bounding box coordinates for the moving average
buffer_size = 150  # half a second for 30 fps video
bounding_box_buffer = deque(maxlen=buffer_size)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, frame_rate / frame_rate_divisor, (224, 224))


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

    if i % frame_rate_divisor != 0:
        continue  # only every fourth frame

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
        max_dimension = max(smoothed_bbox[2], smoothed_bbox[3]) * 3.3  # width or height
        crop_size = int(max_dimension * frame_width)  # proportional size in pixels

        # Calculate top-left corner of the crop area
        top_left_x = int((smoothed_bbox[0] - max_dimension / 2) * frame_width)
        top_left_y = int((smoothed_bbox[1] - max_dimension / 2) * frame_height)

        # Create a blank image with the size of the crop
        crop = np.zeros((int(crop_size * frame_size_multiplier), crop_size, 3), dtype=np.uint8)

        # Calculate the coordinates of the frame on the blank image
        start_x = max(0, -top_left_x)
        start_y = max(0, -top_left_y)

        # Calculate end positions based on the start and the size of the blank image to avoid size mismatch
        end_y = start_y + min(frame_height - max(0, top_left_y), crop.shape[0] - start_y)
        end_x = start_x + min(frame_width - max(0, top_left_x), crop.shape[1] - start_x)

        # Adjust source slice to match destination size exactly
        src_end_y = max(0, top_left_y) + (end_y - start_y)
        src_end_x = max(0, top_left_x) + (end_x - start_x)

        # Copy valid parts of the original frame onto the blank image
        crop[start_y:end_y, start_x:end_x] = frame[max(0, top_left_y):src_end_y, max(0, top_left_x):src_end_x]

        # Resize to the desired size
        crop = cv2.resize(crop, (224, 224))

        # Write the cropped frame to the output video
        out.write(crop)

# Release resources
cap.release()
out.release()
