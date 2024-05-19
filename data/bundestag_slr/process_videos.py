import csv
import logging
import os
from collections import deque
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import av
import cv2
import mediapipe as mp
import numpy as np
import pysrt
from pysrt import SubRipFile, SubRipItem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add time around the subtitle to account for alignment issues
pre_start_seconds = 0
post_end_seconds = 1.5

# Max thread workers
max_workers = 16

# Function to initialize a MediaPipe FaceDetection instance
def create_face_detection():
    mp_face_detection = mp.solutions.face_detection
    return mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

face_detectors = [create_face_detection() for _ in range(max_workers)]

def get_subtitle_items(subs: SubRipFile) -> List[SubRipItem]:
    items: List[SubRipItem] = []

    for sub in subs:
        sub.text = sub.text.replace('\n', ' ')
        items.append(sub)

    return items

def get_sample_aspect_ratio(video_path: str) -> float:
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')
    return video_stream.sample_aspect_ratio.numerator / video_stream.sample_aspect_ratio.denominator

def crop_signer(frame):
    height, width, _ = frame.shape
    right_half = frame[:, int(3 * width / 5):]
    return right_half

def moving_average(bounding_box_buffer):
    avg_bbox = np.mean(bounding_box_buffer, axis=0)
    return avg_bbox

def process_segment(video_path: str, subtitle_item: SubRipItem, frame_rate: float, frame_width: int, frame_height: int, frame_size_multiplier: float, face_detection, output_folder: str, idx: int):
    start_time = subtitle_item.start.ordinal / 1000  # convert milliseconds to seconds
    end_time = subtitle_item.end.ordinal / 1000  # convert milliseconds to seconds

    start_time = start_time - pre_start_seconds
    end_time = end_time + post_end_seconds

    start_frame = int(start_time * frame_rate)
    end_frame = int(end_time * frame_rate)
    output_path = os.path.join(output_folder, f"{idx}.mp4")

    logging.info(f"Processing segment {idx}: {start_frame} to {end_frame} (frames)")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, frame_rate / 12.5, (224, 224))

    buffer_size = 120  # about 10 seconds of video
    bounding_box_buffer = deque(maxlen=buffer_size)
    frame_rate_divisor = int(round(frame_rate / 12.5))  # target approximately 12.5 fps

    i = start_frame
    processing_failed = False

    while cap.isOpened() and i <= end_frame:
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame read failed at frame {i}. Ending segment processing.")
            processing_failed = True
            break

        if i % frame_rate_divisor == 0:
            signer_frame = crop_signer(frame)
            width_diff_rel = (frame.shape[1] - signer_frame.shape[1]) / frame.shape[1]

            results = face_detection.process(cv2.cvtColor(signer_frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                detection = results.detections[0]
                bbox = detection.location_data.relative_bounding_box

                bbox.xmin = width_diff_rel + bbox.xmin * (1 - width_diff_rel)
                bbox.width = bbox.width * (1 - width_diff_rel)

                center_x = bbox.xmin + bbox.width / 2
                center_y = bbox.ymin + bbox.height / 2
                width = bbox.width
                height = bbox.height

                bounding_box_buffer.append([center_x, center_y, width, height])
                smoothed_bbox = moving_average(bounding_box_buffer)
                max_dimension = max(smoothed_bbox[2], smoothed_bbox[3]) * 3.3
                crop_size = int(max_dimension * frame_width)

                top_left_x = int((smoothed_bbox[0] - max_dimension / 2) * frame_width)
                top_left_y = int((smoothed_bbox[1] - max_dimension / 2) * frame_height)
                crop = np.zeros((int(crop_size * frame_size_multiplier), crop_size, 3), dtype=np.uint8)

                start_x = max(0, -top_left_x)
                start_y = max(0, -top_left_y)
                end_y = start_y + min(frame_height - max(0, top_left_y), crop.shape[0] - start_y)
                end_x = start_x + min(frame_width - max(0, top_left_x), crop.shape[1] - start_x)
                src_end_y = max(0, top_left_y) + (end_y - start_y)
                src_end_x = max(0, top_left_x) + (end_x - start_x)

                crop[start_y:end_y, start_x:end_x] = frame[max(0, top_left_y):src_end_y, max(0, top_left_x):src_end_x]
                crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                out.write(crop)

        i += 1

    out.release()
    cap.release()

    if processing_failed:
        if os.path.exists(output_path):
            os.remove(output_path)
        return None
    else:
        return subtitle_item.text

def process_video(video_path: str, subtitle_path: str, output_folder: str):
    logging.info(f"Processing video: {video_path}")

    try:
        frame_size_multiplier = get_sample_aspect_ratio(video_path)
    except Exception as e:
        logging.error("Failed to open video file. Deleting video file and results directory.")
        if os.path.exists(output_folder):
            os.remove(output_folder)
        if os.path.exists(video_path):
            os.remove(video_path)
        return

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    subtitle_list = []
    subs = pysrt.open(subtitle_path)
    subs_items = get_subtitle_items(subs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sub in enumerate(subs_items):
            face_detector = face_detectors[idx % len(face_detectors)]
            futures.append(executor.submit(process_segment, video_path, sub, frame_rate, frame_width, frame_height, frame_size_multiplier, face_detector, output_folder, idx))

        for future in as_completed(futures):
            result = future.result()
            if result:
                subtitle_list.append(result)

    with open(os.path.join(output_folder, "subtitles.txt"), 'w') as file:
        for sub in subtitle_list:
            file.write(sub + "\n")

def main():
    dataset_path = "."  # Change to the actual dataset path
    video_ids_path = os.path.join(dataset_path, "video_ids.csv")
    videos_path = os.path.join(dataset_path, "raw")
    output_base_path = os.path.join(dataset_path, "videos_processed")

    with open(video_ids_path, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            video_id = row['VideoID']
            print(f"Processing video ID: {video_id}")

            video_file_path = os.path.join(videos_path, f"{video_id}.mp4")
            subtitle_file_path = os.path.join(videos_path, f"{video_id}.srt")
            output_folder = os.path.join(output_base_path, video_id)

            if not os.path.exists(video_file_path):
                logging.warning(f"Video file {video_file_path} does not exist. Skipping.")
                continue

            if not os.path.exists(subtitle_file_path):
                logging.warning(f"Subtitle file {video_file_path} does not exist. Skipping.")
                continue

            if os.path.exists(output_folder):
                logging.info(f"Output folder {output_folder} already exists. Skipping.")
                continue

            os.makedirs(output_folder, exist_ok=True)

            logging.info(f"Starting processing for video {video_id}")
            process_video(video_file_path, subtitle_file_path, output_folder)
            logging.info(f"Completed processing for video {video_id}")

if __name__ == "__main__":
    main()
