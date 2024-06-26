import os
import json
import logging
from collections import deque

import av
import cv2
import mediapipe as mp
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def get_sample_aspect_ratio(video_path: str) -> float:
    container = av.open(video_path)
    video_stream = next(s for s in container.streams if s.type == 'video')
    return video_stream.sample_aspect_ratio.numerator / video_stream.sample_aspect_ratio.denominator


def process_video(video_path, subtitle_info, output_folder):
    logging.info(f"Processing video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_size_multiplier = get_sample_aspect_ratio(video_path)

    frame_rate_divisor = int(round(frame_rate / 12.5))  # target approximately 12.5 fps
    buffer_size = 120  # about 10 seconds of video
    bounding_box_buffer = deque(maxlen=buffer_size)

    def moving_average(bounding_box_buffer):
        avg_bbox = np.mean(bounding_box_buffer, axis=0)
        return avg_bbox

    for idx, segment in enumerate(subtitle_info):
        start_time = segment['start'] / 1000  # convert milliseconds to seconds
        end_time = segment['end'] / 1000  # convert milliseconds to seconds
        start_frame = int(start_time * frame_rate)
        end_frame = int(end_time * frame_rate)
        output_path = os.path.join(output_folder, f"{idx}.mp4")

        logging.info(f"Processing segment {idx}: {start_frame} to {end_frame} (frames)")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, frame_rate / frame_rate_divisor, (224, 224))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        i = start_frame

        while cap.isOpened() and i <= end_frame:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Frame read failed at frame {i}. Ending segment processing.")
                break

            if i % frame_rate_divisor == 0:
                results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
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

                    crop[start_y:end_y, start_x:end_x] = frame[max(0, top_left_y):src_end_y,
                                                         max(0, top_left_x):src_end_x]
                    crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
                    out.write(crop)

            i += 1

        logging.info(f"Finished processing segment {idx}")
        out.release()

    cap.release()


def main():
    dataset_path = "."  # Change to the actual dataset path
    subtitles_path = os.path.join(dataset_path, "annotations/subtitles.json")
    videos_path = os.path.join(dataset_path, "videos")
    output_base_path = os.path.join(dataset_path, "videos_processed")

    with open(subtitles_path, "r") as f:
        subtitles = json.load(f)

    for video_id, segments in subtitles.items():
        video_file_path = os.path.join(videos_path, f"{video_id}.mp4")
        output_folder = os.path.join(output_base_path, video_id)

        if not os.path.exists(video_file_path):
            logging.warning(f"Video file {video_file_path} does not exist. Skipping.")
            continue

        if os.path.exists(output_folder):
            logging.info(f"Output folder {output_folder} already exists. Skipping.")
            continue

        os.makedirs(output_folder, exist_ok=True)

        logging.info(f"Starting processing for video {video_id}")
        process_video(video_file_path, segments, output_folder)
        logging.info(f"Completed processing for video {video_id}")


if __name__ == "__main__":
    main()
