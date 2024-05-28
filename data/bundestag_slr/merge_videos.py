import os
import typing

import cv2
import pandas as pd


def process_video(video_id) -> typing.List[dict]:
    video_dir = os.path.join('videos_processed', video_id)
    if not os.path.isdir(video_dir):
        print(f"Directory {video_dir} does not exist, skipping.")
        return []

    merged_video_path = os.path.join(video_dir, '../', f'{video_id}.mp4')
    if os.path.exists(merged_video_path):
        print(f"Video {merged_video_path} already exists, skipping.")
        return []

    out: cv2.VideoWriter = None

    subtitle_file = os.path.join(video_dir, 'subtitle.txt')
    with open(subtitle_file, 'r') as f:
        subtitles = f.readlines()

    merged_data = []
    next_frame_no = 0

    for idx, subtitle in enumerate(subtitles):
        video_file = os.path.join(video_dir, f'{idx}.mp4')
        if not os.path.isfile(video_file):
            print(f"Video file {video_file} not found, skipping.")
            continue

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Unable to open video file {video_file}, skipping.")
            continue

        if out is None:
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            out = cv2.VideoWriter(merged_video_path, fourcc, frame_rate, (width, height))

        data = {
            'VideoID': video_id,
            'SubtitleLine': subtitle.strip(),
            'StartFrame': next_frame_no,
        }

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out.write(frame)
            next_frame_no += 1

        data['EndFrame'] = next_frame_no - 1

        if (data['EndFrame'] - data['StartFrame']) < 25:
            print(f"{video_dir}: Segment {idx} is too short, skipping.")
            continue

        merged_data.append(data)
        cap.release()

    out.release()
    return merged_data


def main():
    with open('video_ids.txt', 'r') as f:
        video_ids = f.read().splitlines()

    merged_data = []

    for video_id in video_ids:
        data = process_video(video_id)
        merged_data.extend(data)

        print(f"Processed video {video_id}")
        pd.DataFrame(merged_data).to_csv('merged.csv', index=False)


if __name__ == '__main__':
    main()
