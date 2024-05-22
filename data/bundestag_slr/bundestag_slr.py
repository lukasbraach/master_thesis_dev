from typing import List, Dict

import cv2
import datasets
import numpy as np
import pandas as pd
from datasets import Sequence, Array3D, Value


def read_non_empty_lines(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file if line.strip()]
    return lines


def load_video(video_path):
    """Load video and return frames as a list of arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


class BundestagSLR(datasets.GeneratorBasedBuilder):
    """BUNDESTAG SLR: Continuous Sign Language Recognition Dataset."""

    VERSION = datasets.Version("1.0.0")
    DEFAULT_WRITER_BATCH_SIZE = 25

    def _info(self):
        features_dict = {
            "id": Value("string"),
            "subtitle": Value("string"),
            "frames": Sequence(Array3D(shape=(224, 224, 3), dtype="uint8")),
        }

        return datasets.DatasetInfo(
            features=datasets.Features(features_dict),
            # No default supervised_keys (as we have to pass both question
            # and context as input).
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        frames = {}
        other_data = {}

        data_csv = dl_manager.download("video_ids.txt")
        df = pd.read_csv(data_csv, header=None, names=['id'])

        video_ids_all = df['id']

        video_ids = {
            datasets.Split.TRAIN: video_ids_all[:int(len(video_ids_all) * 0.8)],
            datasets.Split.VALIDATION: video_ids_all[int(len(video_ids_all) * 0.8):int(len(video_ids_all) * 0.9)],
            datasets.Split.TEST: video_ids_all[int(len(video_ids_all) * 0.9):],
        }

        for split in [
            datasets.Split.TRAIN,
            datasets.Split.VALIDATION,
            datasets.Split.TEST,
        ]:
            video_subtitles = dl_manager.download([
                f"videos/{id}/subtitles.txt"
                for id in video_ids[split]
            ])

            subtitle_files_split = [
                dl_manager.iter_files(url)
                for url in video_subtitles
            ]

            video_file_names_split = []
            video_subtitles_split = []
            video_frames_split = []

            for idx, subtitle_file in zip(video_ids[split], subtitle_files_split):
                lines = read_non_empty_lines(subtitle_file)

                for i, subtitle_line in enumerate(lines):
                    video_file_name = f"videos/{idx}/{i}.mp4"

                    video_file_names_split.append(video_file_name)
                    video_frames_split.append(
                        dl_manager.download(video_file_name)
                    )
                    video_subtitles_split.append(subtitle_line)

            other_data_split = {}

            for video, file_name, subtitle, in zip(video_frames_split, video_file_names_split, video_subtitles_split):
                other_data_split[video] = {
                    "id": file_name,
                    "subtitle": subtitle,
                }

            other_data[split] = other_data_split
            frames[split] = video_frames_split

        return [
            datasets.SplitGenerator(
                name=split,
                gen_kwargs={
                    "videos": frames[split],
                    "other_data": other_data[split],
                },
            )
            for split in [
                datasets.Split.TRAIN,
                datasets.Split.VALIDATION,
                datasets.Split.TEST,
            ]
        ]

    def _generate_examples(self, videos: List[dict], other_data: Dict[dict, dict]):
        """
        _generate_examples generates examples for the HuggingFace dataset.
        It takes a list of frame_archives and the corresponding dict of other data.
        Each frame_archive acts as a key for the further data.

        :param frame_archives: list of ArchiveIterables
        :param other_data: Dict from ArchiveIterables to other data
        """
        for key, frames in enumerate(videos):
            ex = other_data[frames]

            result = {
                "id": ex['id'],
                "subtitle": ex['subtitle'],
                "frames": load_video(frames),
            }

            yield key, result
