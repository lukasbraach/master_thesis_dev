from typing import List, Dict

import cv2
import datasets
import numpy as np
import pandas as pd
from datasets import Sequence, Array3D, Value

base_url = "."


class BundestagSLR(datasets.GeneratorBasedBuilder):
    """BUNDESTAG SLR: Continuous Sign Language Recognition Dataset."""

    VERSION = datasets.Version("1.0.0")
    DEFAULT_WRITER_BATCH_SIZE = 25

    def _info(self):
        features_dict = {
            "id": Value("string"),
            "subtitle": Value("string"),
            "frames": Sequence(Array3D(shape=(3, 224, 224), dtype="uint8")),
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

        data_csv = dl_manager.download(f"{base_url}/metadata.csv")
        df = pd.read_csv(data_csv, sep=",")

        video_ids_all = df['VideoID'].unique().tolist()

        video_ids = {
            datasets.Split.TRAIN: video_ids_all[:int(len(video_ids_all) * 0.9)],
            datasets.Split.VALIDATION: video_ids_all[int(len(video_ids_all) * 0.9):int(len(video_ids_all) * 0.95)],
            datasets.Split.TEST: video_ids_all[int(len(video_ids_all) * 0.95):],
        }

        for split in [
            datasets.Split.TRAIN,
            datasets.Split.VALIDATION,
            datasets.Split.TEST,
        ]:
            video_frames_split = []
            other_data_split = {}

            for idx in video_ids[split]:
                video_file_name = f"{base_url}/videos/{idx}.mp4"
                video = dl_manager.download(video_file_name)
                video_frames_split.append(video)

                video_examples = df[df['VideoID'] == idx]
                video_other_data = []

                for _, row in video_examples.iterrows():
                    video_other_data.append({
                        "id": idx,
                        "subtitle_line": row['SubtitleLine'],
                        "start_frame": int(row['StartFrame']),
                        "end_frame": int(row['EndFrame']),
                    })

                other_data_split[video] = video_other_data

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

    def _generate_examples(self, videos: List[any], other_data: Dict[dict, List[dict]]):
        """
        _generate_examples generates examples for the HuggingFace dataset.
        It takes a list of frame_archives and the corresponding dict of other data.
        Each frame_archive acts as a key for the further data.

        :param frame_archives: list of ArchiveIterables
        :param other_data: Dict from ArchiveIterables to other data
        """
        for key, video_path in enumerate(videos):
            examples = other_data[video_path]

            if len(examples) == 0:
                # no examples for this video, don't bother reading it
                continue

            cap = cv2.VideoCapture(video_path)

            current_read_frames = 0
            current_example_idx = 0
            frames = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                current_read_frames += 1
                ex = examples[current_example_idx]

                if current_read_frames < ex['start_frame']:
                    # skip until the start frame
                    continue

                if frames is None:
                    # initialize the frames numpy array to the final size
                    frames = np.ndarray((ex['end_frame'] - ex['start_frame'], *frame.shape))

                # save the read frame to the frames array
                frames[current_read_frames - ex['start_frame'] - 1] = frame

                if current_read_frames == ex['end_frame']:
                    # frames list is complete, yield the example

                    yield key, {
                        "id": ex['id'],
                        "subtitle": ex['subtitle_line'],
                        "frames": frames,
                    }

                    frames = None
                    current_example_idx += 1

                    if current_example_idx >= len(examples):
                        # no more examples.
                        break

            cap.release()
