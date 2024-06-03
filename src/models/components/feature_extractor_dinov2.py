from typing import List, Union, Optional

import numpy as np
from PIL.Image import Image
from transformers import SequenceFeatureExtractor, BitImageProcessor, BatchFeature
from transformers.utils import logging

logger = logging.get_logger(__name__)


class SignLanguageFeatureExtractor(SequenceFeatureExtractor):
    r"""
    Constructs a SignLanguage feature extractor.

    This feature extractor inherits from [`~feature_extraction_sequence_utils.SequenceFeatureExtractor`] which contains
    most of the main methods. Users should refer to this superclass for more information regarding those methods.

    This class was adapted from transformers.Wav2Vec2FeatureExtractor.

    Args:
        feature_size (`int`, defaults to 224):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        padding_value (`float`, defaults to 0.0):
            The value that is used to fill the padding values.
        return_attention_mask (`bool`, *optional*, defaults to `False`):
            Whether or not [`~SignLanguageFeatureExtractor.__call__`] should return `attention_mask`.
    """

    model_input_names = ["input_values", "attention_mask"]

    def __init__(
            self,
            sampling_rate=25,
            return_every_nth_element=1,
            **kwargs,
    ):
        super().__init__(
            feature_size=np.NaN,  # irrelevant
            padding_value=0.0,
            sampling_rate=sampling_rate, **kwargs
        )
        self.return_every_nth_element = return_every_nth_element

        self._image_processor = BitImageProcessor(
            do_convert_rgb=True,
            do_normalize=True,
            do_resize=True,
            do_center_crop=False,
            size={"width": 224, "height": 224},
            image_std=[0.229, 0.224, 0.225],
            image_mean=[0.485, 0.456, 0.406],
            resample=3,
        )

    def __call__(
            self,
            raw_frames: Union[np.ndarray, List[np.ndarray], List[List[Image]]],
            sampling_rate: Optional[int] = None,
            **kwargs,
    ) -> BatchFeature:
        """
        Main method to featurize and prepare for the model one or several sequence(s).

        Args:
            raw_frames (`np.ndarray`, `List[Image]`, `List[np.ndarray]`, `List[List[Image]]`):
                The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float
                values, a list of numpy arrays or a list of list of Image values.
            sampling_rate (`int`, *optional*):
                The sampling rate at which the `raw_frames` input was sampled. It is strongly recommended to pass
                `sampling_rate` at the forward call to prevent silent errors.
        """

        if sampling_rate is not None:
            if sampling_rate != self.sampling_rate:
                raise ValueError(
                    f"The model corresponding to this feature extractor: {self} was trained using a sampling rate of"
                    f" {self.sampling_rate}. Please make sure that the provided `raw_frames` input was sampled with"
                    f" {self.sampling_rate} and not {sampling_rate}."
                )
        else:
            logger.warning(
                "It is strongly recommended to pass the ``sampling_rate`` argument to this function. "
                "Failing to do so can result in silent errors that might be hard to debug."
            )

        is_batched_numpy = isinstance(raw_frames, np.ndarray) and len(raw_frames.shape) > 1
        if is_batched_numpy and len(raw_frames.shape) != 5:
            raise ValueError(
                f"Input raw_frames has wrong dimension: expected dimension 5 (batch_no, video_frame_no, 3)")

        is_batched = is_batched_numpy or (
                isinstance(raw_frames, (list, tuple)) and (isinstance(raw_frames[0], (np.ndarray, tuple, list)))
        )

        x = raw_frames

        # always return batch
        if not is_batched:
            x = [x]

        def extract(frame_batch):
            frame_batch = frame_batch[::self.return_every_nth_element]
            processed = self._image_processor(images=frame_batch, return_tensors='np')

            return processed['pixel_values']

        x = [
            extract(frame_batch)
            for frame_batch in x
        ]

        data = {"pixel_values": x}
        return BatchFeature(data=data)
