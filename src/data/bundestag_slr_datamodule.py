from typing import Any, Dict, Optional, Union

import datasets
import numpy as np
import torch
from datasets import IterableDataset, Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import BitImageProcessor, PreTrainedTokenizer, PreTrainedTokenizerFast


def next_lower_power_of_2(n):
    if n < 1:
        raise ValueError("Input must be a positive integer")
    # Check if n is already a power of 2
    if (n & (n - 1)) == 0:
        return n
    # Find the next lower power of 2
    return 1 << (n.bit_length() - 1)


class BundestagSLRVideoMAEDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_source='lukasbraach/bundestag_slr',
            dataset_variant=None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
            batch_size: int = 16,
            num_workers: int = 16,
            max_frame_seq_length: int = 80,
            return_every_nth_element: int = 1,
            batch_size_exponential_of_2: bool = True,
            pin_memory=False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

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

        self.dataset = datasets.load_dataset(
            dataset_source,
            name=dataset_variant,
            trust_remote_code=True,
            streaming=True,
        )

        self.tokenizer = tokenizer

        self.batch_size_per_device = batch_size

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """

        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

    def _map_dataset(self, batch):
        labels = None

        if self.tokenizer is not None:
            transcription = [ex['transcription'] for ex in batch]
            labels = self.tokenizer(
                transcription,
                padding=self.batch_size_per_device > 1,
                return_tensors='pt',
                return_length=True,
                return_attention_mask=False,
            )

        # expecting pixel_values to be of shape (batch_size, num_frames, 3, 224, 224)
        pixel_values = [
            # enforce max_frame_seq_length by truncating the frames
            # and only return every nth element
            np.asarray(
                video['frames'] if len(video['frames']) <= self.hparams.max_frame_seq_length else video['frames'][
                                                                                                  :self.hparams.max_frame_seq_length],
                dtype=np.float32
            )[::self.hparams.return_every_nth_element]
            for video in batch
        ]
        video_lengths = torch.IntTensor(torch.tensor([video.shape[0] for video in pixel_values], dtype=torch.int))

        pixel_values = [
            self._image_processor(images=video, return_tensors='np')['pixel_values']
            for video in pixel_values
        ]

        batch_size = len(pixel_values)

        if self.hparams.batch_size_exponential_of_2:
            # Ensure that the batch size is always in 2^n
            # as this is some weird batching restriction of VideoMAE
            batch_size = next_lower_power_of_2(len(pixel_values))

        # Create a padded array of zeros
        # and the original pixel_values into the padded array
        channels, height, width = pixel_values[0][0].shape
        padded_pixel_values = np.zeros(
            (batch_size, self.hparams.max_frame_seq_length, channels, height, width))
        attention_mask = torch.zeros((batch_size, self.hparams.max_frame_seq_length), dtype=torch.bool)

        for i, video in enumerate(pixel_values):
            if i >= batch_size:
                # don't copy more than we have allocated
                break

            attention_mask[i, :len(video)] = 1
            padded_pixel_values[i, :len(video)] = video

        # Convert to tensor
        padded_pixel_values = torch.tensor(padded_pixel_values, dtype=torch.float32)

        result = {
            'pixel_values': padded_pixel_values,
            'video_lengths': video_lengths,
            'attention_mask': attention_mask,
            'ids': [ex['id'] for ex in batch],
        }

        if labels is not None:
            result['labels'] = labels.input_ids,

        return result

    def _instantiate_data_loader(self, dataset: Union[IterableDataset, Dataset]) -> DataLoader:
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            collate_fn=self._map_dataset,
            pin_memory=self.hparams.pin_memory,
        )

        return data_loader

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        subset = self.dataset[datasets.Split.TRAIN]

        return self._instantiate_data_loader(subset)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        subset = self.dataset[datasets.Split.VALIDATION]

        return self._instantiate_data_loader(subset)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        subset = self.dataset[datasets.Split.TEST]

        return self._instantiate_data_loader(subset)

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = BundestagSLRVideoMAEDataModule()
