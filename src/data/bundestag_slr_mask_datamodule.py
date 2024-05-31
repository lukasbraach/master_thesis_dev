import math
from typing import Any, Dict, Optional, Union

import datasets
import torch
from datasets import IterableDataset, Dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor


def prepare_data_for_preprocess(pixel_values: torch.Tensor | list[torch.Tensor]):
    batches = [
        [
            img
            for img in video
        ]
        for video in pixel_values
    ]

    return batches


def create_mask_for(pixel_values: torch.Tensor) -> torch.Tensor:
    '''
    Create mask for the pixel_values
    Args:
        pixel_values: (batch_size, seq_length, 3, 224, 224)

    Returns: mask tensor of shape (batch_size, seq_length)
    '''

    # patch logic can be found in example:
    # https://huggingface.co/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForPreTraining.forward.example

    batch_size, num_frames, channels, height, width = pixel_values.shape

    model_path_size = 16
    model_tubelet_size = 2

    num_patches_per_frame = (height // model_path_size) ** 2
    seq_length = (num_frames // model_tubelet_size) * num_patches_per_frame

    # The amount of masked tokens is set to 30% of the sequence length
    # and must be the same for all sequences in the batch.
    # See: https://discuss.huggingface.co/t/videomae-pretrain-batch-masking/22176/7
    mask_num = math.ceil(seq_length * 0.3)

    mask = torch.zeros((batch_size, seq_length)).bool()
    for i in range(batch_size):
        perm = torch.randperm(seq_length)[:mask_num]
        mask[i][perm] = True

    return mask


class BundestagSLRVideoMAEDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_source='lukasbraach/bundestag_slr',
            batch_size: int = 16,
            num_workers: int = 16,
            max_frame_seq_length: int = None,
            pin_memory=False,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # initialized in setup function.
        self.pre_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        self.dataset = datasets.load_dataset(
            dataset_source,
            trust_remote_code=True,
            streaming=True,
        )

        self.batch_size_per_device = batch_size
        self.max_frame_seq_length = max_frame_seq_length

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
        # expecting pixel_values to be of shape (batch_size, num_frames, 3, 224, 224)
        pixel_values = prepare_data_for_preprocess([ex['frames'] for ex in batch])

        pixel_values = self.pre_processor(
            pixel_values,
            return_tensors='pt',
            do_resize=False,
            do_center_crop=False,
            do_rescale=False
        ).pixel_values

        pixel_values = pixel_values[:, :self.max_frame_seq_length, :, :, :]
        mask = create_mask_for(pixel_values)

        result = {
            'pixel_values': pixel_values,
            'attention_mask': mask,
            'ids': [ex['id'] for ex in batch],
        }

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
    _ = BundestagSLRDataModule()
