from typing import Any, Dict, Optional, Union

import datasets
from datasets import IterableDataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, PreTrainedTokenizer

from src.models.components.feature_extractor_dinov2 import SignLanguageFeatureExtractor


class RWTHPhoenix2014DataModule(LightningDataModule):
    def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            batch_size: int = 1,
            num_workers: int = 24,
            streaming=True,
            pin_memory=False,
            variant='multisigner',
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # initialized in setup function.
        self.pre_processor = SignLanguageFeatureExtractor()
        self.tokenizer = tokenizer
        self.dataset = datasets.load_dataset(
            'lukasbraach/rwth_phoenix_weather_2014',
            variant,
            streaming=True,
            trust_remote_code=True,
        )

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
        transcription = batch['transcription']

        labels = self.tokenizer(
            transcription,
            padding=self.batch_size_per_device > 1,
            return_tensors='pt',
            return_length=True,
        )

        if self.hparams.variant == 'pre-training':
            return {
                'labels': labels.input_ids,
            }

        feature = self.pre_processor(
            batch['frames'],
            sampling_rate=25,
            padding=self.batch_size_per_device > 1,
            return_attention_mask=True,
            return_tensors='pt'
        )

        result = {
            'input_values': feature.input_values,
            'attention_mask': feature.attention_mask,
            'labels': labels.input_ids,
        }

        return result

    def _instantiate_data_loader(self, dataset: IterableDataset) -> DataLoader:
        dataset = dataset.map(
            function=self._map_dataset,
            batched=True,
            batch_size=self.batch_size_per_device,
            remove_columns=['frames'] if self.hparams.variant != 'pre-training' else None,
        )

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            prefetch_factor=3,
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
    _ = RWTHPhoenix2014DataModule()
