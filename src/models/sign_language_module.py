from typing import Any, Dict, Tuple, Optional

import torch
import wandb
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.text import WordErrorRate
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.models.components.sign_language_net import SignLanguageNet


class SignLanguageLitModule(LightningModule):
    def __init__(
            self,
            net: SignLanguageNet,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
    ) -> None:
        """Initialize a `SignLanguageLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=['net'], logger=False)

        self.net = net

        # metric objects for calculating and averaging accuracy across batches
        self.train_wer = WordErrorRate()
        self.val_wer = WordErrorRate()
        self.test_wer = WordErrorRate()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_wer_best = MinMetric()

    def forward(
            self,
            input_values: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
    ) -> Seq2SeqLMOutput:
        """
        Perform a forward pass through the model `self.net`.

        :param input_values: A tensor of images.
        :return: A tensor of logits.
        """
        outputs = self.net(input_values=input_values, attention_mask=attention_mask, labels=labels)
        return outputs

    def on_train_start(self) -> None:
        """
        Lightning hook that is called when training begins.
        """
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_wer.reset()
        self.val_wer_best.reset()

        wandb.watch(self.net, log="all")

    def model_step(
            self, batch: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        output = self.forward(**batch)
        preds = torch.argmax(output.logits, dim=2)

        truth_decoded = self.net.tokenizer.batch_decode(batch["labels"])
        preds_decoded = self.net.tokenizer.batch_decode(preds)

        return output.loss, preds_decoded, truth_decoded

    def training_step(
            self, batch: dict, batch_idx: int
    ) -> torch.Tensor:
        """
        Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """

        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_wer(preds, targets)

        self.log("train/batch_idx", batch_idx, on_step=True, prog_bar=True)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/wer", self.train_wer, on_step=False, on_epoch=True, prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(
                key="train/samples",
                columns=["prediction", "targets"],
                data=zip(preds, targets)
            )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """
        Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_wer(preds, targets)

        self.log("val/batch_idx", batch_idx, on_step=True, prog_bar=True)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/wer", self.val_wer, on_step=False, on_epoch=True, prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(
                key="val/samples",
                columns=["prediction", "targets"],
                data=zip(preds, targets)
            )

    def on_validation_epoch_end(self) -> None:
        """
        Lightning hook that is called when a validation epoch ends.
        :return:
        """
        wer = self.val_wer.compute()  # get current val acc
        self.val_wer_best(wer)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/wer_best", self.val_wer_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """
        Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_wer(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/wer", self.test_wer, on_step=False, on_epoch=True, prog_bar=True)

        if isinstance(self.logger, WandbLogger):
            self.logger.log_text(
                key="test/samples",
                columns=["prediction", "targets"],
                data=zip(preds, targets)
            )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def prepare_data(self) -> None:
        """
        Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
        settings) will result in corrupted data. Lightning ensures this method is called only within a single
        process, so you can safely add your downloading logic within.

        .. warning:: DO NOT set state to the model (use ``setup`` instead)
            since this is NOT called on every device
        """

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SignLanguageLitModule(None, None, None)
