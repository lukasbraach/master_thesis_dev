from typing import Dict, Any, Optional

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.videomae_with_decoder import CustomVideoMAEForPreTraining

# Faster training on Ampere cards...
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class VideoMAEPretrainingModule(LightningModule):
    def __init__(
            self,
            net: CustomVideoMAEForPreTraining,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
    ):
        super().__init__()

        self.net = net

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(
            self,
            pixel_values,
            bool_masked_pos: torch.BoolTensor = None,
            video_lengths: Optional[torch.IntTensor] = None
    ):
        return self.net(pixel_values, bool_masked_pos=bool_masked_pos)

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        bool_masked_pos = batch['attention_mask']
        video_lengths = batch['video_lengths']

        outputs = self.forward(pixel_values, bool_masked_pos=bool_masked_pos, video_lengths=video_lengths)

        self.log("train/loss", outputs.loss, batch_size=len(batch), on_step=True, on_epoch=True, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        bool_masked_pos = batch['attention_mask']
        video_lengths = batch['video_lengths']

        outputs = self.forward(pixel_values, bool_masked_pos=bool_masked_pos, video_lengths=video_lengths)

        self.log("val/loss", outputs.loss, batch_size=len(batch), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        if self.scheduler is not None:
            scheduler = self.scheduler(optimizer=optimizer)
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
    _ = VideoMAEPretrainingModule(None, None)
