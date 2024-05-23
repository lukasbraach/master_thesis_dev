from typing import Dict, Any

import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from src.models.components.spatiotemporal_decoder import SpatiotemporalDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder


class SpatiotemporalPretrainingModule(LightningModule):
    def __init__(
            self,
            encoder: SpatiotemporalEncoder,
            decoder: SpatiotemporalDecoder,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, pixel_values):
        return self.encoder(pixel_values)

    def mask_and_forward(self, pixel_values):
        masked_pixel_values, mask = mask_frames(pixel_values)
        encoded_features = self.encoder(masked_pixel_values)
        memory = encoded_features  # Memory in this context is the output of the encoder
        reconstructed_frames = self.decoder(encoded_features, memory, mask)
        return reconstructed_frames, mask

    def training_step(self, batch, batch_idx):
        pixel_values = batch['input_values']
        reconstructed_frames, mask = self.mask_and_forward(pixel_values)
        loss = self.compute_loss(reconstructed_frames, pixel_values, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['input_values']
        reconstructed_frames, mask = self.mask_and_forward(pixel_values)
        loss = self.compute_loss(reconstructed_frames, pixel_values, mask)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def compute_loss(self, reconstructed_frames, original_frames, mask):
        mse_loss = nn.MSELoss()
        return mse_loss(reconstructed_frames[mask], original_frames[mask])

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


def mask_frames(frames, mask_ratio=0.3):
    """
    Randomly mask frames with the specified mask ratio.
    Args:
        frames (torch.Tensor): Input video frames of shape (batch_size, num_frames, height, width, channels).
        mask_ratio (float): Proportion of frames to mask.

    Returns:
        masked_frames (torch.Tensor): Frames with some masked out.
        mask (torch.Tensor): Boolean tensor indicating which frames were masked.
    """
    batch_size, num_frames, height, width, channels = frames.shape

    mask = torch.rand(batch_size, num_frames) < mask_ratio
    masked_frames = frames.clone()
    masked_frames[mask] = 0  # Mask out the frames

    return masked_frames, mask


if __name__ == "__main__":
    _ = SpatiotemporalPretrainingModule(None, None, None, None)
