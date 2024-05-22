import torch
from lightning import LightningModule
from torch import nn
from torchmetrics import MeanMetric

from src.models.components.spatiotemporal_decoder import SpatiotemporalDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder


def mask_latents(latents, mask_ratio=0.3):
    """
    Randomly mask latent representations with the specified mask ratio.
    Args:
        latents (torch.Tensor): Latent representations of shape (batch_size, num_latents, hidden_size).
        mask_ratio (float): Proportion of latents to mask.

    Returns:
        masked_latents (torch.Tensor): Latent representations with some masked out.
        mask (torch.Tensor): Boolean tensor indicating which latents were masked.
    """
    batch_size, num_latents, hidden_size = latents.shape
    mask = torch.rand(batch_size, num_latents) < mask_ratio
    masked_latents = latents.clone()
    masked_latents[mask] = 0  # Mask out the latents
    
    return masked_latents, mask


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
        encoded_features = self.encoder(pixel_values)
        masked_encoded_features, mask = mask_latents(encoded_features)
        reconstructed_latents = self.decoder(masked_encoded_features, encoded_features, mask)
        return reconstructed_latents, mask

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        reconstructed_latents, mask = self.mask_and_forward(pixel_values)
        loss = self.compute_loss(reconstructed_latents, pixel_values, mask)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        reconstructed_latents, mask = self.mask_and_forward(pixel_values)
        loss = self.compute_loss(reconstructed_latents, pixel_values, mask)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def compute_loss(self, reconstructed_latents, original_latents, mask):
        mse_loss = nn.MSELoss()
        return mse_loss(reconstructed_latents[mask], original_latents[mask])

    def configure_optimizers(self):
        optimizer = self.optimizer(params=self.trainer.model.parameters())
        scheduler = self.scheduler(optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    _ = SignLanguageLitModule(None, None, None, None)
