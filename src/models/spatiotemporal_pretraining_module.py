import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import ExponentialLR, ChainedScheduler, LambdaLR
from torchmetrics import MeanMetric
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices, _sample_negative_indices

from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoderForPreTraining

# Faster training on Ampere cards...
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class SpatiotemporalPretrainingModule(LightningModule):
    def __init__(
            self,
            net: SpatiotemporalEncoderForPreTraining,
            optimizer: torch.optim.Optimizer
    ):
        super().__init__()

        self.net = net

        self.optimizer = optimizer
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, pixel_values, attention_mask=None):
        model_response = self.net(
            pixel_values,
            attention_mask=attention_mask
        )

        return model_response.hidden_states

    def mask_and_forward(self, pixel_values: torch.Tensor, attention_mask=None):
        # expecting pixel_values to be of shape (batch_size, sequence_length, num_channels, height, width)
        batch_size, raw_sequence_length, _, _, _ = pixel_values.shape

        # we basically know that the sequence length is going to be the same as the number of frames,
        # but for good measure we'll "calculate" it anyway, using the model's method
        sequence_length = self.net._get_feat_extract_output_lengths(raw_sequence_length)

        mask_time_indices = _compute_mask_indices(
            shape=(batch_size, sequence_length), mask_prob=0.3, mask_length=2
        )

        sampled_negative_indices = _sample_negative_indices(
            features_shape=(batch_size, sequence_length),
            num_negatives=self.net.config.num_negatives,
            mask_time_indices=mask_time_indices,
        )

        mask_time_indices = torch.tensor(data=mask_time_indices, device=pixel_values.device, dtype=torch.int)
        sampled_negative_indices = torch.tensor(
            data=sampled_negative_indices, device=pixel_values.device, dtype=torch.int
        )

        loss = self.net(
            pixel_values,
            attention_mask=attention_mask,
            mask_time_indices=mask_time_indices,
            sampled_negative_indices=sampled_negative_indices
        ).loss

        return loss

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        attention_mask = batch['attention_mask']

        loss = self.mask_and_forward(pixel_values, attention_mask=attention_mask)

        self.log("train/loss", loss, batch_size=len(pixel_values), on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        attention_mask = batch['attention_mask']

        loss = self.mask_and_forward(pixel_values, attention_mask=attention_mask)

        self.log("val/loss", loss, batch_size=len(pixel_values), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())

        def lr_lambda(epoch, warmup_epochs=10, decay_rate=0.975):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return decay_rate ** (epoch - warmup_epochs)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        scheduler = ChainedScheduler([scheduler])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 2500,
            },
        }


if __name__ == "__main__":
    _ = SpatiotemporalPretrainingModule(None, None, None, None)
