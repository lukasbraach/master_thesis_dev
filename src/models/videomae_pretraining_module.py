import math
from typing import Optional

import torch
from lightning import LightningModule
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics import MeanMetric

from src.models.components.videomae_with_decoder import CustomVideoMAEForPreTraining

# Faster training on Ampere cards...
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


class VideoMAEPretrainingModule(LightningModule):
    def __init__(
            self,
            # default parameters for restoring.
            net: CustomVideoMAEForPreTraining = None,
            optimizer: torch.optim.Optimizer = None,
    ):
        super().__init__()

        self.net = net

        self.optimizer = optimizer
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(
            self,
            pixel_values,
            video_lengths: Optional[torch.IntTensor] = None,
            **kwargs
    ):
        return self.net(pixel_values, video_lengths=video_lengths, **kwargs)

    def mask_and_forward(
            self,
            pixel_values,
            video_lengths: Optional[torch.IntTensor] = None
    ):
        bool_masked_pos = self._create_mask_for(pixel_values, video_lengths)

        return self.net(pixel_values, bool_masked_pos=bool_masked_pos, video_lengths=video_lengths)

    def training_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        video_lengths = batch['video_lengths']

        outputs = self.mask_and_forward(pixel_values, video_lengths=video_lengths)

        self.log("train/loss", outputs.loss, batch_size=len(pixel_values), on_step=True, on_epoch=True, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        video_lengths = batch['video_lengths']

        outputs = self.mask_and_forward(pixel_values, video_lengths=video_lengths)

        self.log("val/loss", outputs.loss, batch_size=len(pixel_values), on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.optimizer(params=self.trainer.model.parameters())

        def lr_lambda(epoch, warmup_epochs=10, decay_rate=0.95):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                return decay_rate ** (epoch - warmup_epochs)

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 2500,
            },
        }

    def get_pretrained_model(self):
        return self.net.videomae

    def _get_seq_length_for(self, pixel_values: torch.Tensor) -> int:
        batch_size, num_frames, channels, height, width = pixel_values.shape

        num_patches_per_frame = (height // self.net.config.patch_size) ** 2
        seq_length = (num_frames // self.net.config.tubelet_size) * num_patches_per_frame

        return seq_length

    def _create_mask_for(self, pixel_values: torch.Tensor, video_frame_lengths: torch.IntTensor) -> torch.Tensor:
        '''
        Create mask for the pixel_values
        Args:
            pixel_values: (batch_size, seq_length, 3, 224, 224)

        Returns: mask tensor of shape (batch_size, seq_length)
        '''

        # patch logic can be found in example:
        # https://huggingface.co/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForPreTraining.forward.example

        batch_size, num_frames, channels, height, width = pixel_values.shape
        seq_length = self._get_seq_length_for(pixel_values)

        mean_video_frame_lengths = torch.mean(video_frame_lengths.float())
        min_video_frame_lengths = torch.min(video_frame_lengths)

        # At least the equivalent of 2 frames must be masked
        # for the shortest video in the batch. This is a safeguard
        # against edge cases with wildly varying video lengths.
        min_mask_frames = num_frames - int(min_video_frame_lengths) + 2
        min_mask_patches = self._get_seq_length_for(pixel_values[:, :min_mask_frames, :, :, :])

        # mean relative amount of underlap – meaning that the video
        # frames are not fully filling the num_frames of the batch.
        mean_relative_video_underlap = 1 - float(mean_video_frame_lengths) / num_frames

        # The amount of masked tokens must be the same for all sequences in the batch.
        # See: https://discuss.huggingface.co/t/videomae-pretrain-batch-masking/22176/7
        mask_num = max(math.ceil(seq_length * (0.3 + mean_relative_video_underlap)), min_mask_patches)

        mask = torch.zeros((batch_size, seq_length)).bool()
        for i in range(batch_size):
            video_underlap_frames = num_frames - int(video_frame_lengths[i])

            # distribute the available masking frames so that the underlapping
            # video frames are always fully masked
            perm = torch.randperm(int(video_frame_lengths[i]))[:mask_num - video_underlap_frames]
            mask[i][perm] = True
            mask[i][-video_underlap_frames:] = True

        return mask


if __name__ == "__main__":
    _ = VideoMAEPretrainingModule(None, None)
