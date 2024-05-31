from typing import Dict, Any

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric
from transformers import VideoMAEForPreTraining, VideoMAEImageProcessor

# Faster training on Ampere cards...
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision('high')


def prepare_data_for_preprocess(pixel_values: torch.Tensor):
    batches = [
        [
            img
            for img in video
        ]
        for video in pixel_values
    ]

    return batches


def calculate_num_patches(height, width, patch_size):
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    return num_patches_height * num_patches_width


class VideoMAEPretrainingModule(LightningModule):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler
    ):
        super().__init__()

        self.model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")
        self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, pixel_values, bool_masked_pos=None):
        return self.model(pixel_values, bool_masked_pos=bool_masked_pos)

    def mask_and_forward(self, pixel_values):
        # expecting pixel_values to be of shape (batch_size, num_frames, 3, 224, 224)
        pixel_values = prepare_data_for_preprocess(pixel_values)
        pixel_values = self.processor(
            pixel_values,
            input_data_format='channels_first',
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
            do_rescale=False
        ).pixel_values
        pixel_values = pixel_values.to(self.device)

        batch_size, num_frames, channels, height, width = pixel_values.shape
        print(f"pixel_values.shape: {pixel_values.shape}")

        # patch logic can be found in example https://huggingface.co/docs/transformers/main/en/model_doc/videomae#transformers.VideoMAEForPreTraining.forward.example
        num_patches_per_frame = calculate_num_patches(height, width, self.model.config.patch_size)
        seq_length = (num_frames // self.model.config.tubelet_size) * num_patches_per_frame
        bool_masked_pos = torch.randint(0, 2, (batch_size, seq_length)).bool()

        outputs = self.forward(pixel_values, bool_masked_pos=bool_masked_pos)

        return outputs, bool_masked_pos

    def training_step(self, batch, batch_idx):
        pixel_values = batch['input_values']
        outputs, mask = self.mask_and_forward(pixel_values)

        self.log("train/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)

        return outputs.loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch['input_values']
        outputs, mask = self.mask_and_forward(pixel_values)

        self.log("val/loss", outputs.loss, on_step=False, on_epoch=True, prog_bar=True)

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
