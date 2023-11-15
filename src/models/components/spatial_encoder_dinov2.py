import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, Dinov2Model


class SpatialEncoderDINOv2(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
            self,
    ) -> None:
        """Initialize a `DINOv2` module."""
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")

    def forward(self, **kwargs) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input image.
        :return: A tensor of predictions.
        """

        with torch.no_grad():
            outputs = self.model(**kwargs)

        return outputs.last_hidden_state

    def preprocess_image(self, x: Image) -> torch.Tensor:
        inputs = self.image_processor(x, return_tensors="pt")

        return inputs


if __name__ == "__main__":
    _ = SpatialEncoderDINOv2()
