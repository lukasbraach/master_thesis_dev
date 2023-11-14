import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor, Dinov2Model


class DINOv2(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
    ) -> None:
        """Initialize a `DINOv2` module."""
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        self.model = Dinov2Model.from_pretrained("facebook/dinov2-base")

    def forward(self, x: Image) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input image.
        :return: A tensor of predictions.
        """
        inputs = self.image_processor(x, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.last_hidden_state


if __name__ == "__main__":
    _ = DINOv2()
