import torch
from PIL import Image
from d2l.torch import d2l
from torch import nn

from src.models.components.spatial_encoder_dinov2 import SpatialEncoderDINOv2
from src.models.components.temporal_encoder import TemporalEncoder


class SpatiotemporalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.spatial_encoder = SpatialEncoderDINOv2()
        self.pos_encoder = d2l.PositionalEncoding()
        self.temporal_encoder = TemporalEncoder()

    def forward(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            X = self.spatial_encoder(**kwargs)

        X = self.pos_encoder(X)
        X = self.temporal_encoder(X)

        return X

    def preprocess_image(self, x: Image) -> torch.Tensor:
        return self.spatial_encoder.preprocess_image(x)


if __name__ == "__main__":
    _ = SpatiotemporalEncoder()
