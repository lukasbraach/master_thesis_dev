import torch
from PIL import Image
from torch import nn


class SpatiotemporalEncoder(nn.Module):

    def __init__(
            self,
            spatial_encoder: torch.nn.Module,
            temporal_encoder: torch.nn.Module,
            pos_encoder: torch.nn.Module,
    ) -> None:
        super().__init__()

        self.spatial_encoder = spatial_encoder
        self.pos_encoder = pos_encoder
        self.temporal_encoder = temporal_encoder

    def forward(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            X = self.spatial_encoder(**kwargs)

        X = self.pos_encoder(X)
        X = self.temporal_encoder(X)

        return X

    def preprocess_image(self, x: Image) -> torch.Tensor:
        return self.temporal_encoder.preprocess_image(x)


if __name__ == "__main__":
    _ = SpatiotemporalEncoder()
