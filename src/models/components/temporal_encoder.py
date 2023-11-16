import torch
from torch import nn


class TemporalEncoder(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # TODO increase embedding size for decoder?
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)

        # TODO add `norm = nn.BatchNorm1d`..?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)


if __name__ == "__main__":
    _ = TemporalEncoder()
