import torch
from torch import nn


class TemporalEncoder(nn.Module):

    def __init__(
            self,
            d_model: int = 768,
            nhead: int = 8,
            num_layers: int = 6,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)

        # TODO add `norm = nn.BatchNorm1d`..?
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer_encoder(x)


if __name__ == "__main__":
    _ = TemporalEncoder()
