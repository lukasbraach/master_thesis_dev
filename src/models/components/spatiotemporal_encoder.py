import math

import torch
from torch import nn
from transformers import Dinov2Model, PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel, BatchFeature


class SpatiotemporalEncoderConfig(PretrainedConfig):
    model_type = "spatiotemporal-encoder"

    def __init__(self,
                 hidden_size: int = 768,
                 dropout: float = 0.1,
                 num_attention_heads: int = 8,
                 num_hidden_layers=6,
                 **kwargs
                 ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            **kwargs
        )


class SpatiotemporalEncoder(PreTrainedModel):
    config_class = SpatiotemporalEncoderConfig
    main_input_name = "x"

    def __init__(self, config: SpatiotemporalEncoderConfig = SpatiotemporalEncoderConfig()) -> None:
        super().__init__(config=config)

        self.pos_encoder = PositionalEncoding(d_model=config.hidden_size, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads,
                                                   dropout=config.dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

    def _init_weights(self, module):
        if isinstance(module, (Dinov2Model)):
            module.init_weights()

    def forward(self, batch_feature: BatchFeature, **kwargs) -> torch.Tensor:
        """
        Arguments:
            batch_feature: transformers.BatchFeature
        :return torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        batch_feature = batch_feature.convert_to_tensors('pt')

        # swap batch and sequence dimensions
        x = torch.transpose(batch_feature["input_values"], 0, 1)
        mask = torch.transpose(x["attention_mask"], 0, 1) if hasattr(x, "attention_mask") else None

        x = self.pos_encoder(x)
        x = self.temporal_encoder(x, mask)

        return x


class PositionalEncoding(nn.Module):
    # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html#define-the-model

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


if __name__ == "__main__":
    _ = SpatiotemporalEncoder()

AutoConfig.register("spatiotemporal-encoder", SpatiotemporalEncoderConfig)
AutoModel.register(SpatiotemporalEncoderConfig, SpatiotemporalEncoder)
