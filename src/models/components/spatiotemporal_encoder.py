import torch
from torch import nn
from transformers import Dinov2Model, PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel, BatchFeature

from src.models.components.positional_encoding import PositionalEncoding


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

    def forward(self, x: BatchFeature, **kwargs) -> BatchFeature:
        x["input_values"] = self.pos_encoder(x["input_values"])
        x["attention_mask"] = x["attention_mask"] if hasattr(x, "attention_mask") else None

        x["input_values"] = torch.vmap(
            self.temporal_encoder,
            randomness='different',
            in_dims=(0, None if x["attention_mask"] is None else 0)
        )(x["input_values"], x["attention_mask"])

        return x


if __name__ == "__main__":
    _ = SpatiotemporalEncoder()

AutoConfig.register("spatiotemporal-encoder", SpatiotemporalEncoderConfig)
AutoModel.register(SpatiotemporalEncoderConfig, SpatiotemporalEncoder)
