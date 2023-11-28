import torch
from PIL import Image
from d2l.torch import d2l
from torch import nn
from transformers import Dinov2Model, PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel


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

    def __init__(self, config: SpatiotemporalEncoderConfig = SpatiotemporalEncoderConfig()) -> None:
        super().__init__(config=config)

        self.spatial_encoder = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.pos_encoder = d2l.PositionalEncoding(num_hiddens=config.hidden_size, dropout=config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads,
                                                   dropout=config.dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_hidden_layers)

    def _init_weights(self, module):
        if isinstance(module, (Dinov2Model)):
            module.init_weights()

    def forward(self, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            X = self.spatial_encoder(**kwargs).pooler_output

        X = self.pos_encoder(X)
        X = self.temporal_encoder(X)

        return X

    def preprocess_image(self, x: Image) -> torch.Tensor:
        return self.spatial_encoder.preprocess_image(x)


if __name__ == "__main__":
    _ = SpatiotemporalEncoder()

AutoConfig.register("spatiotemporal-encoder", SpatiotemporalEncoderConfig)
AutoModel.register(SpatiotemporalEncoderConfig, SpatiotemporalEncoder)
