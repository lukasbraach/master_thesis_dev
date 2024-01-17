import torch
from torch import nn
from transformers import AutoConfig, AutoModel, Wav2Vec2Config, Dinov2Model
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


class SpatiotemporalEncoderConfig(Wav2Vec2Config):
    model_type = "spatiotemporal-encoder"

    def __init__(self,
                 hidden_size: int = 768,
                 intermediate_size: int = 3072,
                 dropout: float = 0.1,
                 num_attention_heads: int = 8,
                 num_hidden_layers: int = 12,
                 mask_time_length: int = 1,
                 hidden_act="gelu",
                 freeze_feature_extractor=True,
                 **kwargs
                 ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout=dropout,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            mask_time_length=mask_time_length,
            hidden_act=hidden_act,
            **kwargs
        )

        self.freeze_feature_extractor = freeze_feature_extractor


class SpatialFeatureEncoder(nn.Module):
    """Construct the features from raw video frames"""

    def __init__(self, config: SpatiotemporalEncoderConfig):
        super().__init__()

        self._spatial_encoder = Dinov2Model.from_pretrained(
            "facebook/dinov2-base",
            torch_dtype="auto",
        )

        self.frozen = config.freeze_feature_extractor

    def forward(self, pixel_values):
        with torch.no_grad() if self.frozen else torch.enable_grad():
            batch_size = pixel_values.shape[0]
            seq_length = pixel_values.shape[1]

            x = pixel_values

            # concatenate batch together
            x = torch.reshape(x, (-1,) + x.shape[2:])
            x = self._spatial_encoder(x)['pooler_output']

            # expand batches
            x = torch.reshape(x, (batch_size, seq_length) + x.shape[1:])

        return x.transpose(1, 2)


class SpatiotemporalFeatureProjection(nn.Module):
    def __init__(self, config: SpatiotemporalEncoderConfig):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # non-projected hidden states are needed for quantization
        norm_hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(norm_hidden_states)
        return hidden_states, norm_hidden_states


class SpatiotemporalEncoder(Wav2Vec2Model):
    config_class = SpatiotemporalEncoderConfig
    base_model_prefix = "spatiotemporal-encoder"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: SpatiotemporalEncoderConfig):
        super().__init__(config)

        # we have to overwrite these parts for our purposes
        self.feature_extractor = SpatialFeatureEncoder(config)
        self.feature_projection = SpatiotemporalFeatureProjection(config)

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == "__main__":
    _ = SpatiotemporalEncoder(SpatiotemporalEncoderConfig())

AutoConfig.register("spatiotemporal-encoder", SpatiotemporalEncoderConfig)
AutoModel.register(SpatiotemporalEncoderConfig, SpatiotemporalEncoder)
