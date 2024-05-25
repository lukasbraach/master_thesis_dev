import math

from torch import nn

from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoderConfig


class SpatiotemporalDecoder(nn.Module):
    def __init__(self, config: SpatiotemporalEncoderConfig):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size // 2,
            nhead=config.num_attention_heads,
            dim_feedforward=config.intermediate_size // 2,
            dropout=config.dropout,
            activation=config.hidden_act
        )

        self.config = config

        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=int(math.ceil(config.num_hidden_layers / 3)))
        self.fc_out = nn.Linear(config.hidden_size,
                                config.image_size[0] * config.image_size[1] * config.image_size[2])  # Assuming image_size is (H, W, C)

    def forward(self, encoded_features, memory, mask=None):
        decoded_features = self.decoder(encoded_features, memory, tgt_mask=mask)
        reconstructed_frames = self.fc_out(decoded_features)
        batch_size, num_frames, _ = reconstructed_frames.shape
        reconstructed_frames = reconstructed_frames.view(batch_size, num_frames, *self.config.image_size)
        return reconstructed_frames
