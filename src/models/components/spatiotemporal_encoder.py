import math
from typing import Optional, Union, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, Wav2Vec2Config
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model


class SpatiotemporalEncoderConfig(Wav2Vec2Config):
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


class SpatiotemporalEncoder(Wav2Vec2Model):

    config_class = SpatiotemporalEncoderConfig
    base_model_prefix = "spatiotemporal-encoder"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True

    def __init__(self, config: SpatiotemporalEncoderConfig):
        super().__init__(config)

    def forward(
        self,
        input_values: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.LongTensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        """

        :param input_values: torch.Tensor [batch_size, sequence_length, embed_dim]
        :param attention_mask:
        :param mask_time_indices:
        :param output_attentions:
        :param output_hidden_states:
        :param return_dict:
        :return:
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                input_values.shape[1], attention_mask, add_adapter=False
            )

        hidden_states = self._mask_hidden_states(
            input_values, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, input_values) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=input_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


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
