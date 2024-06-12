from typing import Optional, Tuple, Union

import torch
from transformers import VideoMAEModel, AutoModel, VideoMAEConfig
from transformers.modeling_outputs import BaseModelOutput


class CustomVideoMAEModel(VideoMAEModel):
    def __init__(self, config: VideoMAEConfig):
        super().__init__(config)

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        return super().forward(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def _get_feature_vector_attention_mask(
            self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        """
        Just a hack to make SpeechEncoderDecoderModel work with VideoMAEModel
        """
        return attention_mask


AutoModel.register(VideoMAEConfig, CustomVideoMAEModel)
