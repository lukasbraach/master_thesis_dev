from typing import Optional, Union

import torch
from torch import nn
from torch.nn import MSELoss
from transformers import VideoMAEConfig
from transformers.models.videomae.modeling_videomae import VideoMAEForPreTrainingOutput, VideoMAEPreTrainedModel, \
    VideoMAEModel, get_sinusoid_encoding_table, VideoMAEDecoder
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class CustomVideoMAEForPreTraining(VideoMAEPreTrainedModel):
    def __init__(self, config: VideoMAEConfig):
        super().__init__(config)
        self.config = config

        self.videomae = VideoMAEModel(config)

        self.encoder_to_decoder = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.position_embeddings = get_sinusoid_encoding_table(
            self.videomae.embeddings.num_patches, config.decoder_hidden_size
        )

        self.decoder = VideoMAEDecoder(config, num_patches=self.videomae.embeddings.num_patches)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            bool_masked_pos: torch.BoolTensor,
            video_lengths: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, VideoMAEForPreTrainingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, sequence_length)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0). Each video in the
            batch must have the same number of masked patches. Sequence length is `(num_frames // tubelet_size) *
            (image_size // patch_size) ** 2`.

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, VideoMAEForPreTraining
        >>> import numpy as np
        >>> import torch

        >>> num_frames = 16
        >>> video = list(np.random.randint(0, 256, (num_frames, 3, 224, 224)))

        >>> image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        >>> model = VideoMAEForPreTraining.from_pretrained("MCG-NJU/videomae-base")

        >>> pixel_values = image_processor(video, return_tensors="pt").pixel_values

        >>> num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
        >>> seq_length = (num_frames // model.config.tubelet_size) * num_patches_per_frame
        >>> bool_masked_pos = torch.randint(0, 2, (1, seq_length)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss = outputs.loss
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.videomae(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.encoder_to_decoder(
            sequence_output
        )  # [batch_size, num_visible_patches, decoder_hidden_size]
        batch_size, seq_len, num_channels = sequence_output.shape

        # we don't unshuffle the correct visible token order, but shuffle the position embeddings accordingly.
        if bool_masked_pos is None:
            raise ValueError("One must provided a boolean mask ")
        expanded_position_embeddings = self.position_embeddings.expand(batch_size, -1, -1).type_as(pixel_values)
        expanded_position_embeddings = expanded_position_embeddings.to(pixel_values.device).clone().detach()
        pos_emb_visible = expanded_position_embeddings[~bool_masked_pos].reshape(batch_size, -1, num_channels)
        pos_emb_mask = expanded_position_embeddings[bool_masked_pos].reshape(batch_size, -1, num_channels)

        # [batch_size, num_patches, decoder_hidden_size]
        x_full = torch.cat([sequence_output + pos_emb_visible, self.mask_token + pos_emb_mask], dim=1)

        # [batch_size, num_masked_patches, num_channels * patch_size * patch_size]
        decoder_outputs = self.decoder(x_full, pos_emb_mask.shape[1])
        logits = decoder_outputs.logits

        loss = None
        with torch.no_grad():
            # calculate the labels to be predicted
            if self.config.num_channels != 3:
                # Can't unnormalize with default means/stds
                frames = pixel_values
            else:
                # first, unnormalize the frames
                device = pixel_values.device
                dtype = pixel_values.dtype
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device=device, dtype=dtype)[None, None, :, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device=device, dtype=dtype)[None, None, :, None, None]
                frames = pixel_values * std + mean  # in [0, 1]

            batch_size, time, num_channels, height, width = frames.shape
            tubelet_size, patch_size = self.config.tubelet_size, self.config.patch_size
            if self.config.norm_pix_loss:
                # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
                frames = frames.view(
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                )
                # step 2: move dimensions to concatenate:
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                # step 3: concatenate:
                frames = frames.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size,
                    num_channels,
                )
                # step 4: normalize. The authors find that the mean is about 0.48 and standard deviation is about 0.08.
                frames_norm = (frames - frames.mean(dim=-2, keepdim=True)) / (
                        frames.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6
                )
                # step 5: reshape to (batch_size, T//ts * H//ps * W//ps, ts * ps * ps * C)
                videos_patch = frames_norm.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                )
            else:
                if self.config.num_channels != 3:
                    raise ValueError(
                        "Can't unnormalize non-RGB images. Consider setting config.norm_pix_loss to False."
                    )
                # step 1: split up dimensions (time by tubelet_size, height by patch_size, width by patch_size)
                frames = frames.view(
                    batch_size,
                    time // tubelet_size,
                    tubelet_size,
                    num_channels,
                    height // patch_size,
                    patch_size,
                    width // patch_size,
                    patch_size,
                )
                # step 2: move dimensions to concatenate: (batch_size, T//ts, H//ps, W//ps, ts, ps, ps, C)
                frames = frames.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
                # step 3: concatenate
                videos_patch = frames.view(
                    batch_size,
                    time // tubelet_size * height // patch_size * width // patch_size,
                    tubelet_size * patch_size * patch_size * num_channels,
                )

            batch_size, _, num_channels = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(batch_size, -1, num_channels)

        # TODO: first validate if the training even works...
        # if video_lengths:
        #     for batch in range(batch_size):
        #         logits[batch, video_lengths[batch]:, :] = 0
        #         labels[batch, video_lengths[batch]:, :] = 0

        loss_fct = MSELoss()
        loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return VideoMAEForPreTrainingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
