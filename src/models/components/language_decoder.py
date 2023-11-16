import torch
import transformers
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


class LanguageDecoder(nn.Module):

    def __init__(
            self,
    ) -> None:
        super().__init__()

        config = transformers.Speech2Text2Config(
            vocab_size=1,
            d_model=768,
            decoder_ffn_dim=2048,
            decoder_layers=6,
            decoder_attention_heads=8,
        )

        self.model = transformers.Speech2Text2ForCausalLM(config)

    def forward(self, x: torch.Tensor) -> CausalLMOutputWithCrossAttentions:
        x = self.model(x)
        return x


if __name__ == "__main__":
    _ = LanguageDecoder()
