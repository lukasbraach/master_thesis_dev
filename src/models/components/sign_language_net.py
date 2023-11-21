from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn, Tensor
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder


class SignLanguageNet(nn.Module):

    def __init__(
            self,
            spatiotemporal_encoder: SpatiotemporalEncoder,
            language_decoder: LanguageDecoder,
    ) -> None:
        super().__init__()

        self.encoder: SpatiotemporalEncoder = spatiotemporal_encoder
        self.decoder: LanguageDecoder = language_decoder

        self.tokenizer = Tokenizer(WordLevel(vocab={}))

    def forward(self, images: Tensor) -> CausalLMOutputWithCrossAttentions:
        x = self.encoder.preprocess_image()
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__ == "__main__":
    _ = SignLanguageNet()
