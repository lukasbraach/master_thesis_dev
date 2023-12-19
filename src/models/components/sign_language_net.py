from transformers import SpeechEncoderDecoderModel

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder, SpatiotemporalEncoderConfig


class SignLanguageNet(SpeechEncoderDecoderModel):

    def __init__(
            self,
    ) -> None:
        encoder = SpatiotemporalEncoder(SpatiotemporalEncoderConfig())
        decoder = LanguageDecoder()

        super().__init__(encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    _ = SignLanguageNet()
