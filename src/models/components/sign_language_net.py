from typing import Union

from transformers import SpeechEncoderDecoderModel, Speech2Text2Config, SpeechEncoderDecoderConfig, \
    PreTrainedTokenizerFast, PreTrainedTokenizer

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder, SpatiotemporalEncoderConfig


class SignLanguageNet(SpeechEncoderDecoderModel):
    base_model_prefix = "sign_language_encoder_decoder"
    main_input_name = "input_values"

    def __init__(
            self,
            encoder: SpatiotemporalEncoder,
            decoder: LanguageDecoder,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.tokenizer = tokenizer

        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        config.update(decoder.config.to_diff_dict())

        config.num_beams = 5

        super().__init__(config=config)


if __name__ == "__main__":
    _ = SignLanguageNet()
