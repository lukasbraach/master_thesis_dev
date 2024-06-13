from typing import Union

from transformers import SpeechEncoderDecoderModel, SpeechEncoderDecoderConfig, \
    PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedModel, AutoConfig, AutoModel

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder, SpatiotemporalEncoderConfig


class SignLanguageNet(SpeechEncoderDecoderModel):
    base_model_prefix = "sign_language_encoder_decoder"
    main_input_name = "input_values"

    def __init__(
            self,
            config: SpeechEncoderDecoderConfig = None,
            encoder: PreTrainedModel = None,
            decoder: LanguageDecoder = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    ) -> None:
        if config is None and tokenizer is not None:
            self.tokenizer = tokenizer

            config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
            config.update(decoder.config.to_diff_dict())

            config.num_beams = 5

        super().__init__(config=config, encoder=encoder, decoder=decoder)


if __name__ == "__main__":
    _ = SignLanguageNet()


AutoConfig.register("spatiotemporal-encoder", SpatiotemporalEncoderConfig)
AutoModel.register(SpatiotemporalEncoderConfig, SpatiotemporalEncoder)