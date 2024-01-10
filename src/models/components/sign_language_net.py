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
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    ) -> None:
        self.tokenizer = tokenizer

        encoder = SpatiotemporalEncoder(SpatiotemporalEncoderConfig())
        decoder = LanguageDecoder(Speech2Text2Config(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,

            d_model=512,
            decoder_ffn_dim=768,
            decoder_layers=6,
            decoder_attention_heads=8,

            layerdrop=0.05
        ))

        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        config.update(decoder.config.to_diff_dict())

        super().__init__(config=config)


if __name__ == "__main__":
    _ = SignLanguageNet()
