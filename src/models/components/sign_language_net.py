from transformers import SpeechEncoderDecoderModel, Speech2Text2Config, SpeechEncoderDecoderConfig, \
    PreTrainedTokenizerFast

from src.models.components.language_decoder import LanguageDecoder
from src.models.components.spatiotemporal_encoder import SpatiotemporalEncoder, SpatiotemporalEncoderConfig


class SignLanguageNet(SpeechEncoderDecoderModel):
    base_model_prefix = "sign_language_encoder_decoder"
    main_input_name = "input_values"

    def __init__(
            self,
            tokenizer_file="../etc/rwth_phoenix_tokenizer.json",
    ) -> None:
        self.tokenizer = PreTrainedTokenizerFast(
            model_input_names=['input_values'],
            pad_token="__PAD__",
            bos_token="__ON__",
            eos_token="__OFF__",
            unk_token="__UNK__",
            tokenizer_file=tokenizer_file,
        )

        encoder = SpatiotemporalEncoder(SpatiotemporalEncoderConfig())
        decoder = LanguageDecoder(Speech2Text2Config(
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            decoder_start_token_id=self.tokenizer.eos_token_id,
            vocab_size=self.tokenizer.vocab_size,
        ))

        config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        config.update(decoder.config.to_diff_dict())

        super().__init__(config=config)


if __name__ == "__main__":
    _ = SignLanguageNet()
