import transformers
from transformers import Speech2Text2ForCausalLM, Speech2Text2Config


class LanguageDecoder(Speech2Text2ForCausalLM):

    def __init__(
            self,
            config: Speech2Text2Config = Speech2Text2Config(
                d_model=1024,
                decoder_ffn_dim=2048,
                decoder_layers=6,
                decoder_attention_heads=8,
            )
    ) -> None:
        super().__init__(config)


if __name__ == "__main__":
    _ = LanguageDecoder()
