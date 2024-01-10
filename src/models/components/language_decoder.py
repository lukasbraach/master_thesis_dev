from transformers import Speech2Text2ForCausalLM, Speech2Text2Config


class LanguageDecoder(Speech2Text2ForCausalLM):

    def __init__(
            self,
            config: Speech2Text2Config = None
    ) -> None:
        super().__init__(config)


if __name__ == "__main__":
    _ = LanguageDecoder()
