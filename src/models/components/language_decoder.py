from transformers import Speech2Text2ForCausalLM, Speech2Text2Config


class LanguageDecoder(Speech2Text2ForCausalLM):

    def __init__(
            self,
            config: Speech2Text2Config = None
    ) -> None:
        print(f"LanguageDecoder.__init__ config: {config}")

        super().__init__(config)
        print(f"LanguageDecoder.__init__ self: {self}")

if __name__ == "__main__":
    _ = LanguageDecoder()
