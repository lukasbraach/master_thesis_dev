from typing import Union

from transformers import Speech2Text2ForCausalLM, Speech2Text2Config, PreTrainedTokenizer, PreTrainedTokenizerFast


class LanguageDecoder(Speech2Text2ForCausalLM):

    def __init__(
            self,
            config: Speech2Text2Config = None,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
    ) -> None:
        self.tokenizer = tokenizer

        config.pad_token_id = self.tokenizer.pad_token_id
        config.bos_token_id = self.tokenizer.bos_token_id
        config.eos_token_id = self.tokenizer.eos_token_id
        config.decoder_start_token_id = self.tokenizer.eos_token_id
        config.vocab_size = self.tokenizer.vocab_size

        super().__init__(config)


if __name__ == "__main__":
    _ = LanguageDecoder()
