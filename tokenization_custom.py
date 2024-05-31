from transformers import PreTrainedTokenizerFast, AddedToken
from typing import Optional, Tuple


class CustomTokenizer(PreTrainedTokenizerFast):

    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_file=None,
        unk_token="<|end|>",
        bos_token=None,
        eos_token="<|end|>",
        pad_token="<|end|>",
        **kwargs
    ):
        bos_token = (
            AddedToken(
                bos_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(bos_token, str)
            else bos_token
        )
        eos_token = (
            AddedToken(
                eos_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(eos_token, str)
            else eos_token
        )
        unk_token = (
            AddedToken(
                unk_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(unk_token, str)
            else unk_token
        )
        pad_token = (
            AddedToken(
                pad_token, lstrip=False, rstrip=False, special=True, normalized=False
            )
            if isinstance(pad_token, str)
            else pad_token
        )

        super().__init__(
            tokenizer_file=tokenizer_file,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            **kwargs,
        )

    # Copied from transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast.save_vocabulary
    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        files = self._tokenizer.model.save(save_directory, name=filename_prefix)
        return tuple(files)
