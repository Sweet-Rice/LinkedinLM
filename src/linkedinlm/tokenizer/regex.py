from linkedinlm.tokenizer.basic import BasicTokenizer
from linkedinlm.tokenizer.pretokenize import GPT4_SPLIT_PATTERN, split_pattern


class RegexTokenizer(BasicTokenizer):
    """Byte-level BPE tokenizer that prevents merges across regex pieces."""

    def __init__(self, pattern: str = GPT4_SPLIT_PATTERN):
        super().__init__()
        self.pattern = pattern

    def _split_text(self, text: str) -> list[str]:
        return split_pattern(text, self.pattern)
