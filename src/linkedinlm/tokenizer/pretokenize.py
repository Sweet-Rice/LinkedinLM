import regex as re


GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
COMPILED_GPT4_SPLIT_PATTERN = re.compile(GPT4_SPLIT_PATTERN, re.UNICODE)


def split_pattern(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, text)


def split_gpt4(text: str) -> list[str]:
    return COMPILED_GPT4_SPLIT_PATTERN.findall(text)
