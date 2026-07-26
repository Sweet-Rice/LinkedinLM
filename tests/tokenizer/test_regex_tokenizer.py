from pathlib import Path

import pytest

from linkedinlm.tokenizer.basic import BasicTokenizer
from linkedinlm.tokenizer.pretokenize import GPT4_SPLIT_PATTERN
from linkedinlm.tokenizer.regex import RegexTokenizer


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Hello, LinkedIn!",
        "We're building in public 🚀",
        "café こんにちは 👩🏽‍💻",
        "1234567890\n#Hiring",
    ],
)
def test_regex_tokenizer_round_trips_text(text):
    tokenizer = RegexTokenizer()
    tokenizer.train(
        "Hello hello 123 123 🚀🚀 café café\nBuilding in public.",
        vocab_size=280,
    )

    token_ids = tokenizer.encode(text)

    assert all(isinstance(token_id, int) for token_id in token_ids)
    assert tokenizer.decode(token_ids) == text


def test_regex_training_never_learns_across_piece_boundaries():
    tokenizer = RegexTokenizer()
    tokenizer.train("aa aa", vocab_size=270)

    assert (ord("a"), ord(" ")) not in tokenizer.merges


def test_basic_and_regex_tokenizers_differ_at_piece_boundaries():
    basic = BasicTokenizer()
    regex = RegexTokenizer()
    basic.train("a a", vocab_size=258)
    regex.train("a a", vocab_size=258)

    assert len(basic.encode("a a")) == 1
    assert len(regex.encode("a a")) == 2
    assert basic.decode(basic.encode("a a")) == "a a"
    assert regex.decode(regex.encode("a a")) == "a a"


def test_regex_tokenizer_uses_gpt4_pattern_by_default():
    tokenizer = RegexTokenizer()

    assert tokenizer.pattern == GPT4_SPLIT_PATTERN


def test_regex_tokenizer_persists_pattern(tmp_path: Path):
    original = RegexTokenizer()
    original.train("hello123 hello123 🚀🚀", vocab_size=270)
    file_prefix = tmp_path / "regex-tokenizer"
    original.save(str(file_prefix))

    loaded = RegexTokenizer()
    loaded.load(str(file_prefix.with_suffix(".model")))

    text = "hello123 🚀"
    assert loaded.pattern == GPT4_SPLIT_PATTERN
    assert loaded.merges == original.merges
    assert loaded.encode(text) == original.encode(text)
    assert loaded.decode(loaded.encode(text)) == text
