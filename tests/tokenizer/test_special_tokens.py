from pathlib import Path

import pytest

from linkedinlm.tokenizer.basic import BasicTokenizer
from linkedinlm.tokenizer.regex import RegexTokenizer


def test_register_special_token_adds_decodable_vocab_entry():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 256})

    assert tokenizer.vocab[256] == b"<|endoftext|>"
    assert tokenizer.decode([256]) == "<|endoftext|>"


@pytest.mark.parametrize(
    "special_tokens",
    [
        {"": 256},
        {"line\nbreak": 256},
        {"tab\tinside": 256},
        {"<|negative|>": -1},
        {"<|boolean|>": True},
        {"<|byte-collision|>": 255},
        {"<|first|>": 256, "<|second|>": 256},
    ],
)
def test_register_special_tokens_rejects_invalid_mappings(special_tokens):
    tokenizer = BasicTokenizer()

    with pytest.raises(ValueError):
        tokenizer.register_special_tokens(special_tokens)


def test_register_special_token_rejects_learned_token_id_collision():
    tokenizer = BasicTokenizer()
    tokenizer.train("aaaa", vocab_size=257)

    with pytest.raises(ValueError, match="collides"):
        tokenizer.register_special_tokens({"<|endoftext|>": 256})


def test_encode_rejects_special_token_by_default():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 256})

    with pytest.raises(ValueError, match="disallowed special token"):
        tokenizer.encode("hello<|endoftext|>")


def test_encode_ordinary_treats_special_literal_as_normal_text():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 256})
    text = "hello<|endoftext|>"

    token_ids = tokenizer.encode_ordinary(text)

    assert 256 not in token_ids
    assert tokenizer.decode(token_ids) == text


def test_encode_allows_explicit_special_token():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 256})

    token_ids = tokenizer.encode(
        "hello<|endoftext|>",
        allowed_special={"<|endoftext|>"},
    )

    assert token_ids[-1] == 256
    assert tokenizer.decode(token_ids) == "hello<|endoftext|>"


def test_encode_allows_all_registered_special_tokens():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens(
        {
            "<|startoftext|>": 256,
            "<|endoftext|>": 257,
        }
    )

    token_ids = tokenizer.encode(
        "<|startoftext|>hello<|endoftext|>",
        allowed_special="all",
    )

    assert token_ids[0] == 256
    assert token_ids[-1] == 257
    assert tokenizer.decode(token_ids) == "<|startoftext|>hello<|endoftext|>"


def test_encode_rejects_disallowed_member_of_registered_set():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens(
        {
            "<|startoftext|>": 256,
            "<|endoftext|>": 257,
        }
    )

    with pytest.raises(ValueError, match="endoftext"):
        tokenizer.encode(
            "<|startoftext|>hello<|endoftext|>",
            allowed_special={"<|startoftext|>"},
        )


def test_encode_rejects_unknown_allowed_special_token():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({"<|endoftext|>": 256})

    with pytest.raises(ValueError, match="unknown special token"):
        tokenizer.encode(
            "hello",
            allowed_special={"<|missing|>"},
        )


def test_encode_prefers_longest_overlapping_special_token():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens(
        {
            "<|tag|>": 256,
            "<|tag|>extended": 257,
        }
    )

    assert tokenizer.encode(
        "<|tag|>extended",
        allowed_special="all",
    ) == [257]


def test_regex_tokenizer_handles_specials_before_pretokenization():
    tokenizer = RegexTokenizer()
    tokenizer.train("hello hello 🚀🚀", vocab_size=265)
    special_id = max(tokenizer.vocab) + 1
    tokenizer.register_special_tokens({"<|endoftext|>": special_id})
    text = "hello<|endoftext|>🚀"

    token_ids = tokenizer.encode(text, allowed_special="all")

    assert special_id in token_ids
    assert tokenizer.decode(token_ids) == text


def test_special_tokens_survive_save_and_load(tmp_path: Path):
    original = RegexTokenizer()
    original.train("hello hello 🚀🚀", vocab_size=265)
    special_id = max(original.vocab) + 1
    original.register_special_tokens({"<|endoftext|>": special_id})
    file_prefix = tmp_path / "special-tokenizer"
    original.save(str(file_prefix))

    loaded = RegexTokenizer()
    loaded.load(str(file_prefix.with_suffix(".model")))

    text = "hello<|endoftext|>🚀"
    assert loaded.special_tokens == original.special_tokens
    assert loaded.encode(text, allowed_special="all") == original.encode(
        text,
        allowed_special="all",
    )
    assert loaded.decode(loaded.encode(text, allowed_special="all")) == text
