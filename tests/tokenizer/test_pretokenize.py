import pytest
from hypothesis import given, settings, strategies as st

from linkedinlm.tokenizer.pretokenize import split_gpt4


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "hello123",
        "We're hiring!",
        "Numbers: 1234567890",
        "café こんにちは",
        "Building in public 🚀🚀",
        "👩🏽‍💻❤️",
        "line one\nline two\r\nline three",
        "spaces   tabs\tand trailing whitespace   ",
        "#Hiring @LinkedIn https://example.com/jobs",
    ],
)
def test_gpt4_split_is_lossless(text):
    pieces = split_gpt4(text)

    assert "".join(pieces) == text
    assert all(piece for piece in pieces)


@pytest.mark.parametrize(
    "text, expected_pieces",
    [
        ("hello123", ["hello", "123"]),
        ("1234", ["123", "4"]),
        ("can't", ["can", "'t"]),
        ("hello world", ["hello", " world"]),
    ],
)
def test_gpt4_split_has_expected_boundaries(text, expected_pieces):
    assert split_gpt4(text) == expected_pieces


UNICODE_SCALAR_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
)


@settings(max_examples=250, deadline=None)
@given(text=UNICODE_SCALAR_TEXT)
def test_gpt4_split_round_trips_generated_unicode(text):
    assert "".join(split_gpt4(text)) == text
