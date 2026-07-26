from linkedinlm.tokenizer.basic import BasicTokenizer
import pytest
tokenizer = BasicTokenizer()

"""
test suite for e2e encode and decode
"""
def test_basic_utf8_range_test():

    for code_point in range(0x10FFFF + 1):
            if 0xD800 <= code_point <= 0xDFFF:
                # Surrogates are not valid standalone Unicode scalar values.
                continue

            char = chr(code_point)
            encoded = tokenizer.encode(char)
            decoded = tokenizer.decode(encoded)
            assert decoded == char, (
                f"Failed for code point {code_point}: "
                f"{char!r} -> {encoded} -> {decoded!r}"
            )

def test_encode_ascii_bytes():
    for i in range(128):  # ASCII range
        char = chr(i)
        encoded = tokenizer.encode(char)
        assert encoded == [i], f"Failed for ASCII character {char}: {encoded}"



@pytest.mark.parametrize(
    "text", 
    [
        "",
        "Hello, World!", 
        "Python 3.9", 
        "Test123",
        "café",  # includes a non-ASCII character
        "こんにちは",  # Japanese greeting
        "😀",  # Emoji
        "👩🏽‍💻",  # Emoji with skin tone and a zero-width joiner
        "❤️",  # Emoji containing a variation selector
        "𐍈",  # A character outside the BMP
        "line one\nline two",  # includes a newline
        "tab\tseparated",  # includes a tab
    ],
)
def test_unicode_roundtrip(text):
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text, f"Failed for text '{text}': {encoded} -> {decoded}"

def test_decode_accepts_an_integer_tuple():
    text = "Hello, World!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(tuple(encoded))  # Pass a tuple instead of a list
    assert decoded == text, f"Failed for text '{text}': {encoded} -> {decoded}"

@pytest.mark.parametrize(
    "invalid_id",
    [
        -1,
        256
    ],
)
def test_decode_rejects_values_outside_byte_range(invalid_id):
    with pytest.raises(ValueError):
        tokenizer.decode([invalid_id])

@pytest.mark.parametrize(
    "token_ids",
    [
        [0xC0],  # Overlong encoding
        [0xC1],  # Overlong encoding
        [0xE0, 0x80],  # Incomplete sequence
        [0xF5],  # Invalid start byte for UTF-8
        [0xFF],  # Invalid byte in UTF-8
    ],
)
def test_decode_rejects_invalid_utf8(token_ids):
    # This test checks if the decode method can handle invalid UTF-8 sequences.
    # We'll create a list of byte values that do not form valid UTF-8 sequences.

    with pytest.raises(UnicodeDecodeError):
        tokenizer.decode(token_ids)
