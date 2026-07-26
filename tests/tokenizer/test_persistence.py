from pathlib import Path

import pytest

from linkedinlm.tokenizer.basic import BasicTokenizer


def test_save_creates_model_and_vocab_files(tmp_path: Path):
    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=259)
    file_prefix = tmp_path / "tokenizer"

    tokenizer.save(str(file_prefix))

    assert file_prefix.with_suffix(".model").is_file()
    assert file_prefix.with_suffix(".vocab").is_file()


def test_save_load_preserves_trained_emoji_tokenizer(tmp_path: Path):
    original = BasicTokenizer()
    original.train("🚀🚀🚀", vocab_size=259)
    file_prefix = tmp_path / "emoji-tokenizer"
    original.save(str(file_prefix))

    loaded = BasicTokenizer()
    loaded.load(str(file_prefix.with_suffix(".model")))

    text = "🚀🚀"
    assert loaded.merges == original.merges
    assert loaded.vocab == original.vocab
    assert loaded.encode(text) == original.encode(text)
    assert loaded.decode(loaded.encode(text)) == text


def test_save_is_deterministic_for_identical_training(tmp_path: Path):
    first = BasicTokenizer()
    second = BasicTokenizer()
    first.train("aaabdaaabac", vocab_size=259)
    second.train("aaabdaaabac", vocab_size=259)

    first_prefix = tmp_path / "first"
    second_prefix = tmp_path / "second"
    first.save(str(first_prefix))
    second.save(str(second_prefix))

    assert first_prefix.with_suffix(".model").read_bytes() == (
        second_prefix.with_suffix(".model").read_bytes()
    )
    assert first_prefix.with_suffix(".vocab").read_bytes() == (
        second_prefix.with_suffix(".vocab").read_bytes()
    )


def test_load_rejects_unsupported_model_version(tmp_path: Path):
    model_file = tmp_path / "unsupported.model"
    model_file.write_text(
        "linkedinlm v999\n\n0\n",
        encoding="utf-8",
    )

    tokenizer = BasicTokenizer()
    with pytest.raises(ValueError, match="Unsupported model version"):
        tokenizer.load(str(model_file))
