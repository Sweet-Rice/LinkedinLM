import pytest
import linkedinlm.tokenizer.basic


@pytest.mark.parametrize(
    "text, vocab_size, expected_vocab_length",
    [
        ("aaabdaaabac", 259, 259),  # Simple case with repeated pairs
        ("abcabcabc", 260, 260),    # All pairs are the same
        ("abcdef", 256, 256),       # No pairs to merge
    ]
)
def test_train_basic_tokenizer(text: str, vocab_size: int, expected_vocab_length: int):
    tokenizer = linkedinlm.tokenizer.basic.BasicTokenizer()
    tokenizer.train(text, vocab_size, verbose=False)
    assert len(tokenizer.vocab) == expected_vocab_length, (
        f"Expected vocab length {expected_vocab_length}, but got {len(tokenizer.vocab)}"
    )


def test_train_learns_and_round_trips_repeated_emoji():
    tokenizer = linkedinlm.tokenizer.basic.BasicTokenizer()
    tokenizer.train("🚀🚀🚀", vocab_size=259)

    encoded = tokenizer.encode("🚀🚀")

    assert tokenizer.vocab[258] == "🚀".encode("utf-8")
    assert encoded == [258, 258]
    assert tokenizer.decode(encoded) == "🚀🚀"


def test_train_reports_merge_progress():
    tokenizer = linkedinlm.tokenizer.basic.BasicTokenizer()
    updates: list[tuple[int, int]] = []

    tokenizer.train(
        "aaaa",
        vocab_size=258,
        progress_callback=lambda completed, total: updates.append(
            (completed, total)
        ),
    )

    assert updates == [(1, 2), (2, 2)]
