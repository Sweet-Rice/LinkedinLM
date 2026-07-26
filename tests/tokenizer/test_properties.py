from hypothesis import given, settings, strategies as st

from linkedinlm.tokenizer.basic import BasicTokenizer


UNICODE_SCALAR_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",)),
)

TRAINED_TOKENIZER = BasicTokenizer()
TRAINED_TOKENIZER.train(
    (
        "Building in public 🚀🚀\n"
        "Café collaboration meets こんにちは and 👩🏽‍💻.\n"
    )
    * 3,
    vocab_size=300,
)


@settings(max_examples=250, deadline=None)
@given(text=UNICODE_SCALAR_TEXT)
def test_untrained_tokenizer_round_trips_generated_unicode(text):
    tokenizer = BasicTokenizer()

    assert tokenizer.decode(tokenizer.encode(text)) == text


@settings(max_examples=250, deadline=None)
@given(text=UNICODE_SCALAR_TEXT)
def test_trained_tokenizer_round_trips_generated_unicode(text):
    assert TRAINED_TOKENIZER.decode(TRAINED_TOKENIZER.encode(text)) == text
