import json
from pathlib import Path

import pytest

from linkedinlm.tokenizer.basic import BasicTokenizer
from linkedinlm.tokenizer.corpus import (
    END_OF_TEXT,
    encode_documents_with_eot,
    iter_jsonl_texts,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_iter_jsonl_texts_extracts_only_requested_field(tmp_path: Path):
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(
        corpus_path,
        [
            {
                "instruction": "ignored instruction",
                "output": "first post",
                "messages": [{"content": "duplicated content"}],
            },
            {
                "instruction": "also ignored",
                "output": "second post 🚀",
            },
        ],
    )

    assert list(iter_jsonl_texts(corpus_path)) == [
        "first post",
        "second post 🚀",
    ]


def test_iter_jsonl_texts_rejects_missing_or_non_string_field(tmp_path: Path):
    corpus_path = tmp_path / "corpus.jsonl"
    write_jsonl(corpus_path, [{"output": None}])

    with pytest.raises(ValueError, match="must be a string"):
        list(iter_jsonl_texts(corpus_path))


def test_train_documents_does_not_merge_across_documents():
    tokenizer = BasicTokenizer()
    tokenizer.train_documents(["a", " a"], vocab_size=258)

    assert (ord("a"), ord(" ")) not in tokenizer.merges


def test_encode_documents_appends_eot_after_every_document():
    tokenizer = BasicTokenizer()
    tokenizer.register_special_tokens({END_OF_TEXT: 256})

    token_ids = encode_documents_with_eot(
        tokenizer,
        ["first", "second 🚀"],
    )
    end_of_text_id = tokenizer.special_tokens[END_OF_TEXT]

    assert token_ids.count(end_of_text_id) == 2
    assert token_ids[-1] == end_of_text_id
    assert tokenizer.decode(token_ids) == (
        f"first{END_OF_TEXT}second 🚀{END_OF_TEXT}"
    )
