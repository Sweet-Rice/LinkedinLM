import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from linkedinlm.tokenizer.basic import BasicTokenizer


END_OF_TEXT = "<|endoftext|>"


def iter_jsonl_texts(
    path: str | Path,
    *,
    field: str = "output",
) -> Iterator[str]:
    """Yield one selected text field per JSONL record."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from error

            text = record.get(field)
            if not isinstance(text, str):
                raise ValueError(
                    f"field {field!r} on line {line_number} must be a string"
                )

            yield text


def encode_documents_with_eot(
    tokenizer: BasicTokenizer,
    documents: Iterable[str],
    *,
    end_of_text: str = END_OF_TEXT,
) -> list[int]:
    """Encode documents independently and terminate each with the EOT ID."""
    try:
        end_of_text_id = tokenizer.special_tokens[end_of_text]
    except KeyError as error:
        raise ValueError(
            f"special token {end_of_text!r} is not registered"
        ) from error

    token_ids: list[int] = []
    for document in documents:
        token_ids.extend(tokenizer.encode_ordinary(document))
        token_ids.append(end_of_text_id)
    return token_ids
