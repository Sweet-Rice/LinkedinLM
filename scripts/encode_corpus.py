#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

from linkedinlm.tokenizer.corpus import (
    END_OF_TEXT,
    encode_documents_with_eot,
    iter_jsonl_texts,
)
from linkedinlm.tokenizer.regex import RegexTokenizer


# calm file that wraps the tokenizer and tokenizes the corpus for model training
#optional func i'd say, but here for convenience sake and less headache
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode JSONL posts as little-endian uint32 GPT token IDs."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/train.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/corpora/train.bin"),
    )
    parser.add_argument("--field", default="output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = RegexTokenizer()
    tokenizer.load(str(args.model))
    documents = list(
        iter_jsonl_texts(
            args.input,
            field=args.field,
        )
    )
    token_ids = encode_documents_with_eot(
        tokenizer,
        documents,
        end_of_text=END_OF_TEXT,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as token_file:
        for start in range(0, len(token_ids), 16_384):
            batch = token_ids[start:start + 16_384]
            token_file.write(struct.pack(f"<{len(batch)}I", *batch))

    metadata = {
        "format": "uint32",
        "endianness": "little",
        "documents": len(documents),
        "tokens": len(token_ids),
        "end_of_text": END_OF_TEXT,
        "end_of_text_id": tokenizer.special_tokens[END_OF_TEXT],
        "source": str(args.input),
        "field": args.field,
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"encoded {len(documents)} documents into {len(token_ids)} tokens")
    print(f"saved {args.output} and {metadata_path}")


if __name__ == "__main__":
    main()
