#!/usr/bin/env python3
import argparse
from pathlib import Path

from linkedinlm.tokenizer.corpus import END_OF_TEXT, iter_jsonl_texts
from linkedinlm.tokenizer.regex import RegexTokenizer

#training the tokenizer!!!


def show_progress(completed: int, total: int) -> None:
    """Render tokenizer-training progress on one terminal line."""
    width = 30
    fraction = completed / total if total else 1.0
    filled = round(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\rtraining merges [{bar}] {completed}/{total} ({fraction:6.1%})",
        end="",
        flush=True,
    )


# once again, not necessary, but convenient maybe
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the LinkedInLM regex BPE tokenizer from JSONL text."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/splits/train.jsonl"),
        help="JSONL training split.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("artifacts/tokenizers/linkedin-gpt4-2048"),
        help="Prefix for the generated .model and .vocab files.",
    )
    parser.add_argument("--field", default="output")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    documents = list(
        iter_jsonl_texts(
            args.input,
            field=args.field,
        )
    )
    if not documents:
        raise ValueError(f"no documents found in {args.input}")

    tokenizer = RegexTokenizer()
    progress_callback = None
    if not args.verbose:
        show_progress(0, args.vocab_size - 256)
        progress_callback = show_progress

    tokenizer.train_documents(
        documents,
        vocab_size=args.vocab_size,
        verbose=args.verbose,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        print()

    end_of_text_id = max(tokenizer.vocab) + 1
    tokenizer.register_special_tokens({END_OF_TEXT: end_of_text_id})

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(args.output_prefix))
    total_bytes = sum(len(document.encode("utf-8")) for document in documents)
    print(
        f"trained on {len(documents)} documents / {total_bytes} UTF-8 bytes; "
        f"mergeable vocab={len(tokenizer.vocab) - 1}; "
        f"{END_OF_TEXT}={end_of_text_id}"
    )
    print(f"saved {args.output_prefix}.model and {args.output_prefix}.vocab")


if __name__ == "__main__":
    main()
