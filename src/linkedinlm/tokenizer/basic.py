import re
from collections.abc import Iterable, Mapping, Sequence, Set as AbstractSet
from typing import Callable, Literal

from linkedinlm.tokenizer.core import Pair, count_pairs, merge_pair, render_token, select_pair


ProgressCallback = Callable[[int, int], None]


class BasicTokenizer:
    merges: dict[Pair, int]
    vocab: dict[int, bytes]

    def __init__(self):
        self.merges = {}
        self.pattern = ""
        self.special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self) -> dict[int, bytes]:
        vocab = {idx: bytes([idx]) for idx in range(256)}  # Initialize with single-byte tokens
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")

        return vocab

    def register_special_tokens(
        self,
        special_tokens: Mapping[str, int],
    ) -> None:
        """Replace the registered special-token mapping after validation."""
        mergeable_ids = set(range(256)) | set(self.merges.values())
        special_ids: list[int] = []

        for special, token_id in special_tokens.items():
            if not isinstance(special, str) or not special:
                raise ValueError("special tokens must be non-empty strings")
            if any(character in special for character in ("\t", "\n", "\r")):
                raise ValueError("special tokens cannot contain tabs or newlines")
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError("special token IDs must be integers")
            if token_id < 0:
                raise ValueError("special token IDs must be non-negative")
            if token_id in mergeable_ids:
                raise ValueError(
                    f"special token ID {token_id} collides with mergeable vocabulary"
                )
            special_ids.append(token_id)

        if len(special_ids) != len(set(special_ids)):
            raise ValueError("special token IDs must be unique")

        self.special_tokens = dict(special_tokens)
        self.vocab = self._build_vocab()

    def _split_text(self, text: str) -> list[str]:
        """Return independent pieces across which BPE merges may not occur."""
        return [text]

    @staticmethod
    def _count_chunk_pairs(token_chunks: Sequence[Sequence[int]]) -> dict[Pair, int]:
        pair_counts: dict[Pair, int] = {}
        for chunk in token_chunks:
            for pair, count in count_pairs(chunk).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + count
        return pair_counts
    # a little confusing. train trains on a single string, train_documents trains
    # train chunks is underlying infra that prevents cross document merges
    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self._train_chunks(
            self._split_text(text),
            vocab_size=vocab_size,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def train_documents(
        self,
        documents: Iterable[str],
        vocab_size: int,
        verbose: bool = False,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Train across documents without permitting cross-document merges."""
        text_chunks = (
            chunk
            for document in documents
            for chunk in self._split_text(document)
        )
        self._train_chunks(
            text_chunks,
            vocab_size=vocab_size,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    def _train_chunks(
        self,
        text_chunks: Iterable[str],
        *,
        vocab_size: int,
        verbose: bool,
        progress_callback: ProgressCallback | None,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        iterations = vocab_size - 256

        token_chunks = [list(chunk.encode("utf-8")) for chunk in text_chunks]

        merges: dict[Pair, int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(iterations):
            pair_counts = self._count_chunk_pairs(token_chunks)
            if not pair_counts:
                break

            pair = select_pair(pair_counts)

            new_token_id = 256 + i
            token_chunks = [
                merge_pair(chunk, pair, new_token_id)
                for chunk in token_chunks
            ]
            merges[pair] = new_token_id
            vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{iterations}: {pair} -> {new_token_id} "
                    f"({vocab[new_token_id]}) had {pair_counts[pair]} occurrences"
                )
            if progress_callback is not None:
                progress_callback(i + 1, iterations)

        self.merges = merges
        self.vocab = vocab

    def save(self, path: str )-> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        .vocab is human readable version of .model
        largely borrowed from andrej karpathy's minbpe 
        """

        model_file = path + ".model"
        vocab_file = path + ".vocab"

        with open(model_file, "w", encoding="utf-8") as f:
            f.write("linkedinlm v1 \n")
            f.write(f"{self.pattern}\n")

            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special}\t{idx}\n")

            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")

        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding = "utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)

                if idx in inverted_merges:

                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}]-> [{s}]{idx}\n")
                else:
                    f.write(f"[{s}]{idx}\n")

    def load(self, model_file) -> None:
        """
        loads only model file
        also borrows from andrej karpathy's minbpe
        """ 
        if not model_file.endswith(".model"):
            raise ValueError("Path must end with .model")

        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r', encoding="utf-8") as f:

            version = f.readline().strip()
            if version != "linkedinlm v1":
                raise ValueError(f"Unsupported model version: {version}")

            self.pattern = f.readline().rstrip("\n")

            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().rstrip("\n").split("\t")
                special_tokens[special] = int(special_idx)

            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = {}
        self.vocab = self._build_vocab()
        self.register_special_tokens(special_tokens)

    def _encode_chunk(self, text_bytes: bytes) -> list[int]:
        ids = list(text_bytes)
        while len(ids) >= 2:
            pair_counts = count_pairs(ids)
            learned_pairs = [pair for pair in pair_counts if pair in self.merges]
            if not learned_pairs:
                break

            pair = min(learned_pairs, key=self.merges.__getitem__)
            new_token_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_token_id)

        return ids

    def encode_ordinary(self, text: str) -> list[int]:
        """
        Encode text without recognizing registered special-token strings.
        """
        token_ids: list[int] = []
        for chunk in self._split_text(text):
            token_ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return token_ids

    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
    ) -> list[int]:
        """Encode text while recognizing only explicitly allowed special tokens."""
        if not self.special_tokens:
            return self.encode_ordinary(text)

        registered = set(self.special_tokens)
        if allowed_special == "all":
            allowed = registered
        elif isinstance(allowed_special, AbstractSet):
            allowed = set(allowed_special)
            unknown = allowed - registered
            if unknown:
                names = ", ".join(sorted(repr(token) for token in unknown))
                raise ValueError(f"unknown special token(s): {names}")
        else:
            raise TypeError(
                "allowed_special must be 'all' or a set of registered token strings"
            )

        disallowed = registered - allowed
        found_disallowed = [
            (text.find(token), -len(token), token)
            for token in disallowed
            if token in text
        ]
        if found_disallowed:
            _, _, token = min(found_disallowed)
            raise ValueError(
                f"encountered disallowed special token {token!r}; "
                "allow it explicitly or call encode_ordinary()"
            )

        allowed_in_text = [token for token in allowed if token in text]
        if not allowed_in_text:
            return self.encode_ordinary(text)

        alternatives = "|".join(
            re.escape(token)
            for token in sorted(allowed_in_text, key=lambda token: (-len(token), token))
        )
        parts = re.split(f"({alternatives})", text)

        token_ids: list[int] = []
        for part in parts:
            if part in allowed:
                token_ids.append(self.special_tokens[part])
            else:
                token_ids.extend(self.encode_ordinary(part))
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode token IDs by joining their vocabulary bytes as strict UTF-8.
        """
        try:
            text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"invalid token ID: {error.args[0]}") from error

        return text_bytes.decode("utf-8")
