from collections.abc import Sequence

from linkedinlm.tokenizer.core import Pair, count_pairs, merge_pair, select_pair


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

    def train(
        self,
        text: str,
        vocab_size: int,
        verbose: bool = False,
    ) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        iterations = vocab_size - 256

        text_bytes = text.encode("utf-8")  # Raw UTF-8 bytes, including emoji.
        token_ids = list(text_bytes)

        merges: dict[Pair, int] = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(iterations):
            pair_counts = count_pairs(token_ids)
            if not pair_counts:
                break

            pair = select_pair(pair_counts)

            new_token_id = 256 + i
            token_ids = merge_pair(token_ids, pair, new_token_id)
            merges[pair] = new_token_id
            vocab[new_token_id] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(
                    f"merge {i + 1}/{iterations}: {pair} -> {new_token_id} "
                    f"({vocab[new_token_id]}) had {pair_counts[pair]} occurrences"
                )

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        """
        Encode text as UTF-8 byte IDs, then apply learned merges by rank.
        """
        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            pair_counts = count_pairs(ids)
            learned_pairs = [pair for pair in pair_counts if pair in self.merges]
            if not learned_pairs:
                break

            pair = min(learned_pairs, key=self.merges.__getitem__)
            new_token_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_token_id)

        return ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """
        Decode token IDs by joining their vocabulary bytes as strict UTF-8.
        """
        try:
            text_bytes = b"".join(self.vocab[token_id] for token_id in token_ids)
        except KeyError as error:
            raise ValueError(f"invalid token ID: {error.args[0]}") from error

        return text_bytes.decode("utf-8")
