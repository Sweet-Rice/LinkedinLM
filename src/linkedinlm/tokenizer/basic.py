from collections.abc import Sequence

from linkedinlm.tokenizer.core import Pair, count_pairs, merge_pair, render_token, select_pair


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


    def save(self, path: str )-> None:
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        .vocab is human readable version of .model
        largely borrowed from andrej karpathy's minbpe 
        """

        model_file = path + ".model"
        vocab_file = path + ".vocab"

        with open (model_file, 'w') as f:
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

            self.pattern = f.readline().strip()

            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split("\t")
                special_tokens[special] = int(special_idx)

            for line in f:
                idx1, idx2 = map(int, line.strip().split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()           


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
