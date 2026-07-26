from collections.abc import Mapping, Sequence
import unicodedata
Pair = tuple[int, int]

def count_pairs(token_ids: Sequence[int])-> dict[Pair, int]:
    pair_counts: dict[Pair, int] = {}
    for i in range(len(token_ids) - 1):
        pair = (token_ids[i], token_ids[i + 1])
        if pair in pair_counts:
            pair_counts[pair] += 1
        else:
            pair_counts[pair] = 1
    return pair_counts


def select_pair(pair_counts: Mapping[Pair, int]) -> Pair:
    sorted_pairs = sorted(pair_counts.items(), key=lambda item: item[1], reverse=True)
    if sorted_pairs:
        highest_counts = [sorted_pairs[0]]
        for pair, count in sorted_pairs:
            if count == highest_counts[0][1]:
                highest_counts.append((pair, count))
            else:
                break
        #Select lexicographically smallest pair among those with the highest count
        return min(pair for pair, count in highest_counts)
            
    else:
        raise ValueError("No pairs found in the input token IDs.")

def merge_pair(
    token_ids: Sequence[int],
    pair: Pair,
    new_token_id: int,
) -> list[int]:
    new_token_ids: list[int] = []
    i = 0
    while i < len(token_ids):
        #if not at the very last position and the pair matches, replace
        if token_ids[i] == pair[0] and i < len(token_ids) - 1 and token_ids[i + 1] == pair[1]:
            new_token_ids.append(new_token_id)
            i += 2  # Skip the next token since it's part of the merged pair
        else:
            new_token_ids.append(token_ids[i])
            i += 1
    return new_token_ids



def replace_control_characters(s: str) -> str:
    """Replace control characters in a string with their Unicode escape sequences."""
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t:bytes) -> str:
    """pretty print, escaping control chars"""
    s = t.decode("utf-8", errors="replace")
    s = replace_control_characters(s)
    return s