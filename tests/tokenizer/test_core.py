import linkedinlm.tokenizer.core  


import pytest


@pytest.mark.parametrize(
    "test_ids, expected_counts",
    [
        (
            [1, 2, 3, 1, 2, 3, 1, 2],
            {(1, 2): 3, (2, 3): 2, (3, 1): 2},
        ),
        ([5, 5, 5, 5], {(5, 5): 3}),
        (
            [1, 2, 3, 4, 5],
            {(1, 2): 1, (2, 3): 1, (3, 4): 1, (4, 5): 1},
        ),
        ([1, 2, 1, 2, 1, 2], {(1, 2): 3, (2, 1): 2}),
        (
            [10, 20, 10, 20, 30],
            {(10, 20): 2, (20, 10): 1, (20, 30): 1},
        ),
        (
            [100, 200, 100, 200, 100],
            {(100, 200): 2, (200, 100): 2},
        ),
    ],
)
def test_count_pairs(test_ids: list[int], expected_counts: dict[tuple[int, int], int]):
    pair_counts = linkedinlm.tokenizer.core.count_pairs(test_ids)
    # Add assertions to verify the expected pair counts
    assert isinstance(pair_counts, dict), "Output should be a dictionary"
    for pair, count in pair_counts.items():
        assert isinstance(pair, tuple) and len(pair) == 2, "Keys should be pairs (tuples of length 2)"
        assert isinstance(count, int) and count > 0, "Counts should be positive integers"
    assert pair_counts == expected_counts, f"Expected counts {expected_counts}, but got {pair_counts}"


@pytest.mark.parametrize(
    "pair_counts, expected_pair",
    [
        ({(1, 2): 3, (2, 3): 2}, (1, 2)),  # Highest count is (1, 2)
        ({(1, 2): 1, (2, 3): 1}, (1, 2)),  # Tie, lexicographically smallest is (1, 2)
        ({(5, 6): 5}, (5, 6)),              # Only one pair
        ({(10, 20): 4, (20, 30): 4}, (10, 20)), # Tie with lexicographical order
    ],
)
def test_select_pair(pair_counts: dict[tuple[int, int], int], expected_pair: tuple[int, int]):
    selected_pair = linkedinlm.tokenizer.core.select_pair(pair_counts)
    assert selected_pair == expected_pair, f"Expected pair {expected_pair}, but got {selected_pair}"

@pytest.mark.parametrize(
    "token_ids, pair, new_token_id, expected_output",
    [
        ([1, 2, 3, 1, 2, 3], (1, 2), 99, [99, 3, 99, 3]),  # Merge (1, 2) into 99
        ([5, 5, 5], (5, 5), 10, [10, 5]),                  # Merge (5, 5) into 10
        ([1, 2, 3], (2, 3), 20, [1, 20]),                  # Merge (2, 3) into 20
        ([10, 20, 30], (40, 50), 60, [10, 20, 30]),       # Pair not found; no change
    ],
)
def test_merge_pair(token_ids: list[int], pair: tuple[int, int], new_token_id: int, expected_output: list[int]):
    merged_output = linkedinlm.tokenizer.core.merge_pair(token_ids, pair, new_token_id)
    assert merged_output == expected_output, f"Expected output {expected_output}, but got {merged_output}"
