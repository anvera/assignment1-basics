#!/usr/bin/env python3

import regex as re
from collections import Counter, defaultdict
import gc
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.train_bpe import compute_pre_token_counts, train_with_pre_token_counts
from multiprocessing import Pool
from functools import partial


def pretokenization_and_counting(
    endpoints: tuple[int], input_path: str, special_tokens: list[str]
):

    begin, end = endpoints[0], endpoints[1]

    with open(input_path, "rb") as f:
        f.seek(begin)
        raw_corpus = f.read(end - begin).decode("utf-8", errors="ignore")

    return compute_pre_token_counts(raw_corpus, special_tokens)


def train_bpe_multiproc(
    input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a byte-pair encoding tokenizer on the
    file at the input path, with maximum given
    vocabulary size, and with the given list of
    special tokens. Outputs a tuple with an index
    of the determined vocabulary,  as well as the
    list of BPE merges, the latter sorted by order
    of creation."""

    assert num_processes >= 1

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    process_inputs = list(
        (start, end) for start, end in zip(boundaries[:-1], boundaries[1:])
    )

    with Pool(processes=num_processes) as pool:
        pre_token_counter_list = pool.map(
            partial(
                pretokenization_and_counting,
                input_path=input_path,
                special_tokens=special_tokens,
            ),
            process_inputs,
            1,
        )

    pre_token_counts = Counter()
    for counter in pre_token_counter_list:
        pre_token_counts += counter

    # Training
    return train_with_pre_token_counts(pre_token_counts, vocab_size, special_tokens)
