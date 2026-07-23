#!/usr/bin/env python3

import regex as re
from collections import Counter, defaultdict
import gc


def compute_pre_token_counts(
    raw_corpus: str, special_tokens: list[str]
) -> dict[bytes, int]:

    sptk_regex = "|".join(map(re.escape, special_tokens))
    corpus = re.split(sptk_regex, raw_corpus)

    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat_rgx = re.compile(PAT)
    pre_token_counts = Counter(
        match.group(0).encode("utf-8")
        for word in corpus
        for match in re.finditer(pat_rgx, word)
    )

    del corpus
    gc.collect()
    return pre_token_counts


def train_with_pre_token_counts(
    pre_token_counts: dict[bytes, int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    merge_rounds = vocab_size - len(special_tokens) - 256
    assert merge_rounds >= 0, "Vocabulary size too small."

    # Actual training
    ## Output placeholders
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    ## Facilities for implementing the merging loop.
    counts: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in pre_token): count
        for pre_token, count in pre_token_counts.items()
    }
    del pre_token_counts
    gc.collect()

    for _ in range(merge_rounds):

        ## Compute the most common pair.
        merge_ranks = defaultdict(int)
        for bseq, count in counts.items():
            for i in range(len(bseq) - 1):
                merge_ranks[(bseq[i], bseq[i + 1])] += count

        if len(merge_ranks) == 0:
            break

        selected_merge = max(merge_ranks, key=lambda k: (merge_ranks[k], k))
        merges.append(selected_merge)

        ## Execute merge.
        vocab[len(vocab)] = selected_merge[0] + selected_merge[1]
        new_counts = defaultdict(int)
        for bseq, count in counts.items():
            merged_seq = []
            i = 0
            while i < len(bseq) - 1:
                if bseq[i] == selected_merge[0] and bseq[i + 1] == selected_merge[1]:
                    merged_seq.append(bseq[i] + bseq[i + 1])
                    i += 2
                else:
                    merged_seq.append(bseq[i])
                    i += 1
            if i == len(bseq) - 1:
                merged_seq.append(bseq[i])
            new_counts[tuple(merged_seq)] += count
        counts = new_counts

    # Add special tokens to the vocabulary too.
    vocab_length_after_merges = len(vocab)
    for i in range(len(special_tokens)):
        vocab[vocab_length_after_merges + i] = special_tokens[i].encode("utf-8")

    return vocab, merges


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path) as f:
        raw_corpus: str = f.read()

    pre_token_counts: dict[bytes, int] = compute_pre_token_counts(
        raw_corpus, special_tokens
    )

    del raw_corpus
    gc.collect()

    return train_with_pre_token_counts(pre_token_counts, vocab_size, special_tokens)


def fast_train_with_pre_token_counts(
    pre_token_counts: dict[bytes, int], vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    merge_rounds = vocab_size - len(special_tokens) - 256
    assert merge_rounds >= 0, "Vocabulary size too small."

    # Actual training
    ## Output placeholders
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[tuple[bytes, bytes]] = []

    ## Facilities for implementing the merging loop.
    ##
    ## In order to merge, we need to count the occurrences
    ## of pairs and find the maximum occurring one with the
    ## highest lexicographic order. And then we need to actually
    ## merge the pairs in the words they appear.
    ##
    ## We define the following facility objects for bookkeeping:
    ## 1. Mapping from word_id to the concrete word, a list of bytes objects.
    ## 2. Mapping counting occurrences of a word: word_id -> number of occurrences.
    ## About (2), note that this occurrence count never changes. The merging process maintains the
    ## count of words, as we don't merge accross word boundaries.
    ## 3. Mapping of pairs of bytes objects to their total occurrence count in the (merged so far) corpus.
    ## 4. Mapping of pairs of bytes objects to, in turn, a mapping of word_ids which words contain the pair,
    ## to the number of occurrences of the pair in the word with such word_id.
    ## 5. Min heap used to efficiently compute the pairs with maximum lexicographic order among those with
    ## maximum occurrence. We naturally need to use a clever key for element comparison so that the min heap
    ## acts as a max heap for the actual desired order. This is because Python3 heaps are all min-heaps.
    ## Facilities for implementing the merging loop.
    counts: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in pre_token): count
        for pre_token, count in pre_token_counts.items()
    }
    del pre_token_counts
    gc.collect()

    for _ in range(merge_rounds):

        ## Compute the most common pair.
        merge_ranks = defaultdict(int)
        for bseq, count in counts.items():
            for i in range(len(bseq) - 1):
                merge_ranks[(bseq[i], bseq[i + 1])] += count

        if len(merge_ranks) == 0:
            break

        selected_merge = max(merge_ranks, key=lambda k: (merge_ranks[k], k))
        merges.append(selected_merge)

        ## Execute merge.
        vocab[len(vocab)] = selected_merge[0] + selected_merge[1]
        new_counts = defaultdict(int)
        for bseq, count in counts.items():
            merged_seq = []
            i = 0
            while i < len(bseq) - 1:
                if bseq[i] == selected_merge[0] and bseq[i + 1] == selected_merge[1]:
                    merged_seq.append(bseq[i] + bseq[i + 1])
                    i += 2
                else:
                    merged_seq.append(bseq[i])
                    i += 1
            if i == len(bseq) - 1:
                merged_seq.append(bseq[i])
            new_counts[tuple(merged_seq)] += count
        counts = new_counts

    # Add special tokens to the vocabulary too.
    vocab_length_after_merges = len(vocab)
    for i in range(len(special_tokens)):
        vocab[vocab_length_after_merges + i] = special_tokens[i].encode("utf-8")

    return vocab, merges


def fast_train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path) as f:
        raw_corpus: str = f.read()

    pre_token_counts: dict[bytes, int] = compute_pre_token_counts(
        raw_corpus, special_tokens
    )

    del raw_corpus
    gc.collect()

    return fast_train_with_pre_token_counts(
        pre_token_counts, vocab_size, special_tokens
    )
