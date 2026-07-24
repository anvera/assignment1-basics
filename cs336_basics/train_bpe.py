#!/usr/bin/env python3

import regex as re
from collections import Counter, defaultdict
import gc
from heapq import heapify, heappop, heappush


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
    ### Note: a list is perfect for this, the ID is the index in this list,
    ### which is constructed one time and never changes.
    words: list[list[bytes]] = []
    ## 2. Mapping counting occurrences of a word: word_id -> number of occurrences.
    word_count: list[int] = []

    for pre_token, pt_count in pre_token_counts.items():
        words.append(list(bytes([b]) for b in pre_token))
        word_count.append(pt_count)

    del pre_token_counts
    gc.collect()

    ## About (2), note that this occurrence count never changes. The merging process maintains the
    ## count of words, as we don't merge accross word boundaries.
    ## 3. Mapping of pairs of bytes objects to their total occurrence count in the (merged so far) corpus.
    ## 4. Mapping of pairs of bytes objects to, in turn, a mapping of word_ids which words contain the pair,
    ## to the number of occurrences of the pair in the word with such word_id.
    total_occurrences_for_pair: dict[tuple[bytes, bytes], int] = defaultdict(int)
    ## TODO: remove the occurrences per word. We don't use it. We only need the list of words a pair appears on,
    ## not the exact positions because anyway we're iterating thorough it.
    per_word_occurrences_for_pair: dict[tuple[bytes, bytes], dict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for word_id, word in enumerate(words):
        # Gather occurrences of pairs in the current word.
        pair_occurrences: dict[tuple[bytes, bytes], int] = defaultdict(int)
        for i in range(len(word) - 1):
            pair_occurrences[(word[i], word[i + 1])] += 1
        # Add to total occurrences and per_word occurrences.
        for p, occ in pair_occurrences.items():
            total_occurrences_for_pair[p] += occ * word_count[word_id]
            per_word_occurrences_for_pair[p][word_id] += occ


    ## 5. Heap used to efficiently compute the pairs with maximum lexicographic order among those with
    ## maximum occurrence. Note: Python only offers max-heap starting on 3.14. That's why we need a
    ## helper class to invert the default order and turn the min-heap into a max-heap.
    class OrderInverterWrapper:
        def __init__(self, val):
            self.val = val

        def __lt__(self, other):
            self.val > other.val

    heap: list[tuple[int, bytes, bytes]] = [
        OrderInverterWrapper((count, p)) for p, count in total_occurrences_for_pair.items()
    ]
    heapify(heap)

    for _ in range(merge_rounds):

        selected_merge = None

        while True:
            in_heap_count, pair = heappop(heap).val
            assert in_heap_count > 0, "Bug: pair with zero count found in heap."
            current_count = total_occurrences_for_pair[pair]
            if in_heap_count == current_count:
                selected_merge = pair
                break
            else:
                if current_count == 0:
                    # Pair no longer exists after all the merges,
                    # so it doesn't participate in merging anymore
                    # and we don't return it to the heap.
                    continue
                else:
                    # Pair still exists but count is stale. Return
                    # to the heap with its updated count.
                    heappush(heap, OrderInverterWrapper((current_count, pair)))

        ## Define new merged token
        vocab[len(vocab)] = new_token = selected_merge[0] + selected_merge[1]
        merges.append(selected_merge)

        ## We iterate over all word_ids affected by the merge.
        ### We keep track of the new pairs that need to be added
        ### to the heap at the very end of the merge, so that the
        ### counters are updated.
        new_pairs: set[tuple[bytes, bytes]] = set()

        for word_id in per_word_occurrences_for_pair[selected_merge]:
            word = words[word_id]
            i = 0
            while i < len(word) - 1:
                if (
                    word[i] == selected_merge[0]
                    and word[i + 1] == selected_merge[1]
                ):
                    # Merge.
                    ## Former previous and proximus pairs
                    prev = (word[i - 1], word[i]) if i > 0 else None
                    prox = (word[i + 1], word[i + 2]) if i < len(word) - 2 else None
                    ## Decrease occurrences of prev and prox pairs.
                    for old_pair in (prev, prox):
                        if old_pair is not None:
                            assert total_occurrences_for_pair[old_pair] >= word_count[word_id]
                            total_occurrences_for_pair[old_pair] -= word_count[word_id]
                            assert word_id in per_word_occurrences_for_pair[old_pair]
                            assert per_word_occurrences_for_pair[old_pair][word_id] >= 1
                            per_word_occurrences_for_pair[old_pair][word_id] -= 1
                    ## Remove pair at (i,i+1) and replace it with the new token at i.
                    ## TODO: use a dict instead of a list to avoid linear replacement.
                    total_occurrences_for_pair[selected_merge] -= word_count[word_id]
                    per_word_occurrences_for_pair[selected_merge][word_id] -= 1
                    del word[i + 1]
                    word[i] = new_token
                    ## New pairs.
                    new_prev = (word[i - 1], word[i]) if i > 0 else None
                    new_prox = (word[i], word[i+1]) if i < len(word) - 1 else None
                    ## Increase occurrences of new_prev and new_prox
                    for new_pair in (new_prev, new_prox):
                        if new_pair is not None:
                            new_pairs.add(new_pair)  # Heap update helper.
                            total_occurrences_for_pair[new_pair] += word_count[word_id]
                            per_word_occurrences_for_pair[new_pair][word_id] += 1
                # Update i.
                i += 1
            assert per_word_occurrences_for_pair[selected_merge][word_id] == 0
        del per_word_occurrences_for_pair[selected_merge]
        # This may fail to be zero in the expected case when
        # occurrences of the selected_merge pair overlap.
        assert total_occurrences_for_pair[selected_merge] >= 0
        del total_occurrences_for_pair[selected_merge]

        ## Heap update
        for new_pair in new_pairs:
            heappush(heap, OrderInverterWrapper((total_occurrences_for_pair[new_pair], new_pair)))

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
