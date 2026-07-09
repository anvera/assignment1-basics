#!/usr/bin/env python3

import regex as re
from collections import Counter, defaultdict
import gc
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool
from functools import partial


def pretokenization_and_counting(endpoints: tuple[int], input_path: str, special_tokens: list[str]):

  begin, end = endpoints[0], endpoints[1]

  with open(input_path, "rb") as f:
    f.seek(begin)
    raw_corpus = f.read(end-begin).decode("utf-8", errors="ignore")

  # Pre-pre tokenization: split by special tokens.
  sptk_regex = "|".join(map(re.escape, special_tokens))
  corpus = re.split(sptk_regex, raw_corpus)

  # Pre-tokenization: 
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  pat_rgx = re.compile(PAT)
  return Counter( match.group(0).encode("utf-8") for word in corpus for match in re.finditer(pat_rgx, word) )


def train_bpe_multiproc(
  input_path: str,
  vocab_size: int,
  special_tokens: list[str],
  num_processes: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
  """Trains a byte-pair encoding tokenizer on the
     file at the input path, with maximum given
     vocabulary size, and with the given list of
     special tokens. Outputs a tuple with an index
     of the determined vocabulary,  as well as the
     list of BPE merges, the latter sorted by order
     of creation."""

  merge_rounds = vocab_size - len(special_tokens) - 256
  assert merge_rounds >= 0, "Vocabulary size too small."
  assert num_processes >= 1

  with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
  
  process_inputs = list( (start, end) for start,end in zip(boundaries[:-1], boundaries[1:]))

  with Pool(processes=num_processes) as pool:
    pre_token_counter_list = pool.map(partial(pretokenization_and_counting, input_path=input_path, special_tokens=special_tokens), process_inputs, 1)

  pre_token_counts = Counter()
  for counter in pre_token_counter_list:
    pre_token_counts += counter


  # Training
  vocab : dict[int, bytes] = { i: bytes([i]) for i in range(256) }
  merges: list[tuple[bytes, bytes]] = []
  counts : dict[tuple[bytes, ...], int] = {
    tuple( bytes([b]) for b in pre_token ) : count for pre_token, count in pre_token_counts.items()
  }
  del pre_token_counts
  gc.collect()


  for _ in range(merge_rounds):

    ## Compute the most common pair.
    merge_ranks = defaultdict(int)
    for bseq, count in counts.items():
      for i in range(len(bseq)-1):
        merge_ranks[(bseq[i], bseq[i+1])] += count
    
    if len(merge_ranks) == 0:
      break

    selected_merge = max(merge_ranks, key= lambda k: (merge_ranks[k], k))
    merges.append(selected_merge)
    
    ## Execute merge.
    vocab[len(vocab)] = selected_merge[0] + selected_merge[1]
    new_counts = defaultdict(int)
    for bseq, count in counts.items():
      merged_seq = []
      i = 0
      while i < len(bseq) - 1:
        if bseq[i] == selected_merge[0] and bseq[i+1] == selected_merge[1]:
          merged_seq.append(bseq[i] + bseq[i+1])
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

