#!/usr/bin/env python3

import regex as re
from collections import Counter, defaultdict

def train_bpe(
  input_path: str,
  vocab_size: int,
  special_tokens: list[str]
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

  # Zeroth pass: read the input bytes into a working list.
  with open(input_path) as f:
    raw_corpus = f.read()
  
  # Pre-pre tokenization: split by special tokens.
  sptk_regex = "|".join(map(re.escape, special_tokens))
  corpus = re.split(sptk_regex, raw_corpus)

  # Pre-tokenization: 
  PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
  pat_rgx = re.compile(PAT)
  pre_tokens = []

  for word in corpus:
    for match in re.finditer(pat_rgx, word):
      pre_tokens.append(match.group(0).encode("utf-8"))      

  pre_token_counts = Counter(pre_tokens)

  # Training
  vocab : dict[int, bytes] = { i: bytes([i]) for i in range(256) }
  merges: list[tuple[bytes, bytes]] = []
  counts : dict[tuple[bytes, ...], int] = {
    tuple( bytes([b]) for b in pre_token ) : count for pre_token, count in pre_token_counts.items()
  }
  
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

