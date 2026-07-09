#!/usr/bin/env python3

from cs336_basics.train_bpe_multiproc import train_bpe_multiproc
from cs336_basics.train_bpe import train_bpe

import pickle
import os
from datetime import datetime
import sys

def train_bpe_on_corpus(
  special_tokens: list[str],
  training_file: str,
  vocabulary_length: int,
  multiproc: bool,
  num_processes: int
  ):

  if multiproc:
    assert len(sys.argv) == 5
    num_processes = int(sys.argv[4])
    vocab, merges = train_bpe_multiproc(
      training_file,
      vocabulary_length,
      special_tokens,
      num_processes
    )
  else:
    vocab, merges = train_bpe(
      training_file,
      vocabulary_length,
      special_tokens
    )

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename_vocab = os.path.join("/tmp", f"vocab_{timestamp}.pkl")
  filename_merges = os.path.join("/tmp", f"merges_{timestamp}.pkl")

  # Save
  with open(filename_vocab, "wb") as f:
    pickle.dump(vocab, f)
    print(f"Saved file {filename_vocab}")

  with open(filename_merges, "wb") as f:
    pickle.dump(merges, f)
    print(f"Saved file {filename_merges}")
 


if __name__ == '__main__':

  assert len(sys.argv) >= 4

  special_tokens = ["<|endoftext|>"]
  training_file = sys.argv[1]
  vocabulary_length = int(sys.argv[2])
  multiproc = True if sys.argv[3].lower() == 'true' else False
  num_processes = int(sys.argv[4]) if multiproc else 0
  
  train_bpe_on_corpus(special_tokens, training_file, vocabulary_length, multiproc, num_processes) 
