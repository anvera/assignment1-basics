#!/usr/bin/env python3

from cs336_basics.train_bpe_multiproc import train_bpe

import pickle
import os
from datetime import datetime

if __name__ == '__main__':

  training_file = "./data/TinyStoriesV2-GPT4-train.txt" 

  vocab, merges = train_bpe(
    training_file,
    10000,
    ["<|endoftext|>"]
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

