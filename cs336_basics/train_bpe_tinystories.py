from cs336_basics.train_bpe_on_corpus import train_bpe_on_corpus
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="BPE Training on TinyStories")
  parser.add_argument("-f", "--fast", action="store_true", help="Use the efficient version of tokenizer training.")
  args = parser.parse_args()
  print(args)
  train_bpe_on_corpus(["<|endoftext|>"], "../data/TinyStoriesV2-GPT4-train.txt", 10000, False, -1, args.fast)
