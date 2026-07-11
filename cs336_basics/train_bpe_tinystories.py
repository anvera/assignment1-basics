from cs336_basics.train_bpe_on_corpus import train_bpe_on_corpus

if __name__ == '__main__':

  train_bpe_on_corpus(["<|endoftext|>"], "./data/TinyStoriesV2-GPT4-train.txt", 10000, True, 6) 
