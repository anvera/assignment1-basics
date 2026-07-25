from cs336_basics.train_bpe_on_corpus import train_bpe_on_corpus
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE Training on TinyStories")
    parser.add_argument(
        "-nv",
        "--naive",
        action="store_true",
        help="Uses the naive and inefficient but more straightforward version of training.",
    )
    parser.add_argument(
        "-mp",
        "--multiproc",
        action="store_true",
        help="Enable parallel processing from raw corpus to pre-tokens occurrence counting.",
    )
    parser.add_argument(
        "-p",
        "--processors",
        type=int,
        default=4,
        help="Number of processors used when multiproc is enabled.",
    )
    args = parser.parse_args()
    train_bpe_on_corpus(
        ["<|endoftext|>"],
        "../data/TinyStoriesV2-GPT4-train.txt",
        10000,
        args.multiproc,
        args.processors,
        args.naive,
    )
