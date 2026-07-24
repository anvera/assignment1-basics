#!/usr/bin/env python3

from cs336_basics.train_bpe_multiproc import train_bpe_multiproc
from cs336_basics.train_bpe import train_bpe, fast_train_bpe

import os
from datetime import datetime
import argparse
import json


def train_bpe_on_corpus(
    special_tokens: list[str],
    training_file: str,
    vocabulary_length: int,
    multiproc: bool,
    num_processes: int,
    fast: bool,
):

    if multiproc:
        vocab, merges = train_bpe_multiproc(
            training_file, vocabulary_length, special_tokens, num_processes
        )
    else:
        if fast:
            vocab, merges = fast_train_bpe(
                training_file, vocabulary_length, special_tokens
            )
        else:
            vocab, merges = train_bpe(training_file, vocabulary_length, special_tokens)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_vocab = os.path.join("/tmp", f"vocab_{timestamp}.json")
    filename_merges = os.path.join("/tmp", f"merges_{timestamp}.json")

    # The following conversions deal with the inability to
    # serialize 'bytes' objects in JSON.
    serializable_vocab = { k:str(v) for k,v in vocab.items() }
    serializable_merges = list(map(str,merges))

    # Save
    with open(filename_vocab, "w") as f:
        json.dump(serializable_vocab, f, indent=2, sort_keys=True)
        print(f"Saved file {filename_vocab}")

    with open(filename_merges, "w") as f:
        json.dump(serializable_merges, f, indent=2)
        print(f"Saved file {filename_merges}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Trains BPE tokenizer on a given corpus with '<|endoftext|>' as the only special token."
    )
    special_tokens = ["<|endoftext|>"]
    parser.add_argument("corpus_filename")
    parser.add_argument(
        "-vl",
        "--vocabulary_length",
        type=int,
        required=True,
        help="The vocabulary length, including the 256 basic bytes and the special token.",
    )
    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Enable use of the efficient version of training.",
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

    print(args)

    train_bpe_on_corpus(
        special_tokens,
        args.corpus_filename,
        args.vocabulary_length,
        args.multiproc,
        args.processors,
        args.fast,
    )
