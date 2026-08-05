import random
import regex as re
import json
import argparse
import os


def document_boundaries(input_path: str, separator: str) -> list[int]:

    BLOCK_SIZE = 4096
    SEP_LEN = len(separator)
    offset = 0
    boundaries = [offset]
    trail = ""

    with open(input_path, "r", encoding="utf-8") as f:

        while (block := f.read(BLOCK_SIZE)) != "":
            chunk = trail + block
            while separator in chunk:
                idx = chunk.index(separator)
                delta = idx + len(separator)
                offset += delta
                boundaries.append(offset)
                chunk = chunk[delta:]
            trail = chunk
        if trail != "":  # At least TinyStories doesn't end with end-of-text.
            offset += len(trail)
            boundaries.append(offset)

    return boundaries


def main():

    parser = argparse.ArgumentParser(description="Tokenizer Experiments app.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Corpus Document Sampling
    sample_parser = subparsers.add_parser(
        "sample", help="Produces samples of documents from a corpus"
    )
    sample_subparsers = sample_parser.add_subparsers(
        dest="sample_command", required=True
    )

    ## Compute boundaries
    boundaries_parser = sample_subparsers.add_parser(
        "precompute_boundaries", help="Precompute corpus documents boundaries"
    )
    boundaries_parser.add_argument("corpus", help="Corpus file path.")
    boundaries_parser.add_argument(
        "--separator", "-s", help="Document separator, defaults to <|endoftext|>"
    )

    # Tokenize
    tokenize_parser = subparsers.add_parser(
        "tokenize", help="Tokenizes a text with a given tokenizer"
    )

    args = parser.parse_args()

    if args.command == "sample" and args.sample_command == "precompute_boundaries":
        separator = args.separator if args.separator else "<|endoftext|>"
        corpus_boundaries = document_boundaries(args.corpus, separator)
        boundaries_filename = os.path.splitext(args.corpus)[0] + ".bnd.json"
        with open(boundaries_filename, "w", encoding="utf-8") as f:
            json.dump(corpus_boundaries, f)
    else:
        raise NotImplementedError


if __name__ == "__main__":

    main()
