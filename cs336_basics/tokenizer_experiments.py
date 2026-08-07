import random
import regex as re
import json
import argparse
import os
import numpy as np
import time
from typing import List, Iterable

from cs336_basics.tokenizer import Tokenizer


def document_boundaries(input_path: str, separator: str) -> list[int]:

    sep_bytes = separator.encode("utf-8")
    BLOCK_SIZE = 4096
    SEP_LEN = len(sep_bytes)
    offset = 0
    boundaries = [offset]
    trail = b""

    with open(input_path, "rb") as f:

        while (block := f.read(BLOCK_SIZE)) != b"":
            chunk = trail + block
            while sep_bytes in chunk:
                idx = chunk.index(sep_bytes)
                delta = idx + len(sep_bytes)
                offset += delta
                boundaries.append(offset)
                chunk = chunk[delta:]
            trail = chunk
        if trail != b"":  # At least TinyStories doesn't end with end-of-text.
            offset += len(trail)
            boundaries.append(offset)

    return boundaries


def boundary_filename_for_corpus(corpus_filename: str):

    return os.path.splitext(corpus_filename)[0] + ".bnd.json"


def handle_boundary_precomputation(corpus_filename: str):

    separator = "<|endoftext|>"
    corpus_boundaries = document_boundaries(corpus_filename, separator)
    with open(
        boundary_filename_for_corpus(corpus_filename), "w", encoding="utf-8"
    ) as f:
        json.dump(corpus_boundaries, f)


def handle_document_sampling(
    corpus_filename: str, output_filename: str, number: int, seed: int | None
):

    with open(
        boundary_filename_for_corpus(corpus_filename), "r", encoding="utf-8"
    ) as f:
        boundaries: List[int] = json.loads(f.read())

    assert (
        len(boundaries) > number
    ), f"Not enough documents. We should have number <= {len(boundaries)-1}."
    rng = random if seed is None else random.Random(seed)
    choice = rng.sample(range(len(boundaries) - 1), k=number)

    sample = []

    with open(corpus_filename, "rb") as f:
        for idx in choice:
            start, end = boundaries[idx], boundaries[idx + 1]
            doc = []
            f.seek(start)
            tranche_size = end - start
            total_read = 0
            while total_read < tranche_size:
                chunk = f.read(tranche_size - total_read)
                total_read += len(chunk)
                doc.append(chunk)
            sample.append(b"".join(doc))

    with open(output_filename, "wb") as f:
        for document in sample:
            f.write(document)

def documents_iter(documents_file: str, separator: str) -> Iterable[str]:

    with open(documents_file, 'r') as f:
        BLOCK_SIZE = 4096
        accumulated = ""
        while (block := f.read(BLOCK_SIZE)) != "":
            while separator in block:
                end_idx = block.index(separator) + len(separator)
                yield accumulated + block[:end_idx]
                accumulated = ""
                block = block[end_idx:]
            accumulated += block
        if accumulated != "":
            yield accumulated
    return


def handle_tokenize(
    vocab_file: str, merges_file: str, documents_file: str, output_file: str
):

    endoftext = "<|endoftext|>"

    tokenizer = Tokenizer.from_files(
        vocab_file, merges_file, special_tokens=[endoftext]
    )

    start_time = time.perf_counter_ns()
    docs_iter = documents_iter(documents_file, separator = endoftext)
    tokens_iter = tokenizer.encode_iterable(docs_iter)
    result = np.fromiter(tokens_iter, dtype=np.uint16)
    total_tokens = len(result)
    with open(output_file, "wb") as out:
        out.write(result.tobytes())
    total_time_sec = (time.perf_counter_ns() - start_time) / float(10**9)
    total_bytes = os.path.getsize(documents_file)

    combined_throughput = total_bytes / total_time_sec
    compression_ratio = total_bytes / total_tokens

    print(f"Total input bytes: {total_bytes}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total time in seconds: {total_time_sec:.3f}")

    print(f"Compression ratio: {compression_ratio:.2f}")
    print(f"End-to-end throughput (bytes/sec): {combined_throughput:,.0f}")


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

    ## Actual random sampling parser
    choose_parser = sample_subparsers.add_parser(
        "documents", help="Uniformly randomly samples documents from a corpus"
    )
    choose_parser.add_argument("corpus", help="Corpus file path")
    choose_parser.add_argument("output_filename", help="Output file path")
    choose_parser.add_argument(
        "--number",
        "-n",
        required=True,
        type=int,
        help="Number of documents in the sample",
    )
    choose_parser.add_argument(
        "--seed", "-s", type=int, help="Seed for the random generator"
    )

    # Tokenize
    tokenize_parser = subparsers.add_parser(
        "tokenize", help="Tokenizes a text with a given tokenizer"
    )
    tokenize_parser.add_argument(
        "vocab_file", help="Tokenizer vocabulary definition file"
    )
    tokenize_parser.add_argument("merges_file", help="Tokenizer merge sequence file")
    tokenize_parser.add_argument(
        "documents_file", help="The file containing the text to tokenize"
    )
    tokenize_parser.add_argument(
        "output_file", help="Output where to serialize the tokenized documents"
    )

    args = parser.parse_args()

    if args.command == "sample":
        if args.sample_command == "precompute_boundaries":
            handle_boundary_precomputation(args.corpus)
        elif args.sample_command == "documents":
            handle_document_sampling(
                args.corpus, args.output_filename, args.number, args.seed
            )
    elif args.command == "tokenize":
        handle_tokenize(
            args.vocab_file, args.merges_file, args.documents_file, args.output_file
        )
    else:
        raise NotImplementedError


if __name__ == "__main__":

    main()
