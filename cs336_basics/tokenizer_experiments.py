import random
import regex as re
import json
import argparse
import os


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

    args = parser.parse_args()

    if args.command == "sample":
        if args.sample_command == "precompute_boundaries":
            handle_boundary_precomputation(args.corpus)
        elif args.sample_command == "documents":
            handle_document_sampling(
                args.corpus, args.output_filename, args.number, args.seed
            )
    else:
        raise NotImplementedError


if __name__ == "__main__":

    main()
