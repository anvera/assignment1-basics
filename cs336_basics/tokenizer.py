from typing import Self
from collections.abc import Iterator, Iterable
from ast import literal_eval
import json
import regex as re
import functools


class Tokenizer:

    PTK_REGEX = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Self:
        """Class method
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        with open(vocab_filepath) as vf:
            d = json.loads(vf.read())
            vocab = {int(k): literal_eval(d[k])}

        with open(merges_filepath) as mf:
            l = json.loads(mf.read())
            merges = list(tuple(literal_eval(m)) for m in l)

        return Tokenizer(vocab, merges, special_tokens)

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens."""
        self.vocab = vocab
        self.vocab_inv = {v: k for k, v in vocab.items()}
        assert len(self.vocab) == len(self.vocab_inv)
        self.merges = merges
        self.sptk_regex = (
            re.compile("|".join(map(re.escape, special_tokens)))
            if special_tokens
            else None
        )

    def corpus_section_iterable(self, text: str) -> Iterator[str]:

        if self.sptk_regex is None:
            return iter((text,))
        else:
            remainder = text
            while len(remainder) > 0:
                splits = self.sptk_regex.split(text, maxsplit=1)
                if len(splits) == 1:
                    if len(splits[0]) > 0:
                        yield splits[0]
                    return
                else:
                    remainder = splits[1]
                    yield splits[0]
        return

    def pre_token_iterable(self, text: str) -> Iterator[bytes]:
        """Given a text, returns a generator of UTF-8 encoded pretokens. The text
        is splitted by special tokens, then pre-tokenized with a regular expression,
        and encoded in UTF-8 before yielding them."""

        for section in self.corpus_section_iterable(text):
            for match in self.PTK_REGEX.finditer(section):
                yield match.group(0).encode("utf-8")
        return

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""

        return list(self.encode_iterable(iter([text])))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is
        required for memory-efficient tokenization of large files that we cannot directly load into
        memory."""

        for text in iterable:
            ptk_gen = self.pre_token_iterable(text)

            for word_bytes in ptk_gen:
                wip = [bytes([b]) for b in word_bytes]
                for l, r in self.merges:
                    i = 0
                    while i < len(wip) - 1:
                        if wip[i] == l and wip[i + 1] == r:
                            wip[i] = l + r
                            del wip[i + 1]
                        i += 1
                for token in wip:
                    yield self.vocab_inv[token]
        return

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""

        def dec_utf8(b: bytes) -> str:
            return b.decode("utf-8", errors="replace")

        return "".join(map(dec_utf8, map(self.vocab.get, ids)))
