"""
Minimal (byte-level) Byte Pair Encoding tokenizer.

Algorithmically follows along the GPT tokenizer:
https://github.com/openai/gpt-2/blob/master/src/encoder.py

Unlike BasicTokenizer:
- RegexTokenizer handles an optional regex splitting pattern.
- RegexTokenizer handles optional special tokens.
"""

import regex as re
from collections import Counter, defaultdict
from .base import Tokenizer, get_stats, merge, fast_merge_inplace


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        """
        A number of optimizations are introduced:
        - delete function call overhead by inlining functions
        - modifying list of ids in place with .pop() instead of creating a new list
        - collapse identical chunks to just the unique ones
        - update counts more cleverly - only around the affected chunks
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # many, many chunks are identical, so we can "collapse" them to just the unique ones
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # Initial count: build stats and position tracking
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> set of chunk indices that contain this pair

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        for i in range(num_merges):
            if not stats:
                break

            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i

            # Get chunks that contain this pair
            affected_chunks = positions[pair]

            # Track count changes for incremental update
            count_changes = defaultdict(int)

            # Replace all occurrences of pair in affected chunks only
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # Track what pairs are being removed/added
                        # Remove: (prev, A), (A, B), (B, next)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # The merged pair disappears
                        count_changes[pair] -= chunk_count

                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # Apply the merge
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)

                        # Add: (prev, C), (C, next)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # Apply incremental changes to stats and positions
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # The merged pair should disappear completely
                    continue

                stats[changed_pair] += delta

                # Update positions for changed pairs - only check affected chunks
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # Remove the merged pair completely
            del stats[pair]
            del positions[pair]

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids
