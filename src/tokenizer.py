#!/usr/bin/env python3

import json
import os
from typing import Dict, List


class CustomTokenizer:
    """A lightweight tokenizer built from a vocabulary JSON file.

    Loads a word-to-ID mapping from disk and provides encode and decode
    methods compatible with the constrained decoding engine. Used as a
    fallback when the model's internal tokenizer is not available.

    Attributes:
        vocab: A dictionary mapping token strings to integer IDs.
        inverse_vocab: A dictionary mapping integer IDs back to token
            strings, built automatically from vocab for O(1) decoding.
    """
    def __init__(self, vocab_path: str) -> None:
        """Initializes the tokenizer by loading the vocabulary from a file.

        Args:
            vocab_path: Path to a JSON file containing a word-to-ID mapping.

        Raises:
            TypeError: If vocab_path is not a string.
            FileNotFoundError: If the vocabulary file does not exist.
            Exception: If the file cannot be parsed or loaded.
        """
        if not isinstance(vocab_path, str):
            raise TypeError(
                f"vocab_path must be a string, got: "
                f"{type(vocab_path).__name__}"
            )
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(
                f"Vocabulary file not found at: {vocab_path}")

        try:
            # 1. Load the word-to-ID map from the model vocab file
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.vocab: Dict[str, int] = json.load(f)

            # 2. Build the inverse map (ID -> word) in O(1) for decode
            self.inverse_vocab: Dict[int, str] = {
                int(v): k for k, v in self.vocab.items()
            }

        except Exception as e:
            print(f"[CustomTokenizer.__init__] Failed to load vocabulary: {e}")
            raise

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs.

        Uses a greedy longest-match strategy, scanning left to right and
        always matching the longest possible substring found in the
        vocabulary. Falls back to single character lookup if no match
        is found, skipping unknown characters to avoid infinite loops.

        Args:
            text: The input string to tokenize.

        Returns:
            A list of integer token IDs. Returns an empty list if the
            input is empty or an error occurs.

        Raises:
            TypeError: If text is not a string.
            ValueError: If text is None.
        """
        try:
            if text is None:
                raise ValueError("Text to encode cannot be None")
            if not isinstance(text, str):
                raise TypeError(
                    f"Expected a string to encode, got: {type(text).__name__}"
                )

            if not text:
                return []

            token_ids = []
            start = 0

            while start < len(text):
                match_found = False

                for end in range(len(text), start, -1):
                    substring = text[start:end]

                    if substring in self.vocab:
                        token_ids.append(int(self.vocab[substring]))
                        start = end
                        match_found = True
                        break

                # Fallback: if no match was found for any substring
                # starting at 'start'.
                # Skip this character to avoid an infinite loop
                if not match_found:
                    char = text[start]
                    if char in self.vocab:
                        token_ids.append(int(self.vocab[char]))
                    start += 1

            return token_ids

        except (TypeError, ValueError) as e:
            print(f"[CustomTokenizer.encode] Validation error: {e}")
            return []
        except Exception as e:
            print(f"[CustomTokenizer.encode] Unexpected error: {e}")
            return []

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a list of token IDs back into a human-readable string.

        Looks up each token ID in the inverse vocabulary and joins the
        results. Invalid or unknown token IDs are skipped with a warning.

        Args:
            token_ids: A list of integer token IDs to decode.

        Returns:
            The reconstructed string. Returns an empty string if the
            input is invalid or an error occurs.

        Raises:
            TypeError: If token_ids is not a list.
            ValueError: If token_ids is None.
        """
        try:
            if token_ids is None:
                raise ValueError("token_ids list cannot be None")
            if not isinstance(token_ids, list):
                raise TypeError(
                    f"token_ids must be a list, got: "
                    f"{type(token_ids).__name__}"
                )

            # Reconstruct the string by looking up each ID in O(1) via inverse
            decoded_parts = []
            for i, token_id in enumerate(token_ids):
                # bool is a subclass of int in Python, must reject explicitly
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    print(
                        f"[CustomTokenizer.decode] Warning: invalid element"
                        f" at index {i}, skipping."
                    )
                    continue

                decoded_parts.append(self.inverse_vocab.get(token_id, ""))

            return "".join(decoded_parts)

        except (TypeError, ValueError) as e:
            print(f"[CustomTokenizer.decode] Validation error: {e}")
            return ""
        except Exception as e:
            print(f"[CustomTokenizer.decode] Unexpected error: {e}")
            return ""
