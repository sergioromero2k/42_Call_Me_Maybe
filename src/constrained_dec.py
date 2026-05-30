#!/usr/bin/env python3

import re
from typing import Any, List, Dict, Optional
from src.trie import FunctionTrie, TrieNode
from src.models import FunctionDefinition


def build_trie(
        functions: List[FunctionDefinition], tokenizer: Any) -> FunctionTrie:
    """Build a FunctionTrie from a list of tokenized function definitions."""

    # If the list is empty, no point iterating — return an empty Trie
    if not functions:
        return FunctionTrie()

    # If there is no tokenizer we cannot process anything — fail fast
    if tokenizer is None:
        raise ValueError("[build_trie] Tokenizer cannot be None.")

    # Create the Trie we will populate and eventually return
    trie = FunctionTrie()

    for function in functions:

        # Skip anything that is not a FunctionDefinition — wrong type
        if not isinstance(function, FunctionDefinition):
            print(f"[build_trie] Invalid element skipped: {function!r}")
            continue

        # Skip functions with no name or a name made of only whitespace
        if not function.name or not function.name.strip():
            print(
                f"[build_trie] Function with empty name skipped: {function!r}")
            continue

        # Try to convert the function name into token ids
        # If the tokenizer fails for any reason, skip this function
        try:
            token_ids = tokenizer.encode(function.name)
        except Exception as e:
            print(f"[build_trie] Failed to tokenize '{function.name}': {e}")
            continue

        # If encoding produced no tokens, there is nothing to insert
        if not token_ids:
            print(
                f"[build_trie] Empty token list for "
                f"'{function.name}', skipping.")
            continue

        # Try to insert the token ids into the Trie
        # If the Trie itself fails for any reason, skip this function
        try:
            trie.insert(token_ids, meta_data={"fn_name": function.name})
        except Exception as e:
            print(
                f"[build_trie] Failed to insert '{function.name}'"
                f" into trie: {e}")
            continue

    # Return whatever the Trie managed to collect
    # could be full, partial, or empty
    return trie
