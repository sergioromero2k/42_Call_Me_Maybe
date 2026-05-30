#!/usr/bin/env python3

import re
from typing import Any, List, Dict, Optional
from src.trie import FunctionTrie, TrieNode
from src.models import FunctionDefinition


def build_trie(functions: List[FunctionDefinition], tokenizer: Any) -> FunctionTrie:
    """"""

    trie = FunctionTrie()

    for function in functions:
        token_ids = tokenizer.encode(functions.name)
        trie.insert(token_ids, meta_data={"fn_name": function.name})

    return trie
