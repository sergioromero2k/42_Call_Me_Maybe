#!/usr/bin/env python3

from typing import Dict, List, Any


class TrieNode:
    def __init__(self):
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end_of_path: bool = False
        self.meta: Dict[str, Any] = {}


class FunctionTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(
            self,
            token_ids: List[int], meta_data: Dict[str, Any] = None) -> None:
        """Inserts a sequence of token IDs into the trie."""
        try:
            if not isinstance(token_ids, list):
                raise TypeError(
                    f"token_ids must be a list, got: "
                    f"{type(token_ids).__name__}"
                )

            if not token_ids:
                return  # empty list, exit silently

            for i, token_id in enumerate(token_ids):
                # bool is a subclass of int in Python, must check explicitly
                if isinstance(token_id, bool) or not isinstance(token_id, int):
                    raise TypeError(
                        f"token_ids[{i}] must be int, "
                        f"got {type(token_id).__name__} ({repr(token_id)})"
                    )

            if meta_data is not None and not isinstance(meta_data, dict):
                raise TypeError(
                    f"meta_data must be a dict or None, "
                    f"got: {type(meta_data).__name__}"
                )

            current = self.root
            for token_id in token_ids:
                if token_id not in current.children:
                    current.children[token_id] = TrieNode()
                current = current.children[token_id]

            current.is_end_of_path = True
            if meta_data:
                current.meta = meta_data

        except TypeError as e:
            print(f"[FunctionTrie.insert] Type error: {e}")
        except MemoryError:
            print(
                "[FunctionTrie.insert] Out of memory while inserting sequence")
        except Exception as e:
            print(f"[FunctionTrie.insert] Unexpected error: {e}")

    def get_valid_next_tokens(self, current_node: TrieNode) -> List[int]:
        """Returns the list of valid next token IDs from the current node."""
        try:
            if current_node is None:
                raise ValueError("current_node cannot be None")
            if not isinstance(current_node, TrieNode):
                raise TypeError(
                    f"current_node must be a TrieNode, "
                    f"got: {type(current_node).__name__}"
                )

            return list(current_node.children.keys())

        except (TypeError, ValueError) as e:
            print(f"[FunctionTrie.get_valid_next_tokens] Error: {e}")
            return []
        except Exception as e:
            print(
                f"[FunctionTrie.get_valid_next_tokens] Unexpected error: {e}")
            return []
