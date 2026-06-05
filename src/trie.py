#!/usr/bin/env python3

from typing import Dict, List, Any, Optional


class TrieNode:
    """A single node in the function trie.

    Attributes:
        children: A mapping from token IDs to child TrieNode instances.
        is_end_of_path: True if this node marks the end of a valid
            function name sequence.
        meta: Optional metadata stored at the end of a path, typically
            containing the original function name under 'fn_name'.
    """
    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.is_end_of_path: bool = False
        self.meta: Dict[str, Any] = {}


class FunctionTrie:
    """A token-level prefix trie for indexing function names.

    Stores function names as sequences of token IDs, enabling
    constrained decoding by restricting model output to valid
    function name prefixes at each generation step.

    Attributes:
        root: The root TrieNode, serving as the entry point for
            all insertions and traversals.
    """
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(
            self, token_ids: List[int],
            meta_data: Optional[dict[str, Any]] = None) -> None:
        """Inserts a token ID sequence into the trie.

        Traverses the trie following the given token IDs, creating
        new nodes as needed. Marks the final node as an end of path
        and stores the provided metadata there.

        Args:
            token_ids: A list of integer token IDs representing a
                tokenized function name.
            meta_data: Optional dictionary to store at the end node,
                typically {"fn_name": "function_name"}.

        Raises:
            TypeError: If token_ids is not a list, contains non-integer
                elements, or meta_data is not a dict.
        """
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
        """Returns the valid next token IDs from a given trie node.

        Used during constrained decoding to determine which tokens
        the model is allowed to generate at each step.

        Args:
            current_node: The current TrieNode during trie traversal.

        Returns:
            A list of integer token IDs representing valid next tokens.
            Returns an empty list if the node is invalid or has no children.

        Raises:
            TypeError: If current_node is not a TrieNode instance.
            ValueError: If current_node is None.
        """
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
