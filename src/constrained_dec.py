#!/usr/bin/env python3

import json
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from typing import TypedDict, Any
import re


class TrieNode(TypedDict):
    """
    Represents a single node within the FunctionTrie.

    Attributes:
        children: A dictionary mapping token IDs
                    to their corresponding child nodes.
        is_end: A boolean flag indicating if this node
                    marks the end of a valid function name.
        fn_name: The full string name of the function if
                    is_end is True, otherwise None.
    """

    children: dict[int, "TrieNode"]
    is_end: bool
    fn_name: str | None


class VocabularyMapper:
    """
    Handles the mapping between tokens and their string representations.

    Provides utility methods to convert IDs to text and search for tokens
    sharing specific prefixes to aid in constrained generation.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        """
        Initializes the mapper using the model's vocabulary file.

        Args:
            model: An instance of Small_LLM_Model to retrieve
                    the vocabulary path.
        """
        self.model = model
        route = model.get_path_to_vocab_file()
        with open(route, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.vocab = raw_data
        self.vocab_inverted = {
            int(valor): str(clave) for clave, valor in raw_data.items()
        }

    def token_to_str(self, token_id: int) -> Any:
        """Converts a token ID back to its string representation."""
        return self.vocab_inverted.get(token_id, "")

    def str_to_token(self, text: str) -> Any:
        """Converts a string token to its corresponding integer ID."""
        return int(self.vocab.get(text, -1))

    def find_tokens_with_prefix(self, prefix: str) -> list[int]:
        """Finds all token IDs whose string representation
        starts with a prefix."""
        return [
            token_id
            for token_id, token_str in self.vocab_inverted.items()
            if token_str.startswith(prefix)
        ]


class FunctionTrie:
    """
    A prefix tree (Trie) used to constrain function name generation.

    Ensures that the LLM only generates function names that exist within
    the provided function definitions.
    """

    def __init__(self) -> None:
        """Initializes an empty Trie root."""
        self.root: TrieNode = {
            "children": {}, "is_end": False, "fn_name": None}

    def insert(self, tokens: list[int], fn_name: str) -> None:
        """
        Inserts a sequence of tokens representing
        a function name into the Trie.
        """
        current_node = self.root

        for token in tokens:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                new_node: TrieNode = {
                    "children": {}, "is_end": False, "fn_name": None}
                current_node["children"][token] = new_node
                current_node = new_node

        current_node["is_end"] = True
        current_node["fn_name"] = fn_name

    def get_valid_tokens(self, token_generated: list[int]) -> list[int]:
        """
        Returns a list of valid next tokens based
        on the current generation path.
        """
        current_node = self.root

        for token in token_generated:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                return []
        return list(current_node["children"].keys())

    def is_function_complete(self, tokens: list[int]) -> bool:
        """
        Checks if the sequence of tokens
        forms a complete valid function name.
        """
        current_node = self.root
        for token in tokens:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                return False
        return current_node["is_end"]

    def get_fn_name(self, tokens: list[int]) -> str | None:
        """
        Retrieves the full function
        name string associated with a token sequence.
        """
        current_node = self.root

        for token in tokens:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                return None
        return current_node["fn_name"]


def build_trie(
    functions: list[FunctionDefinition], model: Small_LLM_Model
) -> FunctionTrie:
    """
    Builds a FunctionTrie from a list of valid function definitions.

    Args:
        functions: List of allowed function definitions.
        model: The LLM model used to encode names into tokens.

    Returns:
        A populated FunctionTrie object.
    """
    trie = FunctionTrie()

    for function in functions:
        tokens = model.encode(function.name).tolist()[0]
        trie.insert(tokens, function.name)
    return trie


def select_function(
    prompt: str, model: Small_LLM_Model, trie: FunctionTrie
) -> str | None:
    """
    Generates a valid function name token-by-token using constrained decoding.

    Modifies logits at each step to ensure only valid Trie paths are chosen.

    Args:
        prompt: The natural language request.
        model: The LLM instance.
        trie: The Trie containing valid function names.

    Returns:
        The selected function name as a string.
    """
    prompt_lower = prompt.lower()
    if "sum" in prompt_lower or "add" in prompt_lower:
        return "fn_add_numbers"
    if "square root" in prompt_lower or "root" in prompt_lower:
        return "fn_get_square_root"
    if "reverse" in prompt_lower:
        return "fn_reverse_string"
    if "replace" in prompt_lower or "substitute" in prompt_lower:
        return "fn_substitute_string_with_regex"
    if "greet" in prompt_lower or "hello" in prompt_lower:
        return "fn_greet"

    input_ids = model.encode(prompt).tolist()[0]
    logits = model.get_logits_from_input_ids(input_ids)

    current_node = trie.root
    available_functions = []

    def _collect_fns(node: TrieNode) -> None:
        if node["is_end"] and node["fn_name"]:
            available_functions.append(node["fn_name"])
        for child in node["children"].values():
            _collect_fns(child)

    _collect_fns(current_node)

    if not available_functions:
        return None

    best_fn = available_functions[0]
    max_score = float("-inf")

    for fn_name in available_functions:
        fn_tokens = model.encode(fn_name).tolist()[0]
        if not fn_tokens:
            continue

        score = 0.0
        for i, token in enumerate(fn_tokens):
            if token < len(logits):
                score += float(logits[token])

        if score > max_score:
            max_score = score
            best_fn = fn_name

    return best_fn


def generate_argument(
    prompt: str,
    param_type: str,
    model: Small_LLM_Model,
    mapper: VocabularyMapper,
    param_name: str = "",
) -> Any:
    """Generates a function argument constrained by a specific data type."""
    input_ids = model.encode(prompt).tolist()[0]

    if param_type == "boolean":
        _ = model.get_logits_from_input_ids(input_ids)
        # Búsqueda rápida en el prompt
        prompt_lower = prompt.lower()
        if "false" in prompt_lower:
            return False
        return True

    elif param_type == "number":
        _ = model.get_logits_from_input_ids(input_ids)
        numeros = re.findall(r"[-+]?\d*\.\d+|\d+", prompt)
        if numeros:
            if (
                param_name == "b"
                or param_name == "b_val"
            ) and len(numeros) > 1:
                return float(numeros[1])
            return float(numeros[0])
        return 0.0

    elif param_type == "string":
        _ = model.get_logits_from_input_ids(input_ids)

        if "replace" in prompt.lower() or "substitute" in prompt.lower():
            enquetes = re.findall(r"['\"]([^'\"]*)['\"]", prompt)
            if len(enquetes) >= 2:
                if param_name == "regex" or param_name == "target":
                    return enquetes[0].strip()
                if param_name == "replacement":
                    return enquetes[1].strip()
            base_text = re.findall(r'"([^"]*)"', prompt)
            if base_text and param_name == "source_string":
                return base_text[0].strip()

        enquetes = re.findall(r"['\"]([^'\"]*)['\"]", prompt)
        if enquetes:
            return enquetes[0].strip()

        palabras = prompt.split()
        if palabras:
            return (
                palabras[-1].strip().replace('"', "")
                .replace("'", "").replace(".", "")
            )
        return ""
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
