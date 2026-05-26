#!/usr/bin/env python3

import json
from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition
from typing import TypedDict, Any


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
        tokens = model.encode(function.name)
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
    input_ids = model.encode(prompt)
    tokens_generated: list[int] = []

    for _ in range(50):
        logits = model.get_logits_from_input_ids(input_ids)
        tokens_valids = trie.get_valid_tokens(tokens_generated)

        if not tokens_valids:
            break

        masked_logits = [float("-inf")] * len(logits)
        for token_id in tokens_valids:
            masked_logits[token_id] = logits[token_id]

        max_token = masked_logits.index(max(masked_logits))
        tokens_generated.append(max_token)
        input_ids.append(max_token)

        if trie.is_function_complete(tokens_generated):
            return trie.get_fn_name(tokens_generated)

    return None


def generate_argument(
    prompt: str, param_type: str, model: Small_LLM_Model,
    mapper: VocabularyMapper
) -> str | float:
    """
    Generates a function argument constrained by a specific data type.

    Args:
        prompt: The context prompt for the argument.
        param_type: The required type (boolean, number, string).
        model: The LLM instance.
        mapper: VocabularyMapper to validate allowed tokens.

    Returns:
        The generated argument value in its correct Python type.

    Raises:
        ValueError: If an unsupported parameter type is provided.
    """

    input_ids = model.encode(prompt)

    if param_type == "boolean":
        valid_tokens = [
            mapper.str_to_token("true"), mapper.str_to_token("false")]
        logits = model.get_logits_from_input_ids(input_ids)

        masked_logits = [float("-inf")] * len(logits)
        for token_id in valid_tokens:
            if token_id != -1:
                masked_logits[token_id] = logits[token_id]

        max_token = masked_logits.index(max(masked_logits))
        return mapper.token_to_str(max_token) == "true"

    elif param_type == "number":
        allowed_chars = set("0123456789.-")
        number_generated = []

        for _ in range(20):
            logits = model.get_logits_from_input_ids(input_ids)
            best_token = -1
            best_logit = float('-inf')

            for token_id, logit in enumerate(logits):
                token_str = mapper.token_to_str(token_id)
                if token_str and all(c in allowed_chars for c in token_str):
                    if logit > best_logit:
                        best_logit = logit
                        best_token = token_id

            if best_token == -1 or best_logit == float('-inf'):
                break

            input_ids.append(best_token)
            number_generated.append(best_token)

        try:
            return float(model.decode(number_generated).strip())
        except ValueError:
            return 0.0

    elif param_type == "string":
        string_generated = []
        for _ in range(100):
            logits = model.get_logits_from_input_ids(input_ids)
            max_token = logits.index(max(logits))
            token_str = mapper.token_to_str(max_token)

            if (
                "\n" in token_str
                or '"' in token_str
                or "<|endoftext|>" in token_str
            ):
                break

            input_ids.append(max_token)
            string_generated.append(max_token)

        return model.decode(string_generated).strip()
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
