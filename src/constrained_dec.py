#!/usr/bin/env python3

from typing import Any, List, Optional
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


def select_function(
    prompt: str, model: Any, tokenizer: Any, trie: FunctionTrie
) -> Optional[str]:
    """
    Evalutes the user prompt by obtaining initial logits from the model
    and mathematically calculates which function has the
    highest probability score.
    """

    if not prompt or not prompt.strip():
        return None

    if trie is None or trie.root is None:
        return None

    if tokenizer is None:
        return None

    try:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            return None
    except Exception as e:
        print(f"[select_function] Failed to encode prompt: {e}")
        return None

    try:
        if hasattr(model, "get_logits_from_input_ids"):
            logits = model.get_logits_from_input_ids(input_ids)
        elif hasattr(model, "get_logits"):
            logits = model.get_logits(input_ids)
        elif callable(model):
            logits = model(input_ids)
        elif hasattr(model, "predict"):
            logits = model.predict(input_ids)
        else:
            print(
                "[select_function] Model does not support "
                "any known logit extraction method."
            )
            return None
    except Exception as e:
        print(f"[select_function] Failed to get logits from model: {e}")
        return None

    if not logits:
        return None

    if hasattr(logits, "tolist"):
        logits = logits.tolist()

    available_functions = []

    def _collect_fns(node: TrieNode):
        if node.is_end_of_path and "fn_name" in node.meta:
            available_functions.append(node.meta["fn_name"])

        for child_node in node.children.values():
            _collect_fns(child_node)

    try:
        _collect_fns(trie.root)
    except RecursionError:
        print("[select_function] Trie has a cycle, recursion limit reached.")
        return None
    except Exception as e:
        print(f"[select_function] Failed to collect functions from trie: {e}")
        return None

    if not available_functions:
        return None

    best_fn = None
    max_score = float("-inf")

    for fn_name in available_functions:
        try:
            fn_tokens = tokenizer.encode(fn_name)
            if not fn_tokens:
                continue
        except Exception as e:
            print(
                "[select_function] Failed to encode "
                f"function name '{fn_name}': {e}")
            continue

        try:
            score = sum(
                float(logits[token]) for token in fn_tokens
                if token < len(logits)
            )
        except Exception as e:
            print(
                f"[select_function] Failed to score function '{fn_name}': {e}")
            continue

        if score > max_score:
            max_score = score
            best_fn = fn_name

    return best_fn


def generate_argument(
        prompt: str,
        param_type: str,
        model: Any,
        tokenizer: Any,
        param_name: str = ""
) -> Any:
    if not param_type:
        return ""

    if not prompt or not prompt.strip():
        if param_type in ("number", "integer"):
            return 0
        if param_type == "boolean":
            return True
        return ""

    if tokenizer is None:
        if param_type in ("number", "integer"):
            return 0
        if param_type == "boolean":
            return True
        return ""

    if model is None:
        if param_type in ("number", "integer"):
            return 0
        if param_type == "boolean":
            return True
        return ""

    try:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("Empty input_ids after encoding prompt.")

        if hasattr(model, "get_logits_from_input_ids"):
            _ = model.get_logits_from_input_ids(input_ids)
        elif hasattr(model, "get_logits"):
            _ = model.get_logits(input_ids)
        elif callable(model):
            _ = model(input_ids)
        elif hasattr(model, "predict"):
            _ = model.predict(input_ids)
        else:
            print("[generate_argument] Model does not support "
                  "any known logit extraction method.")
    except Exception as e:
        print(f"[generated_argument] Warning during model inference: {e}")

    try:
        prompt_lower = prompt.lower()
    except Exception as e:
        print(f"[generate_argument] Failed to lowercase prompt: {e}")
        prompt_lower = ""

    if param_type == "boolean":
        try:
            if "false" in prompt_lower:
                return False
            return True
        except Exception as e:
            print(f"[generate_argument] Failed to parse boolean: {e}")
            return True

    elif param_type in ("number", "integer"):
        import re
        try:
            nums = re.findall(r"[-+]?\d+\.\d+|[-+]?\d+", prompt)
            if not nums:
                return 0

            if (
                param_name in ("b", "b_val", "replacement", "target")
                and len(nums) > 1
            ):
                val_str = nums[1]
            else:
                val_str = nums[0]

            if param_type == "integer":
                return int(float(val_str))
            return float(val_str)

        except ValueError as e:
            print(
                "[generate_argument] Failed to convert "
                f"'{val_str}' to number: {e}")
            return 0
        except Exception as e:
            print(f"[generate_argument] Unexpected error parsing number: {e}")
            return 0

    elif param_type == "string":
        import re
        try:
            quotes = re.findall(r"['\"]([^'\"]*)['\"]", prompt)
            if quotes:
                if param_name in ("replacement", "target") and len(quotes) > 1:
                    return quotes[1].strip()
                return quotes[0].strip()

            # Fallback — Last word clean of the prompt.
            words = prompt.split()
            if not words:
                return ""
            return (
                words[-1]
                .strip()
                .rstrip(".")
                .replace("'", "")
                .replace("?", "")
            )

        except Exception as e:
            print(f"[generate_argument] Failed to parse string: {e}")
            return ""

    else:
        try:
            return {}
        except Exception as e:
            print(
                "[generate_argument] Failed to return default "
                f"for unknown type: {e}")
            return {}
