#!/usr/bin/env python3

import math
import re
from typing import Any, List, Optional
from src.trie import FunctionTrie
from src.models import FunctionDefinition


def build_trie(
        functions: List[FunctionDefinition], tokenizer: Any) -> FunctionTrie:

    if not functions:
        return FunctionTrie()

    if tokenizer is None:
        raise ValueError("[build_trie] Tokenizer cannot be None.")

    trie = FunctionTrie()

    for function in functions:

        if not isinstance(function, FunctionDefinition):
            print(f"[build_trie] Invalid element skipped: {function!r}")
            continue

        if not function.name or not function.name.strip():
            print(
                f"[build_trie] Function with empty name skipped: {function!r}")
            continue

        try:
            token_ids = tokenizer.encode(function.name)
        except Exception as e:
            print(f"[build_trie] Failed to tokenize '{function.name}': {e}")
            continue

        if not token_ids:
            print(
                f"[build_trie] Empty token list for "
                f"'{function.name}', skipping.")
            continue

        try:
            trie.insert(token_ids, meta_data={"fn_name": function.name})
        except Exception as e:
            print(
                f"[build_trie] Failed to insert '{function.name}'"
                f" into trie: {e}")
            continue

    return trie


def select_function(
        prompt: str, model: Any,
        tokenizer: Any, trie: FunctionTrie,
        functions: List = None,
        inference_tokenizer: Any = None) -> Optional[str]:

    if not prompt or not prompt.strip():
        return None
    if trie is None or trie.root is None or tokenizer is None:
        return None

    tok = inference_tokenizer if inference_tokenizer is not None else tokenizer

    # Construir lista de funciones disponibles con descripciones
    available_functions = []
    fn_descriptions = {}
    if functions:
        for fn in functions:
            if hasattr(fn, "name") and hasattr(fn, "description"):
                available_functions.append(fn.name)
                fn_descriptions[fn.name] = fn.description

    if not available_functions:
        return None

    fn_list = "\n".join(
        f"- {name}: {fn_descriptions[name]}"
        for name in available_functions
    )

    prompt_message = (
        f"Here are the available functions:\n{fn_list}\n\n"
        f"Which function name best matches this request: \"{prompt}\"?\n"
        f"Reply with only the function name."
    )

    # Prompt de chat Qwen — igual que tu amigo
    chat_prompt = (
        f"<|im_start|>user\n{prompt_message}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )

    try:
        if hasattr(tok, "encode") and \
                "add_special_tokens" in tok.encode.__code__.co_varnames:
            current_ids = tok.encode(chat_prompt, add_special_tokens=False)
        else:
            current_ids = tok.encode(chat_prompt)
        if not current_ids:
            return None
    except Exception as e:
        print(f"[select_function] Failed to encode prompt: {e}")
        return None

    # Navegar el trie token a token igual que tu amigo filtra por startswith
    current_node = trie.root

    while current_node and not current_node.is_end_of_path:
        valid_next = current_node.children
        if not valid_next:
            break

        try:
            if hasattr(model, "get_logits_from_input_ids"):
                logits = model.get_logits_from_input_ids(list(current_ids))
            elif hasattr(model, "get_logits"):
                logits = model.get_logits(list(current_ids))
            elif callable(model):
                logits = model(list(current_ids))
            else:
                return None

            if hasattr(logits, "tolist"):
                logits = logits.tolist()

            # Elegir el token válido con mayor logit — igual que su sorted_tokens
            best_token = max(
                valid_next.keys(),
                key=lambda t: logits[t] if t < len(logits) else float("-inf")
            )

            current_ids.append(best_token)
            current_node = current_node.children[best_token]

        except Exception as e:
            print(f"[select_function] Error during trie traversal: {e}")
            break

    if current_node and current_node.is_end_of_path and "fn_name" in current_node.meta:
        return current_node.meta["fn_name"]

    return None


def generate_argument(
        prompt: str,
        param_type: str,
        model: Any,
        tokenizer: Any,
        param_name: str = "",
        inference_tokenizer: Any = None,
        function_def: Any = None,
        previous_gen: str = ""
) -> Any:
    if not param_type:
        return ""

    if not prompt or not prompt.strip():
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

    tok = inference_tokenizer if inference_tokenizer is not None else tokenizer

    # Igual que tu amigo — acumula previous_gen + param_name=
    prev = previous_gen + f"{param_name}="

    prompt_message = (
        f"To solve the prompt: {prompt}\n"
        f"Function to use: {function_def}\n"
        f"Provide ONLY the parameter values, no function names.\n"
        f"Keep it concise and don't add custom fields."
    )

    full_prompt = (
        f"<|im_start|>user\n{prompt_message}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{prev}"
    )

    if param_type == "boolean":
        prompt_lower = prompt.lower()
        return False if "false" in prompt_lower else True

    elif param_type in ("number", "integer"):
        argument_progress = ""
        while True:
            try:
                if hasattr(tok, "encode") and \
                        "add_special_tokens" in tok.encode.__code__.co_varnames:
                    input_ids = tok.encode(
                        full_prompt + argument_progress,
                        add_special_tokens=False)
                else:
                    input_ids = tok.encode(full_prompt + argument_progress)

                if hasattr(model, "get_logits_from_input_ids"):
                    logits = model.get_logits_from_input_ids(input_ids)
                elif hasattr(model, "get_logits"):
                    logits = model.get_logits(input_ids)
                else:
                    break

                if hasattr(logits, "tolist"):
                    logits = logits.tolist()

                sorted_token_ids = sorted(
                    range(len(logits)),
                    key=lambda i: logits[i],
                    reverse=True
                )

                for token_id in sorted_token_ids:
                    if hasattr(tok, "decode"):
                        token_str = tok.decode([token_id])
                    else:
                        break

                    if token_str == "":
                        try:
                            return float(argument_progress) \
                                if param_type == "number" \
                                else int(float(argument_progress))
                        except ValueError:
                            argument_progress = ""
                            break

                    valid_chars = "-0123456789.\n"
                    if any(c not in valid_chars for c in token_str):
                        continue
                    if (argument_progress + token_str).count(".") >= 2:
                        continue
                    if (argument_progress + token_str).count("-") >= 2:
                        continue
                    if (argument_progress + token_str).count("-") == 1 \
                            and (argument_progress + token_str)[0] != "-":
                        continue

                    argument_progress += token_str

                    if "\n" in argument_progress:
                        val = argument_progress.split("\n")[0]
                        try:
                            return float(val) if param_type == "number" \
                                else int(float(val))
                        except ValueError:
                            argument_progress = ""
                    break

            except Exception as e:
                print(f"[generate_argument] Error: {e}")
                break

        try:
            return float(argument_progress) if param_type == "number" \
                else int(float(argument_progress))
        except ValueError:
            return 0

    elif param_type == "string":
        argument_progress = ""
        STOPS = ["\n", "<|im_end|>", "<|im_start|>", "regex=", "replacement=", ", "]
        while not any(s in argument_progress for s in STOPS):
            try:
                if hasattr(tok, "encode") and \
                        "add_special_tokens" in tok.encode.__code__.co_varnames:
                    input_ids = tok.encode(
                        full_prompt + argument_progress,
                        add_special_tokens=False)
                else:
                    input_ids = tok.encode(
                        full_prompt + argument_progress)

                if hasattr(model, "get_logits_from_input_ids"):
                    logits = model.get_logits_from_input_ids(input_ids)
                elif hasattr(model, "get_logits"):
                    logits = model.get_logits(input_ids)
                else:
                    break

                if hasattr(logits, "tolist"):
                    logits = logits.tolist()

                best_token_id = logits.index(max(logits))

                if hasattr(tok, "decode"):
                    token_str = tok.decode([best_token_id])
                else:
                    break

                if token_str == "":
                    break

                argument_progress += token_str

            except Exception as e:
                print(f"[generate_argument] Error: {e}")
                break

        result = argument_progress
        for stop in STOPS:
            result = result.split(stop)[0]
        return result.strip().rstrip(",")

    else:
        return {}
