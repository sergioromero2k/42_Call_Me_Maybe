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

            if hasattr(tokenizer, "inverse_vocab"):
                best_token = max(
                    valid_next.keys(),
                    key=lambda t: logits[t] if t < len(
                        logits) else float("-inf")
                )
            else:
                best_token = max(
                    valid_next.keys(),
                    key=lambda t: logits[t] if t < len(
                        logits) else float("-inf")
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
        max_digits = 20  # Límite de seguridad para evitar cuelgues
        digit_count = 0

        # Función auxiliar limpia (DRY) para unificar conversiones y límites
        def _parse_and_validate(val_str: str) -> Any:
            try:
                final_val = float(val_str) if param_type == "number" else int(
                    float(val_str))
                if abs(final_val) > 99999999:
                    return 0
                return final_val
            except ValueError:
                return 0

        while digit_count < max_digits:
            digit_count += 1
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

                should_break_while = False
                token_accepted = False

                for token_id in sorted_token_ids:
                    if hasattr(tok, "decode"):
                        token_str = tok.decode([token_id])
                    else:
                        break

                    # 1. Si el token está vacío, procesamos y retornamos inmediatamente
                    if token_str == "":
                        return _parse_and_validate(argument_progress)

                    # 2. Si contiene texto basura, espacios o caracteres inválidos, forzamos salida total
                    valid_chars = "-0123456789.\n"
                    if any(c not in valid_chars for c in token_str):
                        should_break_while = True
                        break

                    if (argument_progress + token_str).count(".") >= 2:
                        continue
                    if (argument_progress + token_str).count("-") >= 2:
                        continue
                    if (argument_progress + token_str).count("-") == 1 \
                            and (argument_progress + token_str)[0] != "-":
                        continue

                    # Guardamos el carácter válido y marcamos que la iteración fue exitosa
                    argument_progress += token_str
                    token_accepted = True

                    if "\n" in argument_progress:
                        val = argument_progress.split("\n")[0]
                        return _parse_and_validate(val)

                    break  # Salimos del bucle de candidatos para calcular la siguiente posición del while

                # Control de flujo explícito para el bucle while exterior
                if should_break_while:
                    break
                if not token_accepted:
                    break

            except Exception as e:
                print(f"[generate_argument] Error: {e}")
                break

        # Red de seguridad final si se agotan los ciclos de la generación
        return _parse_and_validate(argument_progress)

    elif param_type == "string":
        argument_progress = ""
        STOPS = [
            "\n", "<|im_end|>", "<|im_start|>",
            "regex=", "replacement=", "database=", "encoding=", "query="
        ]
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

                if any(param in token_str for param in ("database=", "encoding=", "replacement=")):
                    break

                argument_progress += token_str

            except Exception as e:
                print(f"[generate_argument] Error: {e}")
                break

        result = argument_progress
        for stop in STOPS:
            result = result.split(stop)[0]

        final_str = result.strip().rstrip(",")

        if param_name in ("query", "template", "path"):
            quotes_found = re.findall(r"['\"]([^'\"]*)['\"]", prompt)
            for q in quotes_found:
                q_clean = q.lower().replace(" ", "")
                final_clean = final_str.lower().replace(" ", "")

                if len(q) > 2 and (final_clean in q_clean or q_clean in final_clean):
                    final_str = q
                    break

            if "{" in prompt and "}" in prompt:
                bracket_match = re.search(
                    r"([a-zA-Z0-9\s\"']*{[^}]+}[a-zA-Z0-9\s\"']*)", prompt)
                if bracket_match and final_str.lower() in bracket_match.group(1).lower():
                    final_str = bracket_match.group(1).strip()

        prompt_words = re.findall(r"[a-zA-Z0-9:\\\/._\-{}]+", prompt)
        for word in prompt_words:
            if word.lower() == final_str.lower():
                final_str = word
                break
            if final_str.lower() in word.lower() and ("config.ini" in word.lower() or "data.json" in word.lower()):
                final_str = word
                break

        if function_def and hasattr(function_def, "parameters") and param_name in function_def.parameters:
            param_meta = function_def.parameters[param_name]
            if isinstance(param_meta, dict) and "enum" in param_meta:
                allowed_enum = param_meta["enum"]
                if final_str not in allowed_enum:
                    for opt in allowed_enum:
                        if opt.lower() in prompt.lower():
                            return opt
                    return allowed_enum[0]
        return final_str
    else:
        return {}
