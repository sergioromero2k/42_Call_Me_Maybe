#!/usr/bin/env python3

import json
from llm_sdk.llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


class VocabularyMapper:
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        route = model.get_path_to_vocab_file()
        with open(route, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.vocab = raw_data
        self.vocab_inverted = {
            valor: clave for clave, valor in raw_data.items()}

    def token_to_str(self, token_id: int) -> str:
        return self.vocab_inverted[token_id]

    def str_to_token(self, text: str) -> int:
        return self.vocab[text]

    def find_tokens_with_prefix(self, prefix) -> list[int]:
        return [
            id for id, data in self.vocab_inverted.items()
            if data.startswith(prefix)
        ]


class FunctionTrie:
    def __init__(self):
        self.root = {"children": {}, "is_end": False, "fn_name": None}

    def insert(self, tokens, fn_name):
        current_node = self.root

        for token in tokens:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                new_node = {"children": {}, "is_end": False, "fn_name": None}
                current_node["children"][token] = new_node
                current_node = new_node

        current_node["is_end"] = True
        current_node["fn_name"] = fn_name

    def get_valid_tokens(self, token_generated):
        current_node = self.root

        for token in token_generated:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                return []
        return list(current_node["children"].keys())

    def is_function_complete(self, tokens):
        current_node = self.root
        for token in tokens:
            if token in current_node["children"]:
                current_node = current_node["children"][token]
            else:
                return False
        return current_node["is_end"]

    def get_fn_name(self, tokens):
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
    trie = FunctionTrie()

    for function in functions:
        tokens = model.encode(function.name).tolist()[0]
        trie.insert(tokens, function.name)
    return trie


def select_function(prompt, model: Small_LLM_Model, trie: FunctionTrie):
    input_ids = model.encode(prompt).tolist()[0]
    tokens_generated = []

    while True:
        logits = model.get_logits_from_input_ids(input_ids)
        tokens_valids = trie.get_valid_tokens(tokens_generated)

        for token_id, value in enumerate(logits):
            if token_id not in tokens_valids:
                logits[token_id] = float("-inf")

        max_token = logits.index(max(logits))
        tokens_generated.append(max_token)
        input_ids.append(max_token)

        if trie.is_function_complete(tokens_generated):
            break
    return trie.get_fn_name(tokens_generated)


def generate_argument(
    prompt: str, param_type: str, model: Small_LLM_Model, mapper:
        VocabularyMapper) -> str:
    if param_type == "boolean":
        valid_tokens = {
            mapper.str_to_token("true"),
            mapper.str_to_token("false")
        }
        inputs_ids = model.encode(prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(inputs_ids)

        masked_logits = [float('-inf')] * len(logits)
        for token_id in valid_tokens:
            masked_logits[token_id] = logits[token_id]

        max_token = masked_logits.index(max(masked_logits)) 
        return mapper.token_to_str(max_token)

    elif param_type == "number":
        valid_tokens = []
        for digit in range(10):
            valid_tokens.extend(mapper.find_tokens_with_prefix(str(digit)))
        valid_tokens.extend(mapper.find_tokens_with_prefix("."))

        input_ids = model.encode(prompt).tolist()[0]
        number_generated = []
        while True:
            logits = model.get_logits_from_input_ids(input_ids)
            masked_logits = [float('-inf')] * len(logits)
            for token_id in valid_tokens:
                masked_logits[token_id] = logits[token_id]
            max_token = masked_logits.index(max(masked_logits)) 
            if max_token not in valid_tokens:
                break
            else:
                input_ids.append(max_token)
                number_generated.append(max_token)
        return float(model.decode(number_generated))

    elif param_type == "string":
        input_ids = model.encode(prompt).tolist()[0]
        string_generated = []
        token_quote = mapper.str_to_token('"')

        while True:
            logits = model.get_logits_from_input_ids(input_ids)
            max_token = logits.index(max(logits))

            if max_token == token_quote:
                break
            else:
                input_ids.append(max_token)
                string_generated.append(max_token)
        return model.decode(string_generated)
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")
