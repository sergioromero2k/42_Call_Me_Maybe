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
        raw_data_inv = {valor: clave for clave, valor in raw_data.items()}
        self.vocab_inverted = raw_data_inv

    def taken_to_str(self, token_id: int) -> str:
        return self.vocab_inverted[token_id]

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


def build_trie(
    functions: list[FunctionDefinition], model: Small_LLM_Model
) -> FunctionTrie:
    trie = FunctionTrie()

    for function in functions:
        tokens = model.encode(function.name).tolist()[0]
        trie.insert(tokens, function.name)
    return trie
