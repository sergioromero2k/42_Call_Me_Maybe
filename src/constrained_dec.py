#!/usr/bin/env python3

import json
from llm_sdk.llm_sdk import Small_LLM_Model


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
