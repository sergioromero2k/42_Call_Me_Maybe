#!/usr/bin/env python3

from src.constrained_dec import (
    generate_argument,
    select_function,
)
from src.models import FunctionCallResult


class FunctionCaller:
    def __init__(self, model, mapper, trie, functions):
        self.model = model
        self.mapper = mapper
        self.trie = trie
        self.functions = functions

    def call(self, prompt):
        fn_name = select_function(prompt, self.model, self.trie)
        print(f"Function selected: {fn_name}")

        for function in self.functions:
            if function.name == fn_name:
                selected_function = function
                break

        args = {}
        for param_name, param_type in selected_function.parameters.items():
            value = generate_argument(
                prompt, param_type.type, self.model, self.mapper)
            args[param_name] = value

        return FunctionCallResult(prompt=prompt, fn_name=fn_name, args=args)
