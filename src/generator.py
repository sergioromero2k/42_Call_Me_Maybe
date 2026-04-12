#!/usr/bin/env python3

from src.constrained_dec import (
    generate_argument,
    select_function,
)
from src.models import FunctionCallResult


class FunctionCaller:
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
    def __init__(self, model, mapper, trie, functions):
        """
        Initializes the generator with necessary LLM and decoding components.

        Args:
            model: The LLM model instance.
            mapper: Utility to map between tokens and strings.
            trie: Prefix tree containing valid function names.
            definitions: List of available function schemas.
        """
        self.model = model
        self.mapper = mapper
        self.trie = trie
        self.functions = functions

    def call(self, prompt):
        """
        Processes a prompt to return a structured function call.

        First selects the function name and then iteratively generates each
        argument based on the identified function's schema.

        Args:
            prompt: The user's natural language request.

        Returns:
            A FunctionResult object containing the prompt,
            function name, and args.
        """
        # Step 1: Identify the function to call using constrained decoding
        fn_name = select_function(prompt, self.model, self.trie)
        print(f"Function selected: {fn_name}")

        for function in self.functions:
            if function.name == fn_name:
                selected_function = function
                break

        args = {}
        # Step 2: Generate each argument according to its defined type
        for param_name, param_type in selected_function.parameters.items():
            value = generate_argument(
                prompt, param_type.type, self.model, self.mapper)
            args[param_name] = value

        return FunctionCallResult(prompt=prompt, fn_name=fn_name, args=args)
