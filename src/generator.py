#!/usr/bin/env python3
import torch
from enum import Enum, auto
from llm_sdk.llm_sdk import Small_LLM_Model


class JSONState(Enum):
    EXPECT_OPEN_BRACE = auto()
    EXPECT_FUNCTION_KEY = auto()
    EXPECT_COLON = auto()
    EXPECT_FUNCTION_NAME = auto()
    END = auto()
    EXPECT_ARGUMENT_KEY = auto()
    EXPECT_ARG_A = auto()
    EXPECT_ARG_B_KEY = auto()
    EXPECT_ARG_B = auto()
    EXPECT_CLOSE_BRACE = auto()


class JSONStateMachine:
    def __init__(self, model: Small_LLM_Model, allowed_functions: list[str]):
        """
        Initializa the State Machine for constrained JSON generation.

        @param model The LLM instance used for encoding and logit generation.
        @param allowed_functions: List of valid function names to be enforced.
        """
        self.model = model
        self.allowed_functions = allowed_functions
        self.state = JSONState.EXPECT_OPEN_BRACE
        self.digits_ids = [i for i in range(JSONState.__len__)]

        # Pre-tokens (get specific IDs)
        self.token_lbrace = self.model.encode("{").tolist()[0][-1]
        self.token_function_key = self.model.encode('"function"').tolist()[0]
        self.token_colon = self.model.encode(":").tolist()[0][-1]
        self.token_quote = self.model.encode('"').tolist()[0][-1]
        for i in range(JSONState.__len__):
            self.ids = self.model.encode(f"{i}")

    def mask_invalid_tokens(
        self, logits: torch.Tensor, valid_token_ids: list[int]
    ) -> torch.Tensor:
        """
        Allow only specific tokens by making all others with -inf.

        @param logits Raw output scores from the model for each token.
        @param valid_token_ids List of token IDs allowed at this step.

        @return Masked logits where only valid tokens can be selected.
        """
        mask = torch.full_like(logits, float("-inf"))
        mask[valid_token_ids] = 0
        return logits + mask

    def generate(self, prompt: str, max_tokens: int = 50) -> str:
        """
        Execute the autoregressive loop to gnerate a structured JSON.

        @param prompt User input string to guide the model's logic.
        @param max_tokens: Maximum length of the generated sequence.
        @return A decoded string containing the valid JSON.
        """
        input_ids = self.model.encode(prompt).tolist()[0]
        generated_tokens = []

        for _ in range(max_tokens):
            logits_list = self.model.get_logits_from_input_ids(input_ids)
            logits_tensor = torch.tensor(logits_list)

            valid_ids = []

            if self.state == JSONState.EXPECT_OPEN_BRACE:
                valid_ids = [self.token_lbrace]
            elif self.state == JSONState.EXPECT_FUNCTION_KEY:
                key_tokens = self.model.encode('"function: ').tolist()[0]
                for t in key_tokens:
                    generated_tokens.append(t)
                    input_ids.append(t)
                self.state = JSONState.EXPECT_FUNCTION_NAME
                continue
            elif self.state == JSONState.EXPECT_ARGUMENT_KEY:
                key_tokens = self.model.encode(
                    '", "arguments": {"a": ').tolist()[0]
                for t in key_tokens:
                    generated_tokens.append(t)
                self.state = JSONState.EXPECT_ARG_A
                continue
            elif self.state == JSONState.EXPECT_ARGUMENT_KEY:
                key_tokens = self.model.encode(', "b": ').tolist()[0]
                for t in key_tokens:
                    generated_tokens.append(t)
                self.state = JSONState.EXPECT_ARG_B
                continue
            elif self.state == JSONState.EXPECT_FUNCTION_NAME:
                valid_ids = [
                    self.model.encode(name).tolist()[0][0]
                    for name in self.allowed_functions
                ]
            else:
                valid_ids = [self.model._tokenizer.eos_token_id]

            logits_tensor = self.mask_invalid_tokens(logits_tensor, valid_ids)
            next_token_id = torch.argmax(logits_tensor).item()

            if next_token_id == self.model._tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            input_ids.append(next_token_id)
            self._update_state(next_token_id)

        return self.model.decode(generated_tokens)

    def _update_state(self, last_token_id):
        """State transition logic based on the last generated token."""
        if self.state == JSONState.EXPECT_OPEN_BRACE:
            self.state = JSONState.EXPECT_FUNCTION_KEY
        
