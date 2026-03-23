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


class JSONStateMachine:
    def __init__(self, model: Small_LLM_Model, allowed_functions: list[str]):
        self.model = model
        self.allowed_functions = allowed_functions
        self.state = JSONState.EXPECT_OPEN_BRACE

        # Pre-tokens (get specific IDs)
        self.token_lbrace = self.model.encode("{").tolist()[0][-1]
        self.token_function_key = self.model.encode('"function"').tolist()[0]
        self.token_colon = self.model.encode(":").tolist()[0][-1]
        self.token_quote = self.model.encode('"').tolist()[0][-1]

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
        input_ids = self.model.encode(prompt).tolist()[0]
        generated_tokens = []

        for _ in range(max_tokens):
            logits_list = self.model.get_logits_from_input_ids(input_ids)
            logits_tensor = torch.tensor(logits_list)

            valid_ids = []

            if self.state == JSONState.EXPECT_OPEN_BRACE:
                valid_ids = [self.token_lbrace]

            elif self.state == JSONState.EXPECT_FUNCTION_KEY:
                valid_ids = [self.token_quote]

            elif self.state == JSONState.EXPECT_COLON:
                valid_ids = [self.token_colon]

            else:
                valid_ids = [self.model._tokenizer.eos_token_id]

            if valid_ids:
                logits_tensor = self.mask_invalid_tokens(logits_tensor, valid_ids)

            next_token_id = torch.argmax(logits_tensor).item()

            if next_token_id == self.model._tokenizer.eos_token_id:
                break

            generated_tokens.append(next_token_id)
            input_ids.append(next_token_id)

            self._update_state(next_token_id)

        return self.model.decode(generated_tokens)

    def _update_state(self, last_token__id):
        if self.state == (
            JSONState.EXPECT_OPEN_BRACE and last_token__id == self.token_lbrace
        ):
            self.state = JSONState.EXPECT_FUNCTION_KEY
        elif (
            self.state == JSONState.EXPECT_FUNCTION_KEY
            and last_token__id == self.token_quote
        ):
            self.state = JSONState.EXPECT_COLON
