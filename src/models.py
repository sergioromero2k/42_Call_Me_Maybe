#!/usr/bin/env python3
"""
Models for function calling data validation using Pydantic.
Ensures strict type checking for input definitions and output results.
"""

from typing import Dict, Any, List, Literal, Optional
from pydantic import BaseModel, Field


class ParameterProperty(BaseModel):
    """
    Schema for defining the data type of a function parameter or return value.

    Attributes:
        type: The allowed data type string (number, integer, string, etc).
        description:
        properties:
        items:
        enum:
    """
    type: Literal["number", "string", "integer", "boolean", "object", "array"]
    description: Optional[str] = None

    properties: Optional[Dict[str, Any]] = None
    items: Optional[Dict[str, Any]] = None
    enum: Optional[List[Any]] = None


class ParametersSchema(BaseModel):
    """Schema definition for a function's parameters block.

    Attributes:
        type: Always 'object', as required by the JSON Schema spec.
        properties: A mapping of parameter names to their property
            definitions, including type, description, and constraints.
        required: A list of parameter names that are mandatory.
            Defaults to an empty list.
    """
    type: Literal["object"]
    properties: Dict[str, ParameterProperty]
    required: List[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    """Represents a callable function available to the model.

    Used to validate and store function metadata loaded from the
    functions_definition.json file. Each instance describes one
    function the constrained decoding engine can select and call.

    Attributes:
        name: The function's identifier, used as the trie key.
        description: A human-readable explanation of what the
            function does, shown to the model during selection.
        parameters: A dictionary mapping parameter names to their
            metadata such as type, description, and enum values.
            Defaults to an empty dict.
        returns: Optional description of the function's return value.
    """
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Optional[Dict[str, Any]] = None


class TestPrompt(BaseModel):
    """Represents a single test case from the test prompts file.

    Attributes:
        prompt: The natural language user request to be processed
            by the constrained decoding engine.
    """
    prompt: str


class FunctionCallResult(BaseModel):
    """
    Schema for the final output of a function calling operation.

    This matches the structure required for 'function_calling_results.json'.

    Attributes:
        prompt: The original input prompt.
        fn_name: The name of the function identified by the model.
        args: A dictionary of key-value pairs representing
                the generated arguments.
    """
    prompt: str
    fn_name: str
    args: Dict[str, Any]
