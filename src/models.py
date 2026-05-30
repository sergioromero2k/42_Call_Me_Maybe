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
    type: Literal["object"]
    properties: Dict[str, ParameterProperty]
    required: List[str] = Field(default_factory=list)


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    returns: Optional[Dict[str, Any]] = None


class TestPrompt(BaseModel):
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
