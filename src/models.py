#!/usr/bin/env python3
"""
Models for cuntion calling data validation using Pydantic.
Ensures strict type checking for input definitions and output results.
"""

from pydantic import BaseModel
from typing import Dict, Any, Literal


class ParameterType(BaseModel):
    """
    Represents the data type of a function parameter or return value.
    """
    type: Literal["number", "string", "boolean"]


class FunctionDefinition(BaseModel):
    """
    Detaild structure of an available function.
    Maps name, description, and its expected parameters/return type.
    """
    name: str
    description: str
    parameters: Dict[str, ParameterType]
    returns: ParameterType


class FunctionCallTest(BaseModel):
    """Input prompt from function_calling_test.json"""
    prompt: str


class FunctionCallResult(BaseModel):
    """
    Expected output structure for function_calling_results.json.
    Includes the original prompt, the selected function,
    and extracted arguments.
    """
    prompt: str
    fn_name: str
    args: Dict[str, Any]
