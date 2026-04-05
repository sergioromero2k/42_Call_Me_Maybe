#!/usr/bin/env python3
import json
from src.models import FunctionDefinition, FunctionCallTest


def load_function_definitions(route: str) -> list[FunctionDefinition]:
    """
    Load function definitions from a JSON file and return validated objects.
    """
    with open(route, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    list_final = []
    for dicc in raw_data:
        # Validate and create object
        object_validate = FunctionDefinition(**dicc)
        list_final.append(object_validate)
    return list_final


def load_function_tests(route: str) -> list[FunctionCallTest]:
    """
    Load functions call tests from a JSON file and return validated objects.
    """
    with open(route, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    list_final = []
    for dicc in raw_data:
        # Validate and create object
        object_validate = FunctionCallTest(**dicc)
        list_final.append(object_validate)
    return list_final
