#!/usr/bin/env python3
import json
from pathlib import Path
from src.models import FunctionDefinition, FunctionCallTest


def load_function_definitions(route: str) -> list[FunctionDefinition]:
    """
    Loads and validates function definitions from a JSON file.

    Args:
        route: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FunctionDefinition objects.
    """
    with open(route, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [FunctionDefinition(**dicc) for dicc in raw_data]


def load_function_tests(route: str) -> list[FunctionCallTest]:
    """
    Loads and validates test cases from a JSON file.

    Args:
        route: Path to the JSON file containing function calling tests.

    Returns:
        A list of validated FunctionCallTest objects.
    """
    with open(route, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    return [FunctionCallTest(**dicc) for dicc in raw_data]


def write_results(results: list, output_path: Path) -> None:
    """
    Saves the generated function calling results to a JSON file.

    Creates the parent directories if they do not exist.

    Args:
        results: List of dictionaries containing the generation results.
        output_path: Path object specifying where to save the file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
