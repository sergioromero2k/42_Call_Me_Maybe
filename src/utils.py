#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from src.models import FunctionDefinition, FunctionCallTest
from typing import Any


def load_function_definitions(route: Path) -> list[FunctionDefinition]:
    """
    Loads and validates function definitions from a JSON file.
    Gracefully handles missing files or invalid JSON systanx.

    Args:
        route: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FunctionDefinition objects.
    """
    if not route.exists():
        msg = f"Error: The definitions file does not exist at '{route}'."
        print(msg, file=sys.stderr)
        sys.exit(1)

    try:
        with open(route, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            msg = f"Error: Expected a JSON array in '{route}'."
            print(msg, file=sys.stderr)
            sys.exit(1)
        return [FunctionDefinition(**dicc) for dicc in raw_data]
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format inside '{route}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading definitions: {e}", file=sys.stderr)
        sys.exit(1)


def load_function_tests(route: Path) -> list[FunctionCallTest]:
    """
    Loads and validates test cases from a JSON file.
    Gracefully handles missing files or invalid JSON syntax.

    Args:
        route: Path to the JSON file containing function calling tests.

    Returns:
        A list of validated FunctionCallTest objects.
    """

    if not route.exists():
        msg = "Error: The input test file does not exist at '{route}'."
        print(msg, file=sys.stderr)
        sys.exit(1)
    try:
        with open(route, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            msg = f"Error: Expected a JSON array in '{route}'."
            print(msg, file=sys.stderr)
            sys.exit(1)
        return [FunctionCallTest(**dicc) for dicc in raw_data]
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format inside '{route}'.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading tests: {e}", file=sys.stderr)
        sys.exit(1)


def write_results(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Saves the generated function calling results to a JSON file.
    Creates the parent directories if they do not exist.

    Args:
        results: List of dictionaries containing the generation results.
        output_path: Path object specifying where to save the file.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    except Exception as e:
        msg = f"Error writing output file to '{output_path}': {e}"
        print(msg, file=sys.stderr)
        sys.exit(1)
