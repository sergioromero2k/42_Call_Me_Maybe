import argparse
import json
import sys
import time
from pathlib import Path
from pydantic import ValidationError
from src.utils import (
    load_function_definitions, load_function_tests, write_results)
from src.generator import FunctionCaller
from llm_sdk import Small_LLM_Model
from src.constrained_dec import build_trie, VocabularyMapper


def main() -> None:
    """
    Main entry point for the LLM function calling tool.

    Orchestrates the loading of function definitions and test prompts.
    initializes the LLM and the constrained decoding components,
    and executes the inference process to generate structured JSON output.

    Args:
        None (Uses command-line arguments:
            --functions_definition, --input, --output).

    Raises:
        FileNotFoundError: If input JSON files are missing.
        json.JSONDecodeError: If input files contain invalid JSON.
        ValidationError: If data does not match Pydantic schemas.
        SystemExit: On any fatal error to ensure a graceful crash
    """
    parser = argparse.ArgumentParser(
        description="42 Call Me Maybe - LLM Function Caller"
    )
    # Command-line arguments for input and output directions
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        type=str
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        type=str
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        type=str
    )

    args = parser.parse_args()
    route_definitions = Path(args.functions_definition)
    route_tests = Path(args.input)
    output_path = Path(args.output)

    try:
        # Load and validate function definitions and tests
        functions = load_function_definitions(route_definitions)
        tests = load_function_tests(route_tests)
        print(
            f"Success: Loaded {len(functions)} "
            f"functions and {len(tests)} tests.")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e.msg}")
        sys.exit(1)
    except ValidationError as e:
        print(f"Error: Data validation failed - {e.json()}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

    print("Initializing LLM model...")
    try:
        model = Small_LLM_Model()
        print("Model loaded successfully!")

        # Components for constrained decoding.
        mapper = VocabularyMapper(model)
        trie = build_trie(functions, model)
        caller = FunctionCaller(model, mapper, trie, functions)
    except Exception as e:
        print(f"Error initializing LLM components: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    start = time.time()
    for test in tests:
        # Generate structured output using constrained decoding.
        result = caller.call(test.prompt)
        results.append(result.model_dump())

    elapsed = time.time() - start
    print(f"Tiempo total: {elapsed:.2f} segundos")

    try:
        write_results(results, output_path)
        print(f"Results successfully saved to: {output_path}")
    except Exception as e:
        print(f"Error writing output results: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
