import argparse
import json
import sys
import time
from pathlib import Path
from pydantic import ValidationError
from src.utils import (
    load_function_definitions, load_function_tests, write_results)
from src.constrained_dec import build_trie
from src.generator import FunctionCaller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_dec import VocabularyMapper


def main() -> None:
    """
    Main entry point for the LLM function calling tool.

    Orchestrates the loading of function definitions and test prompts.
    initializes the LLM and the constrained decoding components,
    and executes the inference process to generate structured JSON output.

    Args:
        None (Uses command-line arguments: --input, --output).

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
    parser.add_argument("--input", default="data/input", type=str)
    parser.add_argument("--output", default="data/output", type=str)
    args = parser.parse_args()

    input_path = Path(args.input)
    route_definitions = input_path / "functions_definition.json"
    route_tests = input_path / "function_calling_tests.json"

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
    model = Small_LLM_Model()
    print("Model loaded successfully!")

    # Components for constrained decoding.
    mapper = VocabularyMapper(model)
    trie = build_trie(functions, model)
    caller = FunctionCaller(model, mapper, trie, functions)

    resultados = []
    start = time.time()
    for test in tests:
        # Generate structured output using constrained decoding.
        resultado = caller.call(test.prompt)
        resultados.append(resultado.model_dump())

    try:
        elapsed = time.time() - start
        print(f"Tiempo total: {elapsed:.2f} segundos")
        output_path = Path(args.output) / "function_calling_results.json"
        write_results(resultados, output_path)
    except Exception as e:
        print(f"Error initializing components: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
