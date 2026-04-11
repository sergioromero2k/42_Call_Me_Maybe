import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.utils import load_function_definitions, load_function_tests
from src.constrained_dec import build_trie
from src.generator import FunctionCaller
from llm_sdk.llm_sdk import Small_LLM_Model
from src.constrained_dec import VocabularyMapper


def main() -> None:
    """
    Load function definitions and test cases from JSON files,
    validate them, and print the numeber of loaded items.
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

    mapper = VocabularyMapper(model)
    trie = build_trie(functions, model)
    caller = FunctionCaller(model, mapper, trie, functions)

    resultados = []
    for test in tests[:1]:
        print(f"Processing: {test.prompt}")
        resultado = caller.call(test.prompt)
        print(f"Result: {resultado}")
        resultados.append(resultado.model_dump())

    output_path = Path(args.output) / "function_calling_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
