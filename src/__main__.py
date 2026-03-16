import argparse
import json
import sys
from pathlib import Path
from pydantic import ValidationError
from src.utils import load_function_definitions, load_function_tests
from llm_sdk.llm_sdk import Small_LLM_Model


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

    # Step 2: Initialize the Small LLM MOdel (loads weights into memory)
    print("Initializing LLM model...")
    model = Small_LLM_Model()

    # Step 3: Test the pipeline with the first available test case
    first_test = tests[0]

    # Convert the text prompt into numerical token IDs (Tensors)
    input_tensor = model.encode(first_test.prompt)

    # Convert Tensor to a flat Python list for the logit function
    input_ids_list = input_tensor.tolist()[0]
    print(f"Tokens del prompt: {input_ids_list}")

    # Retrieve the logits (raw probabilities) for the next predicted token
    logits = model.get_logits_from_input_ids(input_ids_list)

    # Display internal state for verification
    print(f"Cantidad de tokens en el vocabulario: {len(logits)}")
    print(f"Primeros 5 logits (probabilidades brutas): {logits[:5]}")


if __name__ == "__main__":
    main()
