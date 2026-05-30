#!/usr/bin/env python3

import sys
import json
import os
import argparse

from typing import Any, List, Dict, Optional
from llm_sdk import Small_LLM_Model

from src.models import FunctionDefinition
from src.tokenizer import CustomTokenizer
from src.constrained_dec import build_trie, select_function, generate_argument


def print_visual_step(step_name: str, status: str, details: str = "") -> None:
    """Helper for process visualization in the terminal."""
    emoji = "⚙️"
    if "OK" in status:
        emoji = "✅"
    elif "ERROR" in status:
        emoji = "❌"
    elif "RUN" in status:
        emoji = "🚀"
    print(f"[{emoji} {step_name:<20}] -> {status:<8} | {details}")


def write_empty_output(output_path: str) -> None:
    """Writes a default empty JSON schema to output_path in case of failure."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"function": "", "arguments": {}}, f)
    except Exception as e:
        print_visual_step(
            "Saving Output", "ERROR", f"Could not write empty output: {e}"
        )


def main() -> None:

    # 1. CLI arguments — --input and --output as per subject IV.3.2
    parser = argparse.ArgumentParser(
        description="Constrained Decoding Engine for Function Calling"
    )
    parser.add_argument(
        "--input",
        default="data/input/",
        help="Path to the input JSON file or directory"
    )
    parser.add_argument(
        "--output",
        default="data/output/",
        help="Path to the output JSON file or directory"
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output

    # If paths point to directories, append standard file names
    if os.path.isdir(input_path) or input_path.endswith("/"):
        input_path = os.path.join(input_path, "example.json")

    if os.path.isdir(output_path) or output_path.endswith("/"):
        output_path = os.path.join(
            output_path, "function_calling_results.json")

    print("\n" + "=" * 60)
    print("== STARTING CONSTRAINED DECODING ENGINE (PRO VERSION) ==")
    print("=" * 60)

    # 2. Load and validate the input JSON
    try:
        print_visual_step("File Upload", "RUN", f"Reading {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_input_data = json.load(f)
    except FileNotFoundError:
        print_visual_step(
            "File Load", "ERROR",
            f"File not found: {input_path}")
        write_empty_output(output_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_visual_step("File Upload", "ERROR", f"Invalid JSON: {e}")
        write_empty_output(output_path)
        sys.exit(1)

    # 3. Extract and validate functions using Pydantic
    # Each function is validated individually so one
    # bad entry does not kill the rest
    try:
        prompt = raw_input_data.get("prompt", "")
        raw_functions = raw_input_data.get("functions", [])

        validated_functions: List[FunctionDefinition] = []
        for raw_fn in raw_functions:
            try:
                validated_functions.append(FunctionDefinition(**raw_fn))
            except Exception as e:
                print_visual_step(
                    "Pydantic Validation", "ERROR",
                    f"Invalid function ignored: {e}"
                )
                continue

        print_visual_step(
            "Pydantic Validation", "OK",
            f"Found {len(validated_functions)} valid functions."
        )
    except Exception as e:
        print_visual_step(
            "Pydantic Validation", "ERROR",
            f"Failed to process functions: {e}"
        )
        write_empty_output(output_path)
        sys.exit(1)

    # 4. Load the LLM model
    try:
        print_visual_step(
            "LLM Load", "RUN",
            "Instantiating Small_LLM_Model...")
        model = Small_LLM_Model()
        print_visual_step("LLM Load", "OK", "Model instantiated.")
    except Exception as e:
        print_visual_step("LLM Load", "ERROR", f"Could not load model: {e}")
        write_empty_output(output_path)
        sys.exit(1)

    # 5. Load the custom tokenizer using the vocabulary path from the model
    try:
        vocab_path = model.get_path_to_vocab_file()
        print_visual_step(
            "Manual Tokenizer", "RUN",
            f"Loading vocabulary from: {vocab_path}"
        )
        tokenizer = CustomTokenizer(vocab_path)
        print_visual_step("Manual Tokenizer", "OK", "Vocabulary initialized.")
    except Exception as e:
        print_visual_step(
            "Manual Tokenizer", "ERROR",
            f"Could not load tokenizer: {e}"
        )
        write_empty_output(output_path)
        sys.exit(1)

    # 6. Build the prefix Trie from the validated functions
    try:
        print_visual_step(
            "Construct Trie", "RUN",
            "Indexing function tokens...")
        trie = build_trie(validated_functions, tokenizer)
        print_visual_step(
            "Trie Construction", "OK",
            "Numeric prefix tree ready.")
    except Exception as e:
        print_visual_step(
            "Trie Construction", "ERROR",
            f"Failed to construct the Trie: {e}"
        )
        write_empty_output(output_path)
        sys.exit(1)

    # 7. Constrained Decoding Phase 1 — select the best function via logits
    try:
        print_visual_step(
            "Phase 1: Logits Fn", "RUN",
            f"Evaluating prompt: '{prompt}'"
        )
        selected_fn_name = select_function(prompt, model, tokenizer, trie)

        if not selected_fn_name:
            print_visual_step(
                "Phase 1: Logits Fn", "ERROR", "No function selected."
            )
            write_empty_output(output_path)
            return

        print_visual_step(
            "Phase 1: Logits Fn", "OK",
            f"Winner -> {selected_fn_name}"
        )
    except Exception as e:
        print_visual_step(
            "Phase 1: Logits Fn", "ERROR",
            f"Failed in select_function: {e}"
        )
        write_empty_output(output_path)
        sys.exit(1)

    # 8. Constrained Decoding Phase 2 — extract arguments based on
    # parameter types
    # If one parameter fails we store "" and continue to recover partial points
    extracted_arguments: Dict[str, Any] = {}

    try:
        target_fn: Optional[FunctionDefinition] = next(
            (fn for fn in validated_functions if fn.name == selected_fn_name),
            None
        )

        if (
            target_fn
            and target_fn.parameters
            and target_fn.parameters.properties
        ):
            properties_dict = target_fn.parameters.properties
            print_visual_step(
                "Phase 2: Arguments", "RUN",
                f"Processing {len(properties_dict)} parameters..."
            )

            for param_name, param_prop in properties_dict.items():
                try:
                    val = generate_argument(
                        prompt=prompt,
                        param_type=param_prop.type,
                        model=model,
                        tokenizer=tokenizer,
                        param_name=param_name,
                    )
                    extracted_arguments[param_name] = val
                    print_visual_step(
                        "Phase 2: Arguments", "OK",
                        f"↳ [{param_name}] ({param_prop.type}) -> {repr(val)}"
                    )
                except Exception as e:
                    print_visual_step(
                        "Phase 2: Arguments", "ERROR",
                        f"Parameter error [{param_name}]: {e}"
                    )
                    extracted_arguments[param_name] = ""
        else:
            print_visual_step(
                "Phase 2: Arguments", "OK",
                "The function does not require parameters."
            )
    except Exception as e:
        print_visual_step(
            "Phase 2: Arguments", "ERROR",
            f"Failed to extract arguments: {e}"
        )

    # 9. Build and write the final output payload
    try:
        output_payload = {
            "function": selected_fn_name,
            "arguments": extracted_arguments,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=4, ensure_ascii=False)
        print_visual_step("Output Saved", "OK", f"Written to: {output_path}")
        print("=" * 60 + "\nPROCESS COMPLETED SUCCESSFULLY\n")
    except Exception as e:
        print_visual_step(
            "Saving Output", "ERROR",
            f"Could not write output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
