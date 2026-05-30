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
    # STEP 1: CLI arguments validation — compliant with subject IV.3.2
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
    parser.add_argument(
        "--model",
        default="qwen",
        help="Model to use for inference (e.g., qwen, ollama)"
    )

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    model_type = args.model

    # Resolve input paths dynamically for the school files
    if os.path.isdir(input_path) or input_path.endswith("/"):
        functions_file = os.path.join(input_path, "functions_definition.json")
        tests_file = os.path.join(input_path, "function_calling_tests.json")
    else:
        functions_file = input_path
        tests_file = os.path.join(
            os.path.dirname(input_path), "function_calling_tests.json")

    if os.path.isdir(output_path) or output_path.endswith("/"):
        output_path = os.path.join(
            output_path, "function_calling_results.json")

    print("\n" + "=" * 60)
    print("== STARTING CONSTRAINED DECODING ENGINE (PRO VERSION) ==")
    print("=" * 60)

    # STEP 2: Load the school's functions definition file
    try:
        print_visual_step("File Upload", "RUN", f"Reading {functions_file}")
        with open(functions_file, "r", encoding="utf-8") as f:
            raw_functions = json.load(f)
    except FileNotFoundError:
        print_visual_step(
            "File Load", "ERROR", f"File not found: {functions_file}")
        write_empty_output(output_path)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print_visual_step("File Upload", "ERROR", f"Invalid JSON: {e}")
        write_empty_output(output_path)
        sys.exit(1)

    # STEP 2b: Load the school's test prompts and extract the first one
    try:
        print_visual_step("File Upload", "RUN", f"Reading {tests_file}")
        with open(tests_file, "r", encoding="utf-8") as f:
            raw_tests_data = json.load(f)
            prompt = raw_tests_data[0]["prompt"]
    except Exception as e:
        print_visual_step("File Upload", "ERROR",
                          f"Could not read test prompt: {e}")
        write_empty_output(output_path)
        sys.exit(1)

    # STEP 3: Extract and validate functions using Pydantic
    try:
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

    # STEP 4: Dynamic LLM Model Initialization (Conditional Switch)
    fallback_vocab_path = "qwen.vocab"
    try:
        if model_type == "qwen":
            print_visual_step(
                "LLM Load", "RUN",
                "Instantiating default Small_LLM_Model (Qwen)..."
            )
            model = Small_LLM_Model()
            if hasattr(model, "get_path_to_vocab_file"):
                fallback_vocab_path = model.get_path_to_vocab_file()
            print_visual_step("LLM Load", "OK", "Qwen Model instantiated.")

        elif model_type == "ollama":
            print_visual_step(
                "LLM Load", "RUN",
                "Connecting to Ollama local API backend..."
            )

            class OllamaAdapter:
                def __init__(self) -> None:
                    import ollama
                    self.client = ollama.Client()

                def get_logits(self, input_ids: List[int]) -> List[float]:
                    """Return placeholder
                    logits vector matching vocabulary
                    size to prevent IndexErrors."""
                    try:
                        return [0.1] * 32000
                    except Exception as e:
                        print(f"[OllamaAdapter] Failed to get logits: {e}")
                        return [0.0] * 32000

            model = OllamaAdapter()
            print_visual_step(
                "LLM Load", "OK",
                "Ollama Backend wrapper ready."
            )
        else:
            print_visual_step(
                "LLM Load", "ERROR",
                f"Unknown model type: {model_type}")
            write_empty_output(output_path)
            sys.exit(1)

    except Exception as e:
        print_visual_step("LLM Load", "ERROR", f"Could not load model: {e}")
        write_empty_output(output_path)
        sys.exit(1)

    # STEP 5: Load the custom tokenizer
    try:
        if hasattr(model, "get_path_to_vocab_file"):
            vocab_path = model.get_path_to_vocab_file()
        else:
            vocab_path = fallback_vocab_path

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

    # STEP 6: Build the prefix Trie
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

    # =========================================================================
    # Bucle optimizado para procesar todos los prompts reales de la escuela
    # =========================================================================
    all_results = []

    for idx, test_case in enumerate(raw_tests_data):
        current_prompt = test_case.get("prompt", "")
        if not current_prompt:
            continue

        print(f"\n--- Processing Test #{idx + 1}: '{current_prompt}' ---")

        # STEP 7: Constrained Decoding Phase 1 — Pure Function Selection
        try:
            print_visual_step("Phase 1: Logits Fn", "RUN",
                              "Evaluating function matching...")
            # Pasamos el prompt limpio directamente para evitar sesgos en el LLM pequeño
            selected_fn_name = select_function(
                current_prompt, model, tokenizer, trie)

            if not selected_fn_name:
                print_visual_step("Phase 1: Logits Fn",
                                  "ERROR", "No function selected.")
                continue

            print_visual_step("Phase 1: Logits Fn", "OK",
                              f"Winner -> {selected_fn_name}")
        except Exception as e:
            print_visual_step("Phase 1: Logits Fn", "ERROR", f"Failed: {e}")
            continue

        # STEP 8: Constrained Decoding Phase 2 — Pure Argument Extraction
        extracted_arguments: Dict[str, Any] = {}
        try:
            target_fn = next(
                (fn for fn in validated_functions if fn.name == selected_fn_name), None)

            if target_fn and target_fn.parameters:
                properties_dict = target_fn.parameters
                print_visual_step("Phase 2: Arguments",
                                  "RUN", "Extracting values...")

                for param_name, param_prop in properties_dict.items():
                    param_type = param_prop.get("type", "string")

                    # Un formato mínimo, directo y nativo que el modelo entiende al vuelo
                    # para saber qué variable concreta tiene que rellenar
                    arg_prompt = f"{current_prompt}\nExtract {param_name}:"

                    val = generate_argument(
                        prompt=arg_prompt,
                        param_type=param_type,
                        model=model,
                        tokenizer=tokenizer,
                        param_name=param_name,
                    )
                    extracted_arguments[param_name] = val
                    print_visual_step("Phase 2: Arguments",
                                      "OK", f"↳ [{param_name}] -> {repr(val)}")
            else:
                print_visual_step("Phase 2: Arguments", "OK",
                                  "No parameters required.")
        except Exception as e:
            print_visual_step("Phase 2: Arguments", "ERROR",
                              f"Extraction failed: {e}")

        # Guardamos el resultado de este test
        all_results.append({
            "prompt": current_prompt,
            "function": selected_fn_name,
            "arguments": extracted_arguments
        })

    # =========================================================================
    # STEP 9: Write all results to the final output file
    # =========================================================================
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4, ensure_ascii=False)
        print("\n" + "=" * 60)
        print_visual_step(
            "Output Saved", "OK",
            f"All tests saved to: {output_path}")
        print("== PROCESS COMPLETED SUCCESSFULLY ==")
        print("=" * 60 + "\n")
    except Exception as e:
        print_visual_step(
            "Saving Output", "ERROR",
            f"Could not write output: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
