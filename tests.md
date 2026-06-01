┌[serromer☮c3r14s4.42madrid.com]-(/sgoinfre/students/serromer/42_Call_Me_Maybe)-[git://main ✔]-
└> make run
Running the project...
uv run python -m src

============================================================
== STARTING CONSTRAINED DECODING ENGINE (PRO VERSION) ==
============================================================
[🎉​ File Upload         ] -> RUN      | Reading data/input/functions_definition.json
[🎉​ File Upload         ] -> RUN      | Reading data/input/function_calling_tests.json
[🆗​ Pydantic Validation ] -> OK       | Found 5 valid functions.
[🎉​ LLM Load            ] -> RUN      | Instantiating default Small_LLM_Model (Qwen)...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 311/311 [00:13<00:00, 23.90it/s]
[🆗​ LLM Load            ] -> OK       | Qwen Model instantiated.
[🎉​ Manual Tokenizer    ] -> RUN      | Loading vocabulary from: /sgoinfre/students/serromer/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca/vocab.json
[🆗​ Manual Tokenizer    ] -> OK       | Vocabulary initialized.
[🆗​ Inference Tokenizer ] -> OK       | Using model's internal tokenizer for scoring.
[🎉​ Construct Trie      ] -> RUN      | Indexing function tokens...
[🆗​ Trie Construction   ] -> OK       | Numeric prefix tree ready.

--- Processing Test #1: 'What is the sum of 2 and 3?' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: What is the sum of 2 and 3?...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_add_numbers
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [a] -> 2.0
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [b] -> 3.0

--- Processing Test #2: 'What is the sum of 265 and 345?' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: What is the sum of 265 and 345?...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_add_numbers
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [a] -> 265.0
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [b] -> 345.0

--- Processing Test #3: 'Greet shrek' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Greet shrek...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_greet
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [name] -> 'fn_greet<|im_end|>'

--- Processing Test #4: 'Greet john' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Greet john...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_greet
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [name] -> 'John<|im_end|>'

--- Processing Test #5: 'Reverse the string 'hello'' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Reverse the string 'hello'...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_reverse_string
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [s] -> 'hello<|im_end|>'

--- Processing Test #6: 'Reverse the string 'world'' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Reverse the string 'world'...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_reverse_string
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [s] -> 'world<|im_end|>'

--- Processing Test #7: 'What is the square root of 16?' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: What is the square root of 16?...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_get_square_root
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [a] -> 16.0

--- Processing Test #8: 'Calculate the square root of 144' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Calculate the square root of 144...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_get_square_root
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [a] -> 144.0

--- Processing Test #9: 'Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Replace all numbers in "Hello 34 I'm 233 years old" with NUMBERS...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_substitute_string_with_regex
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [source_string] -> "Hello 34 I'm 233 years old, regex=.*\\d+|.*\\d+|.*\\d+, replacement=NUMBERS<|im_end|>"
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [regex] -> '.*\\d+|.*\\d+|.*\\d+<|im_end|>'
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [replacement] -> 'NUMBERS<|im_end|>'

--- Processing Test #10: 'Replace all vowels in 'Programming is fun' with asterisks' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Replace all vowels in 'Programming is fun' with asterisks...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_substitute_string_with_regex
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [source_string] -> 'Programming is fun'
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [regex] -> '.*[aeiouAEIOU]'
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [replacement] -> '*<|im_end|>'

--- Processing Test #11: 'Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'' ---
[🎉​ Phase 1: Logits Fn  ] -> RUN      | Evaluating prompt: Substitute the word 'cat' with 'dog' in 'The cat sat on the mat with another cat'...
[🆗​ Phase 1: Logits Fn  ] -> OK       | Winner -> fn_substitute_string_with_regex
[🎉​ Phase 2: Arguments  ] -> RUN      | Extracting values...
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [source_string] -> 'The cat sat on the mat with another cat'
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [regex] -> 'cat'
[🆗​ Phase 2: Arguments  ] -> OK       | ↳ [replacement] -> 'dog<|im_end|>'
[​🎃 Saving Output       ] -> ERROR    | Could not write output: [Errno 2] No such file or directory: 'data/output/function_calling_results.json'
make: *** [Makefile:47: run] Error 1