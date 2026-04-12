# Evaluation Guide — Call Me Maybe

---

## Concepts you must master

### What does the project do?
Translates natural language requests into structured, machine-executable function calls in JSON.

Example:
```
INPUT:   "What is the sum of 2 and 3?"
OUTPUT:  {"prompt": "...", "fn_name": "fn_add_numbers", "args": {"a": 2.0, "b": 3.0}}
```

---

### What is Constrained Decoding?
LLMs generate text token by token. At each step they produce logits — one score
for each of the ~150,000 tokens in the vocabulary.

Without constrained decoding the model can generate anything.
With constrained decoding we intercept the logits BEFORE selecting the token
and set all invalid tokens to -infinity.

```
Step 1: model produces 151,936 logits
Step 2: we decide which tokens are valid right now
Step 3: logits[invalid_tokens] = -inf
Step 4: argmax(logits) -> can only select valid tokens
Step 5: append token, repeat
```

Result: 100% valid JSON guaranteed by construction, not by probability.

---

### Why a Trie?
To select the correct function we need to know which tokens are valid
at each step of the function name generation.

The Trie is a prefix tree that lets us instantly know
which tokens can continue the current sequence:

```
root
└── fn (8822)
    ├── _add (2891)
    │   └── _numbers (32964) -> fn_add_numbers
    ├── _greet (????) -> fn_greet
    └── _reverse (????) -> _string -> fn_reverse_string
```

Why not iterate over all 150k tokens at each step?
Because the Trie is built ONCE at startup and queries are instantaneous.

---

### Why two phases?

Phase 1 — Function selection:
- The model reads the prompt and chooses which function to use
- Constrained decoding restricts output to valid function names
- Uses the Trie to know which tokens are valid

Phase 2 — Argument generation:
- We already know the function -> we already know which parameters it needs
- For each parameter we generate the value with constrained decoding
- Valid tokens depend on the type: number, string, boolean

---

### How do you handle types?

| Type    | Valid tokens           | Stop condition                              |
|---------|------------------------|---------------------------------------------|
| boolean | only true or false     | single token                                |
| number  | digits 0-9 and dot     | when the model selects a non-numeric token  |
| string  | any token              | when the model generates a closing quote    |

---

### What is the vocabulary and why do you use it?
The vocabulary is a JSON that maps each token to its string:
```
{"fn": 8822, "{": 90, "_add": 2891, ...}
```

We invert it to look up by ID:
```python
vocab_inverted = {8822: "fn", 90: "{", ...}
```

We use it in constrained decoding to know what text each token ID represents
and thus decide whether it is valid or not.

---

### Why does fn_add_numbers tokenize into 3 tokens?
The model uses BPE (Byte-Pair Encoding) — an algorithm that splits text
into frequent fragments. fn_add_numbers is not common enough as a single token,
so it is split into:
```
fn_add_numbers -> ["fn", "_add", "_numbers"] -> [8822, 2891, 32964]
```

That is why the Trie works with tokens, not with complete strings.

---

## Project structure

```
src/
├── __main__.py        # Orchestrator — coordinates everything
├── models.py          # Pydantic: FunctionDefinition, FunctionCallTest, FunctionCallResult
├── utils.py           # load_function_definitions(), load_function_tests(), write_results()
├── constrained_dec.py # VocabularyMapper, FunctionTrie, select_function(), generate_argument()
├── generator.py       # FunctionCaller — connects both phases
└── tools.py           # Actual function implementations
```

---

## Frequently asked questions in evaluation

Why do you use Pydantic?
To validate that the input data has the correct format.
If a field is missing or the type is wrong, Pydantic raises a clear error
instead of silently crashing later.

Why is the Trie built in __init__ and not in generate()?
Because the available functions do not change during execution.
Building it once is more efficient than rebuilding it for each prompt.

What happens if the input JSON file is malformed?
The program catches json.JSONDecodeError and shows a clear message
without crashing.

Why not use dspy/outlines/transformers directly?
The subject explicitly forbids it. We only use the public methods
of the provided SDK.

Why greedy decoding (argmax) and not sampling?
Because we already restrict the valid tokens — we do not need randomness.
Greedy is faster and fully deterministic.

How do you guarantee 100% valid JSON?
It is not statistical — it is structural. The decoder can never generate
an invalid token because we have set them to -infinity. It is impossible
to produce broken JSON.

---

## Tests to show during evaluation

### 1. Run the program
```bash
uv sync
make run
```

### 2. Check the output
```bash
cat data/output/function_calling_results.json
```

### 3. Pass lint
```bash
make lint
make lint-strict
```

### 4. Test with custom input
```bash
uv run python -m src --input data/input/function_calling_tests.json \
                     --output data/output/function_calling_results.json
```

### 5. Test error handling
```bash
uv run python -m src --input data/input/does_not_exist.json
# Should show: Error: File not found - ...
```

---

## Points that demonstrate real understanding

1. Know why -infinity and not 0
   Because softmax(0) is still a positive probability.
   -infinity -> exact probability of 0.

2. Know why the Trie and not a simple if fn_name in functions
   Because generation is token by token, not word by word.

3. Know the difference between tokenization and vocabulary
   Tokenization: process of splitting text into tokens
   Vocabulary: the dictionary that maps tokens to IDs

4. Know why small models fail without constrained decoding
   Only 600M parameters -> limited understanding -> broken JSON ~70% of the time

5. Know what VocabularyMapper does
   Bridge between token IDs (numbers) and strings
   Needed to know whether a token is valid at each step

---

## Why do we pass the function definitions as JSON and not just ask the AI directly?

Good question. You might think: "If the AI is intelligent, why not just ask it: hey, what function should I call and with what arguments?" The answer has several layers.

**First**, the AI without any structure can respond in any way it wants. If you ask "What function should I call for the sum of 2 and 3?" it might respond "You should call the addition function with the values two and three" — which is completely useless for a program that needs to execute code.

**Second**, the JSON with function definitions is not for the AI to "read it like a human". It is the contract between your program and the AI. Your program reads that JSON, knows exactly what functions exist, what parameters they need and of what type. That information is what lets you build the Trie and apply constrained decoding. Without it you would not know which tokens are valid.

**Third**, the AI does not execute anything. It only generates text. The JSON is what your program uses to validate, structure and eventually execute the real function. The AI suggests, your program decides and validates.

**In short**: we use JSON because computers need exact, structured, predictable contracts. The AI is the brain that understands language, but JSON is the language that the program understands. Your project is precisely the bridge between the two worlds — that is the whole point of function calling.


## Is this project not about an AI? Should the AI not do everything itself?

Yes, the project uses an AI — the Qwen3-0.6B model. But there is an important difference between what the AI does well and what it does not.
The AI is very good at understanding human language. If you say "What is the sum of 2 and 3?" it perfectly understands that you are talking about an addition and that the numbers are 2 and 3. That is real intelligence.

What the AI does poorly — especially small models with 600M parameters — is producing reliable structured output. If you ask it to generate valid JSON on its own, it only succeeds about 30% of the time. The other 70% it produces broken JSON, free text, or anything else.

So the project is not about replacing the AI — it is about combining the best of both worlds. The AI provides semantic understanding — it knows which function makes the most sense for each prompt. Your code provides structural reliability — it guarantees that the output is always 100% valid JSON.

Think of it like a surgeon and a nurse. The surgeon has the knowledge and makes the decisions. The nurse ensures the environment is safe and controlled. Without the nurse the surgeon could make avoidable mistakes. Without the surgeon the nurse cannot operate. In this project the AI is the surgeon and the constrained decoding is the nurse.


## Exactly! So is the AI in this project useless? Would a more powerful AI do everything on its own?

**Yes and no. Let me be honest.**
If you used GPT-4 or Claude instead of Qwen3-0.6B, a simple prompt would probably get you valid JSON most of the time. Larger models have seen so much training data that they are much better at following structured output instructions.

But there are three reasons why constrained decoding still matters even with powerful models.

**First**, cost. Calling GPT-4 for every prompt costs money — real money at scale. A small 600M model running locally is essentially free. If you can make a small model perform like a large one using constrained decoding, you win.

**Second**, reliability. Even GPT-4 fails sometimes. "Most of the time" is not good enough in production systems. A bank, a hospital, an autonomous system — they need 100% reliability, not 95%. Constrained decoding gives you that guarantee by construction, regardless of model size.

**Third**, speed and privacy. Running a small model locally is much faster than an API call and your data never leaves your machine. For sensitive applications that matters a lot.

So yes — a more powerful AI would do better on its own. But constrained decoding solves a real engineering problem: making any model, big or small, produce perfectly structured output every single time. That is the actual skill this project teaches you.