# Test Suite — Constrained Decoding Engine

40 tests covering all core components of the pipeline.
Run with: `uv run pytest tests/ -v`

---

## TestTrieNode — 2 tests

Verifies that trie nodes are created with correct default values
and that child nodes can be assigned properly.

---

## TestFunctionTrie — 8 tests

Verifies the trie data structure:
- Root node exists on creation
- Single token sequences are inserted and marked as end of path
- Two functions with a shared prefix reuse the same nodes
- Empty lists are silently ignored
- Non-list types are rejected
- Boolean values are rejected even though `bool` is a subclass of `int`
- Valid next tokens are returned correctly from any node
- `None` nodes are handled without crashing

---

## TestCustomTokenizer — 10 tests

Verifies the vocabulary-based tokenizer:
- Vocabulary is loaded correctly from a JSON file
- Known tokens are encoded to their correct IDs
- Empty strings return empty lists
- `None` input is handled gracefully
- Known IDs are decoded back to their original strings
- Unknown IDs return empty strings instead of crashing
- Empty ID lists return empty strings
- Boolean values in ID lists are skipped
- `FileNotFoundError` is raised for missing vocab files
- `TypeError` is raised if the path is not a string

---

## TestBuildTrie — 5 tests

Verifies the `build_trie` function:
- An empty function list returns an empty trie
- A `None` tokenizer raises `ValueError`
- Valid functions are correctly indexed into the trie
- Invalid elements (non-`FunctionDefinition`) are skipped
- The original function name is stored in node metadata

---

## TestSelectFunction — 4 tests

Verifies the `select_function` function:
- An empty prompt returns `None`
- A `None` trie returns `None`
- An empty function list returns `None`
- The return value is always `str` or `None`, never another type

---

## TestGenerateArgument — 7 tests

Verifies the `generate_argument` function:
- An empty `param_type` returns `""`
- An empty prompt returns `0` for numbers and `True` for booleans
- The word `"false"` in the prompt returns `False` for booleans
- The default boolean value is `True` when `"false"` is absent
- A `None` model returns `0` for number types
- An unrecognized type returns an empty dict `{}`

---

## TestPydanticModels — 4 tests

Verifies the Pydantic data models:
- `FunctionDefinition` is created correctly with all fields
- Empty parameters default to an empty dict
- Missing required fields raise a validation error
- `FunctionCallResult` stores prompt, function name and arguments correctly