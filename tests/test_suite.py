#!/usr/bin/env python3
# uv add pytest --dev, uv run pytest tests/ -v
"""
Comprehensive test suite for the constrained decoding engine.
Tests cover build_trie, select_function, generate_argument,
CustomTokenizer, FunctionTrie, and Pydantic models.
"""

import pytest
from unittest.mock import MagicMock
from src.trie import FunctionTrie, TrieNode
from src.tokenizer import CustomTokenizer
from src.models import FunctionDefinition, FunctionCallResult
from src.constrained_dec import build_trie, select_function, generate_argument


# ─────────────────────────────────────────────
# Fake tokenizer class (avoids __code__ issues)
# ─────────────────────────────────────────────

class FakeTokenizer:
    """Simple tokenizer that maps each char to its ASCII value."""

    def encode(self, text):
        if not isinstance(text, str):
            return []
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids if isinstance(i, int))


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture
def simple_functions():
    return [
        FunctionDefinition(
            name="fn_add",
            description="Add two numbers.",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}}
        ),
        FunctionDefinition(
            name="fn_greet",
            description="Greet a person by name.",
            parameters={"name": {"type": "string"}}
        ),
    ]


@pytest.fixture
def fake_tokenizer():
    return FakeTokenizer()


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.get_logits_from_input_ids = MagicMock(
        return_value=[0.1] * 200000
    )
    return model


# ─────────────────────────────────────────────
# TrieNode tests
# ─────────────────────────────────────────────

class TestTrieNode:
    def test_default_values(self):
        node = TrieNode()
        assert node.children == {}
        assert node.is_end_of_path is False
        assert node.meta == {}

    def test_children_type(self):
        node = TrieNode()
        child = TrieNode()
        node.children[42] = child
        assert isinstance(node.children[42], TrieNode)


# ─────────────────────────────────────────────
# FunctionTrie tests
# ─────────────────────────────────────────────

class TestFunctionTrie:
    def test_empty_trie_has_root(self):
        trie = FunctionTrie()
        assert trie.root is not None
        assert isinstance(trie.root, TrieNode)

    def test_insert_single_sequence(self):
        trie = FunctionTrie()
        trie.insert([1, 2, 3], meta_data={"fn_name": "fn_test"})
        node = trie.root.children[1].children[2].children[3]
        assert node.is_end_of_path is True
        assert node.meta["fn_name"] == "fn_test"

    def test_insert_shared_prefix(self):
        trie = FunctionTrie()
        trie.insert([1, 2, 3])
        trie.insert([1, 2, 4])
        assert 3 in trie.root.children[1].children[2].children
        assert 4 in trie.root.children[1].children[2].children

    def test_insert_empty_list(self):
        trie = FunctionTrie()
        trie.insert([])
        assert trie.root.children == {}

    def test_insert_invalid_type(self):
        trie = FunctionTrie()
        trie.insert("not a list")  # type: ignore
        assert trie.root.children == {}

    def test_insert_bool_rejected(self):
        trie = FunctionTrie()
        trie.insert([True, 2, 3])  # type: ignore
        assert trie.root.children == {}

    def test_get_valid_next_tokens(self):
        trie = FunctionTrie()
        trie.insert([10, 20])
        trie.insert([10, 30])
        tokens = trie.get_valid_next_tokens(trie.root.children[10])
        assert set(tokens) == {20, 30}

    def test_get_valid_next_tokens_none_node(self):
        trie = FunctionTrie()
        result = trie.get_valid_next_tokens(None)  # type: ignore
        assert result == []


# ─────────────────────────────────────────────
# CustomTokenizer tests
# ─────────────────────────────────────────────

class TestCustomTokenizer:
    @pytest.fixture
    def vocab_file(self, tmp_path):
        import json
        vocab = {"hello": 1, "world": 2, " ": 3, "h": 4, "e": 5}
        path = tmp_path / "vocab.json"
        path.write_text(json.dumps(vocab))
        return str(path)

    def test_loads_vocab(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        assert "hello" in tok.vocab
        assert 1 in tok.inverse_vocab

    def test_encode_known_token(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        result = tok.encode("hello")
        assert result == [1]

    def test_encode_empty_string(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        assert tok.encode("") == []

    def test_encode_none_returns_empty(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        assert tok.encode(None) == []  # type: ignore

    def test_decode_known_ids(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        result = tok.decode([1, 3, 2])
        assert result == "hello world"

    def test_decode_unknown_id(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        result = tok.decode([99999])
        assert result == ""

    def test_decode_empty_list(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        assert tok.decode([]) == ""

    def test_decode_bool_skipped(self, vocab_file):
        tok = CustomTokenizer(vocab_file)
        result = tok.decode([True])  # type: ignore
        assert result == ""

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            CustomTokenizer("/nonexistent/path/vocab.json")

    def test_invalid_path_type(self):
        with pytest.raises(TypeError):
            CustomTokenizer(123)  # type: ignore


# ─────────────────────────────────────────────
# build_trie tests
# ─────────────────────────────────────────────

class TestBuildTrie:
    def test_empty_functions_returns_empty_trie(self, fake_tokenizer):
        trie = build_trie([], fake_tokenizer)
        assert trie.root.children == {}

    def test_none_tokenizer_raises(self, simple_functions):
        with pytest.raises(ValueError):
            build_trie(simple_functions, None)

    def test_builds_trie_with_valid_functions(
            self, simple_functions, fake_tokenizer):
        trie = build_trie(simple_functions, fake_tokenizer)
        assert trie.root.children != {}

    def test_invalid_element_skipped(self, fake_tokenizer):
        functions = ["not_a_function"]  # type: ignore
        trie = build_trie(functions, fake_tokenizer)
        assert trie.root.children == {}

    def test_function_name_stored_in_meta(
            self, simple_functions, fake_tokenizer):
        trie = build_trie(simple_functions, fake_tokenizer)
        found = False

        def walk(node):
            nonlocal found
            if node.is_end_of_path and "fn_name" in node.meta:
                found = True
            for child in node.children.values():
                walk(child)

        walk(trie.root)
        assert found


# ─────────────────────────────────────────────
# select_function tests
# ─────────────────────────────────────────────

class TestSelectFunction:
    def test_empty_prompt_returns_none(
            self, mock_model, fake_tokenizer, simple_functions):
        trie = build_trie(simple_functions, fake_tokenizer)
        result = select_function(
            "", mock_model, fake_tokenizer, trie,
            functions=simple_functions)
        assert result is None

    def test_none_trie_returns_none(self, mock_model, fake_tokenizer):
        result = select_function(
            "hello", mock_model, fake_tokenizer, None)
        assert result is None

    def test_no_functions_returns_none(
            self, mock_model, fake_tokenizer, simple_functions):
        trie = build_trie(simple_functions, fake_tokenizer)
        result = select_function(
            "hello", mock_model, fake_tokenizer, trie,
            functions=[])
        assert result is None

    def test_returns_string_or_none(
            self, mock_model, fake_tokenizer, simple_functions):
        trie = build_trie(simple_functions, fake_tokenizer)
        result = select_function(
            "add two numbers", mock_model, fake_tokenizer, trie,
            functions=simple_functions)
        assert result is None or isinstance(result, str)


# ─────────────────────────────────────────────
# generate_argument tests
# ─────────────────────────────────────────────

class TestGenerateArgument:
    def test_empty_param_type_returns_empty(
            self, mock_model, fake_tokenizer):
        result = generate_argument(
            "hello", "", mock_model, fake_tokenizer)
        assert result == ""

    def test_empty_prompt_number_returns_zero(
            self, mock_model, fake_tokenizer):
        result = generate_argument(
            "", "number", mock_model, fake_tokenizer)
        assert result == 0

    def test_empty_prompt_boolean_returns_true(
            self, mock_model, fake_tokenizer):
        result = generate_argument(
            "", "boolean", mock_model, fake_tokenizer)
        assert result is True

    def test_boolean_false_detected(self, mock_model, fake_tokenizer):
        result = generate_argument(
            "set active to false", "boolean", mock_model, fake_tokenizer)
        assert result is False

    def test_boolean_true_by_default(self, mock_model, fake_tokenizer):
        result = generate_argument(
            "enable notifications", "boolean", mock_model, fake_tokenizer)
        assert result is True

    def test_none_model_number_returns_zero(self, fake_tokenizer):
        result = generate_argument(
            "give me 5", "number", None, fake_tokenizer)
        assert result == 0

    def test_unknown_type_returns_empty_dict(
            self, mock_model, fake_tokenizer):
        result = generate_argument(
            "hello", "unknown_type", mock_model, fake_tokenizer)
        assert result == {}


# ─────────────────────────────────────────────
# Pydantic model tests
# ─────────────────────────────────────────────

class TestPydanticModels:
    def test_function_definition_valid(self):
        fn = FunctionDefinition(
            name="fn_test",
            description="A test function.",
            parameters={"x": {"type": "number"}}
        )
        assert fn.name == "fn_test"

    def test_function_definition_empty_parameters(self):
        fn = FunctionDefinition(
            name="fn_test",
            description="No params.",
        )
        assert fn.parameters == {}

    def test_function_definition_missing_name(self):
        with pytest.raises(Exception):
            FunctionDefinition(description="Missing name.")  # type: ignore

    def test_function_call_result_valid(self):
        result = FunctionCallResult(
            prompt="add 1 and 2",
            fn_name="fn_add",
            args={"a": 1, "b": 2}
        )
        assert result.fn_name == "fn_add"
        assert result.args["a"] == 1