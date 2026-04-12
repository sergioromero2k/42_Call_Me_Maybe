#!/usr/bin/env python3
from re import sub


def fn_add_numbers(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def fn_greet(name: str) -> str:
    """Return a greeting message fot the given name."""
    return f"Hi {name}, How are you?"


def fn_reverse_string(s: str) -> str:
    """Return the reversed version of the input string."""
    return s[::-1]


def fn_get_square_root(a: int) -> float:
    """Return the square root of a number."""
    return float(a**(0.5))


def fn_substitute_string_with_regex(
        source_string: str, regex: str, replacement: str) -> str:
    """Replace matches of a regex pattern in a string."""
    return sub(regex, replacement, source_string)

# def fn_substitute_string_with_regex(
#         source_string: str, regex: str, replacement: str) -> str:
#     for i in source_string:
#         if i == regex:
#             source_string[i] = regex
#     return source_string
