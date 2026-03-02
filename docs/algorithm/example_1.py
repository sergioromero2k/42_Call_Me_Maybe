import random


def fake_model_next_token():
    tokens = [
        "{", "}", ":", ",",
        '"name"', '"age"',
        '"Alice"', '"Bob"',
        "25", "30"
    ]
    return random.choice(tokens)


print(fake_model_next_token()