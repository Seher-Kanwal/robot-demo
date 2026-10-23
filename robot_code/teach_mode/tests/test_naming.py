import pytest

from naming import name_key, normalize_name


@pytest.mark.parametrize("raw, expected", [
    ("TV remote", "TV remote"),
    ("Um, it's a TV remote.", "TV remote"),
    ("it\u2019s a TV remote", "TV remote"),
    ("this is called the lotion bottle", "lotion bottle"),
    ("I think it's a hair dryer!", "hair dryer"),
    ("  A   Fan!! ", "Fan"),
    ("okay so that's my headphones", "headphones"),
    ("we call it a teddy bear", "teddy bear"),
    ("sofa", "sofa"),            # "so" is a filler only as a whole word
    ("itunes card", "itunes card"),
    ("", ""),
    ("   ...  ", ""),
])
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_normalize_name_caps_length():
    assert len(normalize_name("x" * 100)) == 40


def test_name_key_ignores_case_and_spacing():
    assert name_key("TV  Remote") == name_key("tv remote")
