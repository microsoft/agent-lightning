# test_attributes_flattening.py

import pytest

# Adjust this import to wherever your functions live:
# from your_module import flatten_attributes, unflatten_attributes
from agentlightning.emitter.utils import flatten_attributes, unflatten_attributes

# ---------
# flatten_attributes tests
# ---------


def test_flatten_simple_nested_dict_and_list():
    data = {"a": {"b": 1, "c": [2, 3]}}
    result = flatten_attributes(data)
    assert result == {
        "a.b": 1,
        "a.c.0": 2,
        "a.c.1": 3,
    }


def test_flatten_empty_dict():
    data = {}
    assert flatten_attributes(data) == {}


def test_flatten_empty_list():
    data = []
    # No elements -> no keys
    assert flatten_attributes(data) == {}


def test_flatten_root_list_of_primitives():
    data = [10, 20, 30]
    result = flatten_attributes(data)
    assert result == {
        "0": 10,
        "1": 20,
        "2": 30,
    }


def test_flatten_nested_lists_and_dicts():
    data = {
        "users": [
            {"name": "Alice", "tags": ["admin", "staff"]},
            {"name": "Bob", "tags": []},
        ]
    }
    result = flatten_attributes(data)
    assert result == {
        "users.0.name": "Alice",
        "users.0.tags.0": "admin",
        "users.0.tags.1": "staff",
        "users.1.name": "Bob",
        # Empty list yields no extra keys
    }


def test_flatten_mixed_types_and_none():
    data = {
        "a": True,
        "b": None,
        "c": 3.14,
        "d": "hello",
        "e": {"f": False},
    }
    result = flatten_attributes(data)
    assert result == {
        "a": True,
        "b": None,
        "c": 3.14,
        "d": "hello",
        "e.f": False,
    }


def test_flatten_non_string_key_raises_value_error():
    data = {
        "a": {
            1: "bad",  # non-string key inside nested dict
        }
    }
    with pytest.raises(ValueError) as excinfo:
        flatten_attributes(data)

    msg = str(excinfo.value)
    assert "Only string keys are supported in dictionaries" in msg
    # Ensure the offending key is mentioned
    assert "'1'" in msg
    assert "type <class 'int'>" in msg


def test_flatten_root_primitive_is_allowed():
    # Even though the type hint says Dict/List, function behavior supports primitives.
    data = 42
    result = flatten_attributes(data)  # type: ignore[arg-type]
    assert result == {"": 42}


# ---------
# unflatten_attributes tests
# ---------


def test_unflatten_simple_nested_dict():
    flat = {
        "a.b": 1,
        "a.c": 2,
    }
    result = unflatten_attributes(flat)
    assert result == {"a": {"b": 1, "c": 2}}


def test_unflatten_consecutive_numeric_keys_to_list():
    flat = {
        "a.0": "x",
        "a.1": "y",
        "a.2": "z",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": ["x", "y", "z"],
    }


def test_unflatten_non_consecutive_numeric_keys_stays_dict():
    flat = {
        "a.0": "first",
        "a.2": "third",
    }
    result = unflatten_attributes(flat)
    # Keys are numeric but not consecutive -> remains dict
    assert result == {
        "a": {
            "0": "first",
            "2": "third",
        }
    }


def test_unflatten_mixed_numeric_and_non_numeric_keys_stays_dict():
    flat = {
        "a.0": "zero",
        "a.foo": "bar",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": {
            "0": "zero",
            "foo": "bar",
        }
    }


def test_unflatten_root_list_from_numeric_keys():
    flat = {
        "0": "a",
        "1": "b",
        "2": "c",
    }
    result = unflatten_attributes(flat)
    # Root dict with all numeric keys 0..n-1 becomes list
    assert result == ["a", "b", "c"]


def test_unflatten_empty_flat_dict_returns_empty_dict():
    flat = {}
    result = unflatten_attributes(flat)
    assert result == {}


def test_unflatten_nested_lists_and_dicts():
    flat = {
        "users.0.name": "Alice",
        "users.0.tags.0": "admin",
        "users.0.tags.1": "staff",
        "users.1.name": "Bob",
        "users.1.tags.0": "guest",
    }
    result = unflatten_attributes(flat)
    assert result == {
        "users": [
            {"name": "Alice", "tags": ["admin", "staff"]},
            {"name": "Bob", "tags": ["guest"]},
        ]
    }


def test_unflatten_list_of_lists():
    flat = {
        "a.0.0": 1,
        "a.0.1": 2,
        "a.1.0": 3,
    }
    result = unflatten_attributes(flat)
    assert result == {
        "a": [
            [1, 2],
            [3],
        ]
    }


def test_unflatten_conflicting_primitive_and_nested_path_prefers_nested():
    # "a" is first set to a primitive, then to a nested dict via "a.b"
    flat = {
        "a": 1,
        "a.b": 2,
    }
    result = unflatten_attributes(flat)
    # Primitive is overwritten by nested dict structure
    assert result == {"a": {"b": 2}}


# ---------
# Round-trip / property tests
# ---------


@pytest.mark.parametrize(
    "value",
    [
        {"a": {"b": 1, "c": [2, 3]}},
        {"x": [1, 2, {"y": 3}]},
        {"root": [{"k": "v"}, {"k": "w"}]},
        [{"name": "Alice"}, {"name": "Bob", "scores": [10, 20]}],
    ],
)
def test_round_trip_flatten_then_unflatten_preserves_structure(value):
    flat = flatten_attributes(value)  # type: ignore[arg-type]
    reconstructed = unflatten_attributes(flat)
    assert reconstructed == value


@pytest.mark.parametrize(
    "flat",
    [
        {"a.b": 1, "a.c": 2},
        {"0": "x", "1": "y"},
        {
            "users.0.name": "Alice",
            "users.1.name": "Bob",
        },
    ],
)
def test_round_trip_unflatten_then_flatten_preserves_flat_structure(flat):
    nested = unflatten_attributes(flat)
    re_flat = flatten_attributes(nested)
    # Order of items in dict shouldn't matter
    assert re_flat == flat


def test_round_trip_with_empty_list_information_loss_is_expected():
    # This documents the corner case: empty list flattens to {},
    # which unflattens back to {} (empty dict), losing the distinction.
    data = []
    flat = flatten_attributes(data)
    assert flat == {}
    reconstructed = unflatten_attributes(flat)
    assert reconstructed == {}
    assert reconstructed != data  # explicit documentation of the behavior
