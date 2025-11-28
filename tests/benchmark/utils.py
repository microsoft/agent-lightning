# Copyright (c) Microsoft. All rights reserved.

"""Generating random test data for benchmarking."""

import random
import string
from typing import Any, Callable, Dict, Optional, Tuple, Union, cast


def random_string(length: int, *, alphabet: Optional[str] = None) -> str:
    """
    Generate a random string of fixed length.

    Args:
        length: Length of the generated string.
        alphabet: Optional character set to draw from. If None, uses [A-Za-z0-9].
    """
    if length < 0:
        raise ValueError("String length cannot be negative.")

    alphabet = alphabet or (string.ascii_letters + string.digits)
    return "".join(random.choices(alphabet, k=length))


def _resolve_param(value: Union[int, Tuple[int, int]], name: str) -> int:
    """
    Convert parameter into a concrete integer.
    If value is an int, return it.
    If value is a tuple, interpret it as (low, high) and sample uniformly.
    """
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name} cannot be negative.")
        return value

    if (
        isinstance(value, tuple)  # type: ignore
        and len(value) == 2
        and isinstance(value[0], int)  # type: ignore
        and isinstance(value[1], int)  # type: ignore
    ):
        low, high = value
        if low < 0 or high < 0:
            raise ValueError(f"{name} range cannot contain negative values.")
        if low > high:
            raise ValueError(f"{name} tuple must be (low, high) with low <= high.")
        return random.randint(low, high)

    raise TypeError(f"{name} must be an int or a 2-tuple of ints.")


def default_value_factory(length: int) -> str:
    """Default value factory for generating string payloads."""
    return random_string(length)


def random_dict(
    *,
    depth: Union[int, Tuple[int, int]],
    breadth: Union[int, Tuple[int, int]],
    key_length: Union[int, Tuple[int, int]],
    value_length: Union[int, Tuple[int, int]],
    value_factory: Optional[Callable[[int], Any]] = None,
) -> Dict[str, Any]:
    """
    Generate a nested dictionary with configurable depth, breadth, and
    value sizes. Integer or (low, high) tuples are supported for
    all structural parameters.

    Args:
        depth: Number of nested levels or a tuple specifying a range.
        breadth: Number of keys per level or a tuple range.
        key_length: Length of each key or a tuple range.
        value_length: Length of each value or a tuple range.
        value_factory: Function mapping `value_length` → value.

    Returns:
        A nested dictionary of arbitrary size.
    """
    # Default factory
    if value_factory is None:
        value_factory = random_string

    def build(level: int) -> Dict[str, Any]:
        # For each level, breadth/key/value lengths may vary, so draw fresh each time
        current_breadth = _resolve_param(breadth, "breadth")

        if current_breadth < 0:
            raise ValueError("Breadth cannot be negative.")

        target_depth = depth if isinstance(depth, int) else _resolve_param((level, depth[1]), "depth")

        if level == target_depth:
            # leaf nodes
            return {
                random_string(_resolve_param(key_length, "key_length")): value_factory(
                    _resolve_param(value_length, "value_length")
                )
                for _ in range(current_breadth)
            }

        # nested nodes
        return {
            random_string(_resolve_param(key_length, "key_length")): build(level + 1) for _ in range(current_breadth)
        }

    return build(1)


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested dictionary into a single level dictionary. Keys are joined by dots."""

    result: Dict[str, Any] = {}
    for key, value in d.items():
        if isinstance(value, dict):
            result.update(flatten_dict(cast(Dict[str, Any], value), f"{prefix}.{key}" if prefix else key))
        else:
            result[f"{prefix}.{key}" if prefix else key] = value
    return result


if __name__ == "__main__":
    # Example usage
    import json

    structured_dict = random_dict(
        depth=(1, 3),
        breadth=(2, 6),
        key_length=(3, 20),
        value_length=(5, 300),
    )

    print(json.dumps(flatten_dict(structured_dict), indent=2))
