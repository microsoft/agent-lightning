# Copyright (c) Microsoft. All rights reserved.

"""Helper functions to make pydantic more useful."""

from typing import Any, List

from pydantic import ValidationError


def to_plain_object(object: Any, path: List[str]) -> Any:
    if type(object).__name__ == "ValidatorIterator":
        try:
            return [to_plain_object(item, path + [str(i)]) for i, item in enumerate(object)]
        except ValidationError as exc:
            raise ValueError(
                "Failed to convert ValidatorIterator to list.\n"
                "ValidatorIterator path (see below for subpath): " + ".".join(path) + "\nError: " + str(exc)
            ) from exc
    elif isinstance(object, dict):
        return {k: to_plain_object(v, path + [k]) for k, v in object.items()}  # type: ignore
    elif isinstance(object, list):
        return [to_plain_object(item, path + [str(i)]) for i, item in enumerate(object)]  # type: ignore
    elif isinstance(object, tuple):
        return tuple(to_plain_object(item, path + [str(i)]) for i, item in enumerate(object))  # type: ignore
    else:
        return object
