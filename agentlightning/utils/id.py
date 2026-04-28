# Copyright (c) Microsoft. All rights reserved.

import hashlib
import uuid

__all__ = ["generate_id"]


def generate_id(length: int) -> str:
    """Generate a random hexadecimal ID of the given length.

    Args:
        length: The desired length of the generated ID. Must be a positive integer.

    Returns:
        A random hexadecimal ID string of the given length.

    Raises:
        ValueError: If length is not a positive integer.
    """
    if length <= 0:
        raise ValueError("length must be a positive integer")

    return hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:length]
