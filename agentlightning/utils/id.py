# Copyright (c) Microsoft. All rights reserved.

import hashlib
import uuid

__all__ = ["generate_id"]


def generate_id(length: int) -> str:
    """Generate a random ID of the given length.

    Args:
        length: The length of the ID to generate.

    Returns:
        A random ID of the given length.
    """
    return hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:length]
