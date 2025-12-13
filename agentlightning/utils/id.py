# Copyright (c) Microsoft. All rights reserved.

import hashlib
import uuid

__all__ = ["generate_id"]


def generate_id(length: int) -> str:
    return hashlib.sha1(uuid.uuid4().bytes).hexdigest()[:length]
