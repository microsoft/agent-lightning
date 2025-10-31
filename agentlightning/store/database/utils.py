# Copyright (c) Microsoft. All rights reserved.
"""This file contains utility functions for database operations.
"""

from __future__ import annotations

from typing import Any
import tenacity

__retry_config__: dict[str, Any] = {
    "default": {
        "wait": {
            "_type": "wait_fixed",  # corresponds to tenacity.wait_fixed
            "_args": [1000],  # wait 1000 milliseconds between retries
            "_kwargs": {},
        }
    }
}

def register_retry_config(name: str, config: dict[str, dict[str, Any]]) -> None:
    """Register a retry configuration for database operations.
    Args:
        name: The name of the retry configuration.
        config: A dictionary containing tenacity retry parameters.
    Example:
        register_retry_config("my_config", {
            "wait": {
                "_type": "wait_fixed", # corresponds to tenacity.wait_fixed
                "_args": [2],  # wait 2 seconds between retries
                "_kwargs": {},
            },
            "stop": {
                "_type": "stop_after_attempt",
                "_args": [5],  # stop after 5 attempts
                "_kwargs": {},
            },
        })
    """
    dic = {} # deserialized config
    for key, item in config.items():
        _type = item["_type"]
        _args = item.get("_args", [])
        _kwargs = item.get("_kwargs", {})
        tenacity_fn = getattr(tenacity, _type)
        dic[key] = tenacity_fn(*_args, **_kwargs)
    __retry_config__[name] = dic


class ConfigurableRetry:
    def __init__(self, config_key: str, **kwargs: Any) -> None:
        # In a real application, you would load this from a global config store
        self.config = __retry_config__.get(config_key, __retry_config__["default"])
        self.config.update(kwargs)

    def __call__(self, fn: function) -> function:
        # Return the actual tenacity decorator, configured dynamically
        return tenacity.retry(**self.config)(fn)



