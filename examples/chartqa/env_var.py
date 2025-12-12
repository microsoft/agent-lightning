# Copyright (c) Microsoft. All rights reserved.

import os

__all__ = [
    "CHARTQA_ROOT_DIR",
    "CHARTQA_DATA_DIR",
    "CHARTQA_IMAGES_DIR",
    "USE_BASE64_IMAGES",
    "USE_LLM_PROXY",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
]

CHARTQA_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

CHARTQA_DATA_DIR = os.getenv("CHARTQA_DATA_DIR", os.path.realpath(os.path.join(CHARTQA_ROOT_DIR, "data")))

CHARTQA_IMAGES_DIR = os.getenv("CHARTQA_IMAGES_DIR", os.path.realpath(os.path.join(CHARTQA_ROOT_DIR, "data", "images")))

USE_BASE64_IMAGES = os.getenv("USE_BASE64_IMAGES", "false").lower() in ("1", "true", "yes")

USE_LLM_PROXY = os.getenv("USE_LLM_PROXY", "false").lower() in ("1", "true", "yes")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "token-abc123")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
