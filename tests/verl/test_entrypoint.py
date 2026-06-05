from __future__ import annotations

from agl_lite.verl.entrypoint import _copy_worker_env


def test_copy_worker_env_includes_llm_sandbox_hook_vars() -> None:
    target = {"AGL_MODEL_NAME": "existing-model"}

    _copy_worker_env(
        {
            "AGL_KEY": "agent-key",
            "AGL_BASE_URL": "http://agl",
            "AGL_MODEL_ENDPOINT": "http://vllm/v1",
            "AGL_NAMESPACE": "agl-ns",
            "AGL_HOOKS": "examples/llm-in-sandbox/hooks.py",
            "AGL_POD_SPEC_TEMPLATE": "examples/llm-in-sandbox/job-template.yaml",
            "AGL_MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
            "AGL_OPENAI_MODEL_PREFIX": "openai/",
            "AGL_LLM_TEMPERATURE": "1.0",
            "OPENAI_TIMEOUT": "900",
            "MAX_TOKENS_PER_CALL": "20000",
            "HF_TOKEN": "hf-token",
            "WANDB_API_KEY": "wandb-key",
            "WANDB_ENTITY": "entity",
            "WANDB_PROJECT": "project",
            "WANDB_DIR": "/tmp/wandb",
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": "run-id",
            "WANDB_RESUME": "allow",
        },
        target,
    )

    assert target == {
        "AGL_KEY": "agent-key",
        "AGL_BASE_URL": "http://agl",
        "AGL_MODEL_ENDPOINT": "http://vllm/v1",
        "AGL_NAMESPACE": "agl-ns",
        "AGL_HOOKS": "examples/llm-in-sandbox/hooks.py",
        "AGL_POD_SPEC_TEMPLATE": "examples/llm-in-sandbox/job-template.yaml",
        "AGL_MODEL_NAME": "Qwen/Qwen3-4B-Instruct-2507",
        "AGL_OPENAI_MODEL_PREFIX": "openai/",
        "AGL_LLM_TEMPERATURE": "1.0",
        "OPENAI_TIMEOUT": "900",
        "MAX_TOKENS_PER_CALL": "20000",
        "HF_TOKEN": "hf-token",
        "WANDB_API_KEY": "wandb-key",
        "WANDB_ENTITY": "entity",
        "WANDB_PROJECT": "project",
        "WANDB_DIR": "/tmp/wandb",
        "WANDB_MODE": "offline",
        "WANDB_RUN_ID": "run-id",
        "WANDB_RESUME": "allow",
    }


def test_copy_worker_env_ignores_empty_values() -> None:
    target = {"AGL_POD_SPEC_TEMPLATE": "keep-existing"}

    _copy_worker_env({"AGL_POD_SPEC_TEMPLATE": "", "AGL_MODEL_NAME": "model"}, target)

    assert target == {"AGL_POD_SPEC_TEMPLATE": "keep-existing", "AGL_MODEL_NAME": "model"}
