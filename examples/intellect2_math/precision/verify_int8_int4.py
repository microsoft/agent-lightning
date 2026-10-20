# Copyright (c) Microsoft. All rights reserved.

import argparse
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from vllm import LLM, SamplingParams


def inspect_model(model):
    weights = Counter()
    linear = []
    attention = []
    recurrent = []
    for _, weight in model.named_parameters():
        weights[str(weight.dtype)] += weight.numel()
    for name, module in model.named_modules():
        scheme = getattr(module, "scheme", None)
        if scheme is not None and type(scheme).__name__ == "CompressedTensorsW8A8Int8":
            linear.append(
                dict(
                    name=name,
                    weight_dtype=str(module.weight.dtype),
                    kernel=type(scheme.kernel).__name__,
                    dynamic_activation=not scheme.is_static_input_scheme,
                )
            )
        if hasattr(module, "kv_cache_dtype"):
            attention.append(
                dict(
                    name=name,
                    dtype=module.kv_cache_dtype,
                    storage_dtype=str(getattr(module.kv_cache, "dtype", None)),
                    shape=list(module.kv_cache.shape),
                )
            )
        if type(module).__name__ == "GatedDeltaNetAttention":
            recurrent.append(dict(name=name, state_dtypes=list(map(str, module.get_state_dtype()))))
    return dict(weights=dict(weights), linear=linear, attention=attention, recurrent=recurrent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        quantization="compressed-tensors",
        kv_cache_dtype="int4_per_token_head",
        attention_config={"backend": "TRITON_ATTN"},
        kernel_config={"linear_backend": "triton"},
        language_model_only=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.35,
        max_model_len=4096,
        max_num_seqs=4,
        max_num_batched_tokens=1024,
        max_logprobs=128,
        enable_prefix_caching=False,
        enforce_eager=True,
        compilation_config={"mode": 0},
        seed=42,
    )
    tokenizer = llm.get_tokenizer()
    questions = [("17 + 25", "42"), ("13 * 7", "91"), ("144 / 12", "12"), ("123 - 58", "65")]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Calculate {question}. Output only the integer."}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for question, _ in questions
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64, logprobs=128))
    generations = [
        dict(
            question=question,
            expected=answer,
            text=output.outputs[0].text,
            logprobs_present=bool(output.outputs[0].logprobs),
        )
        for (question, answer), output in zip(questions, outputs, strict=True)
    ]
    second = llm.generate(prompts[:1], SamplingParams(temperature=0, max_tokens=64, logprobs=128))
    runtime = llm.apply_model(inspect_model)
    assert all(item["linear"] and item["attention"] for item in runtime)
    assert all(
        layer["weight_dtype"] == "torch.int8" and layer["dynamic_activation"]
        for item in runtime
        for layer in item["linear"]
    )
    assert all(layer["dtype"] == "int4_per_token_head" for item in runtime for layer in item["attention"])
    assert all(item["text"].strip() == item["expected"] and item["logprobs_present"] for item in generations)
    assert second[0].outputs[0].text.strip() == "42"
    Path(args.output).write_text(
        json.dumps(
            dict(
                status="passed",
                versions={name: version(name) for name in ["vllm", "torch", "transformers", "flashinfer-python"]},
                generations=generations,
                runtime=runtime,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
