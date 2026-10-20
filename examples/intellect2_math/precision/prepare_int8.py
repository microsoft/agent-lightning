# Copyright (c) Microsoft. All rights reserved.

import argparse
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForImageTextToText, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = args.model if Path(args.model).is_dir() else snapshot_download(args.model)
    model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, device_map="cuda")
    oneshot(
        model=model,
        processor=AutoTokenizer.from_pretrained(source),
        recipe=QuantizationModifier(targets="Linear", scheme="W8A8", ignore=["lm_head", "re:model.visual.*"]),
        output_dir=args.output,
    )
    AutoTokenizer.from_pretrained(source).save_pretrained(args.output)


if __name__ == "__main__":
    main()
