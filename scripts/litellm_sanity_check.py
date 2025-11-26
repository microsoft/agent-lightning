# Copyright (c) Microsoft. All rights reserved.

"""Utility script to perform a sanity check on LiteLLM proxy server."""

import sys

import openai


def main() -> None:
    client = openai.OpenAI(timeout=30.0)
    models = client.models.list()
    print("Available models:", models)

    total_requests = 0
    success_count = 0

    for model in models.data:
        try:
            total_requests += 1
            response = client.chat.completions.create(
                model=model.id,
                messages=[{"role": "user", "content": "Hello!"}],
            )
            print(f"Chat completion from model {model.id}:", response)
            success_count += 1
        except Exception as e:
            print(f"Chat completion failed for model {model.id}: {e}")

        try:
            total_requests += 1
            response = client.responses.create(
                model=model.id,
                input="Hello, world!",
            )
            print(f"Response from model {model.id}:", response)
            success_count += 1
        except Exception as e:
            print(f"Response failed for model {model.id}: {e}")

    if total_requests == 0:
        print("No requests made.")
        sys.exit(1)

    success_rate = success_count / total_requests
    print(f"Success rate: {success_rate * 100:.2f}% ({success_count}/{total_requests})")

    if success_rate >= 0.8:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
