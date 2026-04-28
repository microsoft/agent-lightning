# Copyright (c) Microsoft. All rights reserved.

import os

import requests
from openai import OpenAI

# Most common Azure OpenAI setup:
#   AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
#   AZURE_OPENAI_API_KEY="..."
# Optional (only if your endpoint requires it):
#   AZURE_OPENAI_API_VERSION="2025-xx-xx"
#
# This script treats "delete finetune job" as "cancel finetune job"
# because fine-tune jobs are typically cancellable, not deletable.


def _client() -> OpenAI:
    # This script assumes AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set in the environment.
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    return OpenAI(api_key=api_key, base_url=endpoint)


def list_data_files():
    c = _client()
    return c.files.list(limit=100)


def list_finetune_jobs():
    c = _client()
    return c.fine_tuning.jobs.list(limit=100)


def delete_data_file(file_id: str):
    c = _client()
    return c.files.delete(file_id)


def cancel_finetune_job(job_id: str):
    c = _client()
    return c.fine_tuning.jobs.cancel(job_id)


def delete_finetune_job(job_id: str):
    # This script assumes AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set in the environment.
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    root = endpoint.split("/openai")[0]

    url = f"{root}/openai/fine_tuning/jobs/{job_id}"
    params = {"api-version": os.environ["AZURE_OPENAI_API_VERSION"]}

    resp = requests.delete(url, headers={"api-key": api_key}, params=params, timeout=60)
    resp.raise_for_status()
    return resp.content


if __name__ == "__main__":
    # Quick demo: print IDs you could delete
    jobs = list_finetune_jobs().data
    files = list_data_files().data

    print("JOBS:")
    for j in jobs:
        print(f"  {j.id}  {getattr(j, 'status', '')}  {getattr(j, 'model', '')}")

    print("\nFILES:")
    for f in files:
        print(f"  {f.id}  {getattr(f, 'filename', '')}  {getattr(f, 'status', '')}")

    # Delete them all WITHOUT CONFIRMATION!
    for j in jobs:
        print(f"Deleting job {j.id}")
        try:
            if j.status == "running":
                cancel_finetune_job(j.id)
            delete_finetune_job(j.id)
        except Exception as exc:
            print(f"  Error deleting job {j.id}: {exc}")

    for f in files:
        print(f"Deleting file {f.id}")
        try:
            delete_data_file(f.id)
        except Exception as exc:
            print(f"  Error deleting file {f.id}: {exc}")
