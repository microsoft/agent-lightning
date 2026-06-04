# Copyright (c) Microsoft. All rights reserved.

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Any

import datasets
import faiss  # type: ignore[reportMissingTypeStubs]
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from numpy.typing import NDArray
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    queries: list[str]
    topk: int | None = None
    return_scores: bool = True


@dataclass
class RetrievalConfig:
    index_path: str
    corpus_path: str
    topk: int
    retriever_name: str
    retriever_model: str
    device: str
    faiss_gpu: bool
    query_max_length: int
    batch_size: int
    use_fp16: bool


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return masked.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class DenseEncoder:
    def __init__(self, config: RetrievalConfig) -> None:
        self.model_name = config.retriever_name
        self.max_length = config.query_max_length
        self.device = _resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.retriever_model, use_fast=True, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(config.retriever_model, trust_remote_code=True).to(self.device)
        self.model.eval()
        if config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, queries: list[str]) -> NDArray[np.float32]:
        if "e5" in self.model_name.lower():
            queries = [f"query: {query}" for query in queries]

        inputs = self.tokenizer(
            queries,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output = self.model(**inputs, return_dict=True)
        pooled = _pool(output.last_hidden_state, inputs["attention_mask"])
        embeddings = torch.nn.functional.normalize(pooled, dim=-1)
        return embeddings.detach().cpu().numpy().astype(np.float32, order="C")


def _load_corpus(corpus_path: str) -> Any:
    return datasets.load_dataset("json", data_files=corpus_path, split="train")


def _doc_from_row(row: dict[str, Any]) -> dict[str, str]:
    contents = str(row.get("contents", ""))
    if contents:
        title, _, text = contents.partition("\n")
        return {"title": title.strip('"'), "text": text, "contents": contents}
    title = str(row.get("title", ""))
    text = str(row.get("text", ""))
    joined = f"{title}\n{text}" if title else text
    return {"title": title, "text": text, "contents": joined}


class DenseRetriever:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.topk = config.topk
        self.batch_size = config.batch_size
        self.index = faiss.read_index(config.index_path)
        if config.faiss_gpu:
            self.index = self._try_move_index_to_gpu(self.index)
        self.corpus = _load_corpus(config.corpus_path)
        self.encoder = DenseEncoder(config)

    @staticmethod
    def _try_move_index_to_gpu(index: Any) -> Any:
        if not torch.cuda.is_available():
            log.warning("--faiss-gpu requested but CUDA is unavailable; using CPU FAISS index")
            return index
        try:
            options = faiss.GpuMultipleClonerOptions()
            options.useFloat16 = True
            options.shard = True
            return faiss.index_cpu_to_all_gpus(index, co=options)
        except AttributeError:
            log.warning("Installed faiss package has no GPU helpers; using CPU FAISS index")
            return index

    def batch_search(self, queries: list[str], topk: int, return_scores: bool) -> list[list[dict[str, Any]]]:
        responses: list[list[dict[str, Any]]] = []
        for start in range(0, len(queries), self.batch_size):
            batch = queries[start : start + self.batch_size]
            embeddings = self.encoder.encode(batch)
            scores, indexes = self.index.search(embeddings, topk)
            for batch_indexes, batch_scores in zip(indexes, scores, strict=True):
                items: list[dict[str, Any]] = []
                for doc_idx, score in zip(batch_indexes, batch_scores, strict=True):
                    if int(doc_idx) < 0:
                        continue
                    item: dict[str, Any] = {"document": _doc_from_row(dict(self.corpus[int(doc_idx)]))}
                    if return_scores:
                        item["score"] = float(score)
                    items.append(item)
                responses.append(items)
        return responses


app = FastAPI(title="Search-R1 retrieval server")
retriever: DenseRetriever | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/retrieve")
def retrieve(request: QueryRequest) -> dict[str, Any]:
    if retriever is None:
        raise RuntimeError("retriever is not initialized")
    topk = request.topk or retriever.topk
    return {"result": retriever.batch_search(request.queries, topk, request.return_scores)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a Search-R1 compatible dense retrieval endpoint.")
    parser.add_argument("--index-path", default="data/e5_Flat.index", help="FAISS index path")
    parser.add_argument("--corpus-path", default="data/wiki-18.jsonl", help="Wikipedia JSONL corpus path")
    parser.add_argument("--topk", type=int, default=3, help="Default passages per query")
    parser.add_argument("--retriever-name", default="e5", help="Retriever family name")
    parser.add_argument("--retriever-model", default="intfloat/e5-base-v2", help="HF retriever model id/path")
    parser.add_argument("--device", default="auto", help="Torch device: auto, cuda, cuda:0, or cpu")
    parser.add_argument("--faiss-gpu", action="store_true", help="Move FAISS index to all GPUs when possible")
    parser.add_argument("--query-max-length", type=int, default=256, help="Retriever query max length")
    parser.add_argument("--batch-size", type=int, default=512, help="Retriever encoding batch size")
    parser.add_argument("--use-fp16", action="store_true", help="Use fp16 retriever model on CUDA")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = RetrievalConfig(
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        topk=args.topk,
        retriever_name=args.retriever_name,
        retriever_model=args.retriever_model,
        device=args.device,
        faiss_gpu=args.faiss_gpu,
        query_max_length=args.query_max_length,
        batch_size=args.batch_size,
        use_fp16=args.use_fp16,
    )
    global retriever
    retriever = DenseRetriever(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
