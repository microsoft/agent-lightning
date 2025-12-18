# Copyright (c) Microsoft. All rights reserved.

# Copied and adapted from https://github.com/PeterGriffinJin/Search-R1/blob/main/search_r1/search/retrieval_server.py

import argparse
import json
import warnings
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import datasets
import faiss  # type: ignore[reportMissingTypeStubs]
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from numpy.typing import NDArray
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ---- Small helpers / aliases
Doc = Dict[str, Any]
Docs = List[Doc]
BatchDocs = List[Docs]
Scores = List[float]
BatchScores = List[Scores]


def load_corpus(corpus_path: str) -> Any:
    corpus: Any = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)  # type: ignore
    return corpus


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def load_docs(corpus: Any, doc_idxs: Sequence[int]) -> Docs:
    results: Docs = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False) -> Tuple[torch.nn.Module, Any]:
    # we call AutoConfig to ensure trust_remote_code init side-effects
    _model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
    model: torch.nn.Module = AutoModel.from_pretrained(model_path, trust_remote_code=True)  # type: ignore
    model.eval()  # type: ignore
    model.cuda()  # type: ignore
    if use_fp16:
        model = model.half()  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)  # type: ignore
    return model, tokenizer  # type: ignore


def pooling(
    pooler_output: torch.Tensor,
    last_hidden_state: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    pooling_method: str = "mean",
) -> torch.Tensor:
    if pooling_method == "mean":
        assert attention_mask is not None, "attention_mask is required for mean pooling"
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
    ) -> None:
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()  # type: ignore
    def encode(self, query_list: Union[List[str], str], is_query: bool = True) -> NDArray[np.float32]:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}" for query in query_list
                ]

        inputs: Dict[str, torch.Tensor] = self.tokenizer(
            query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt"
        )  # type: ignore[call-arg]
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(
                inputs["input_ids"].device
            )
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_np: NDArray[np.float32] = query_emb.detach().cpu().numpy().astype(np.float32, order="C")  # type: ignore

        # cleanup
        del inputs, output
        torch.cuda.empty_cache()

        return query_np


class Config:
    """
    Minimal config class (simulating your argparse)
    Replace this with your real arguments or load them dynamically.
    """

    def __init__(
        self,
        retrieval_method: str = "bm25",
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128,
    ) -> None:
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class BaseRetriever:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.retrieval_method: str = config.retrieval_method
        self.topk: int = config.retrieval_topk

        self.index_path: str = config.index_path
        self.corpus_path: str = config.corpus_path

    def _search(self, query: str, num: Optional[int], return_score: bool) -> Union[Docs, Tuple[Docs, Scores]]:
        raise NotImplementedError

    def _batch_search(
        self, query_list: List[str], num: Optional[int], return_score: bool
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        raise NotImplementedError

    def search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        return self._search(query, num, return_score)

    def batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # import locally to avoid hard dependency at import/type time
        try:
            from pyserini.search.lucene import LuceneSearcher  # type: ignore
        except Exception:  # pragma: no cover - typing convenience
            LuceneSearcher = Any  # type: ignore[assignment]

        self.searcher: Any = LuceneSearcher(self.index_path)  # type: ignore
        self.contain_doc: bool = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus: Any = load_corpus(self.corpus_path)
        self.max_process_num: int = 8

    def _check_contain_doc(self) -> bool:
        doc = self.searcher.doc(0)
        try:
            _ = doc.raw()
            return True
        except Exception:
            return False

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        hits: List[Any] = self.searcher.search(query, k)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores: Scores = [float(hit.score) for hit in hits]
        if len(hits) < k:
            warnings.warn("Not enough documents retrieved!")
        else:
            hits = hits[:k]

        if self.contain_doc:
            all_contents: List[str] = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results: Docs = [
                {
                    "title": content.split("\n")[0].strip('"'),
                    "text": "\n".join(content.split("\n")[1:]),
                    "contents": content,
                }
                for content in all_contents
            ]
        else:
            results = load_docs(self.corpus, [int(hit.docid) for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        results: BatchDocs = []
        scores: BatchScores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)  # type: ignore[misc]
            results.append(item_result)  # type: ignore[arg-type]
            scores.append(item_score)  # type: ignore[arg-type]
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        index: Any = faiss.read_index(self.index_path)  # type: ignore[no-untyped-call]
        if config.faiss_gpu:
            # Some faiss GPU helpers may be missing type stubs; treat as Any.
            co: Any = faiss.GpuMultipleClonerOptions()  # type: ignore[attr-defined]
            co.useFloat16 = True
            co.shard = True
            index = faiss.index_cpu_to_all_gpus(index, co=co)  # type: ignore[no-untyped-call]

        self.index: Any = index
        self.corpus: Any = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ) -> Union[Docs, Tuple[Docs, Scores]]:
        k = self.topk if num is None else num
        query_emb = self.encoder.encode(query)
        scores_np, idxs_np = self.index.search(query_emb, k=k)  # type: ignore[no-untyped-call]
        idxs: Sequence[int] = list(map(int, idxs_np[0]))
        scores: Scores = [float(s) for s in scores_np[0]]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(
        self, query_list: List[str], num: Optional[int] = None, return_score: bool = False
    ) -> Union[BatchDocs, Tuple[BatchDocs, BatchScores]]:
        if isinstance(query_list, str):
            query_list = [query_list]
        k = self.topk if num is None else num

        results: BatchDocs = []
        scores: BatchScores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process: "):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores_np, batch_idxs_np = self.index.search(batch_emb, k=k)  # type: ignore[no-untyped-call]
            batch_scores = batch_scores_np.tolist()
            batch_idxs = batch_idxs_np.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs: List[int] = sum(batch_idxs, [])  # type: ignore
            batch_results_flat = load_docs(self.corpus, flat_idxs)
            # chunk them back
            chunked: List[Docs] = [batch_results_flat[i * k : (i + 1) * k] for i in range(len(batch_idxs))]

            results.extend(chunked)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results_flat
            torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config: Config) -> BaseRetriever:
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server below
#####################################


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app: FastAPI = FastAPI()

# Globals created under __main__; keep typed placeholders for pyright
config: Config  # will be set in __main__
retriever: BaseRetriever  # will be set in __main__


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest) -> Dict[str, Any]:
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    search_out = retriever.batch_search(
        query_list=request.queries, num=int(request.topk), return_score=request.return_scores
    )

    # Unpack depending on return_scores
    if request.return_scores:
        results, scores = search_out  # type: ignore[misc]
    else:
        results = search_out  # type: ignore[assignment]
        scores = None

    # Format response
    resp: List[Any] = []
    for i, single_result in enumerate(results):  # type: ignore[arg-type]
        if request.return_scores and scores is not None:
            combined: List[Dict[str, Any]] = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": float(score)})
            resp.append(combined)
        else:
            resp.append(single_result)
    return {"result": resp}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch the local faiss retriever.")
    parser.add_argument(
        "--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file."
    )
    parser.add_argument(
        "--corpus_path",
        type=str,
        default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl",
        help="Local corpus file.",
    )
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument(
        "--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model."
    )
    parser.add_argument("--faiss_gpu", action="store_true", help="Use GPU for computation")

    args = parser.parse_args()

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=int(args.topk),
        faiss_gpu=bool(args.faiss_gpu),
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
