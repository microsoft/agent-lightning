# RAG Agent Example

[![rag workflow status](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml/badge.svg)](https://github.com/microsoft/agent-lightning/actions/workflows/examples-rag.yml)

This example demonstrates training a Retrieval-Augmented Generation (RAG) agent using Agent-Lightning with retrieval capabilities. The agent answers multi-hop questions from a tiny MuSiQue dataset by retrieving and reasoning over Wikipedia passages.

## Overview

This example can run on a single GPU for demonstration purposes.

**Step 1:** Set up the environment. It is recommended to setup with uv and activate the virtual environment with:

```bash
uv sync --frozen --extra apo --group agents --group torch-gpu-stable --extra verl --group rag
source .venv/bin/activate
```

**Step 2:** Prepare the tiny dataset.

```bash
pip install gdown

# tiny training dataset
cd examples/rag
gdown --fuzzy "https://drive.google.com/file/d/1Pq4Ag8zVoN8gUtLu0LcBfY35Dm5zL0hq/view?usp=drive_link" \
    -O dataset_tiny.parquet

# chunks_candidate_tiny.pkl
gdown --fuzzy "https://drive.google.com/file/d/1REXCpRLbeZu1KfWWKhIGEQe_WNHUOBkS/view?usp=drive_link" \
    -O chunks_candidate_tiny.pkl

# index_hnsw_faiss_n32e40_tiny.index
gdown --fuzzy "https://drive.google.com/file/d/1f6P-h_8KSRhe5pqDHWbRQWvUhTygfZ-c/view?usp=drive_link" \
    -O index_hnsw_faiss_n32e40_tiny.index
```

**Step 3:** Start the MCP server. Open a terminal and run:

```bash
python wiki_retriever_mcp.py
```

**Step 4:** Start training. Open another terminal and run:

```bash
python train_rag.py
```

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `rag_agent.py` | RAG agent example using the OpenAI Agents SDK, with debugging utils |
| `train_rag.py` | Initiates the GRPO training process |
| `metric_utils.py` | Scoring utilities for exact match, F1 score, and response parsing |
| `wiki_retriever_mcp.py` | MCP server for Wikipedia retrieval |

## How to Prepare the Retrieval Corpus Yourself

To enable semantic retrieval with this MCP server, you need two files:

1. **FAISS index file** (`.index`)
2. **Chunk list file** (`.pkl`)

These two files work together: the FAISS index stores the vector embeddings and their mapping to integer IDs, while the pickle file stores the actual text chunks. The integer IDs in the index correspond exactly to the positions in the chunk list.

### Step 1: Collecting Text Chunks

First, you need a collection of text passages (chunks). For example, you can download a Wikipedia-based dataset such as `wiki18_100w.zip` from the [FlashRAG_dataset](https://huggingface.co/datasets/FlashRAG) or use other pre-split corpora.

### Step 2: Creating the FAISS Index (`nq_hnsw_faiss_n32e40.index`)

- Use a sentence embedding model (e.g., `BAAI/bge-large-en-v1.5`) to encode each chunk into a vector.
- Build a FAISS index from these vectors.
- In this example, we use an **HNSW index** (Hierarchical Navigable Small World graph), which supports efficient approximate nearest-neighbor search.
- The index stores only embeddings and integer IDs (no raw text).

### Step 3: Creating the Chunk List (`nq_list.pkl`)

- Store the raw text chunks in a Python list.
- Save this list with `pickle`.
- The index ID returned by FAISS corresponds to the list index in this file. For example, if FAISS search returns `I[0][i] = 12345`, then the corresponding text chunk is `chunks[12345]`.

### Example Schema

- **`nq_hnsw_faiss_n32e40.index`**
  - Type: FAISS HNSW index
  - Contains:
    - Vector embeddings
    - Graph structure for fast search
    - Integer IDs mapping to chunk positions

- **`nq_list.pkl`**
  - Type: Pickled Python list
  - Element type: string (or dict with text + metadata, depending on preprocessing)
  - Example:
    ```python
    [
        "The Eiffel Tower is located in Paris, France.",
        "Albert Einstein developed the theory of relativity.",
        ...
    ]
    ```

### Step 4: Code Example - Building Index and Chunk List

**Warning:** The following example demonstrates a small-scale workflow only. In practice, for large datasets, you should encode the text in batches and incrementally add them to the index.

```python
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# 1. Prepare your text chunks (list of strings)
chunk_texts = [
    "The Eiffel Tower is located in Paris, France.",
    "Albert Einstein developed the theory of relativity.",
    "Python is a popular programming language.",
    # ... more chunks
]

# 2. Load embedding model
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# 3. Encode text chunks into embeddings
embeddings = model.encode(chunk_texts, normalize_embeddings=True)

# 4. Build FAISS HNSW index
dim = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dim, 32)   # 32 neighbors by default
index.hnsw.efConstruction = 40         # efConstruction parameter
index.add(embeddings)

# 5. Save FAISS index
faiss.write_index(index, "nq_hnsw_faiss_n32e40.index")

# 6. Save chunk list
with open("nq_list.pkl", "wb") as f:
    pickle.dump(chunk_texts, f)

print("Index and chunk list saved successfully.")
```
