# RAG Agent Example

This example demonstrates training a Retrieval-Augmented Generation (RAG) agent using Agent-Lightning with Wikipedia retrieval capabilities. The agent answers multi-hop questions from the MuSiQue dataset by retrieving and reasoning over Wikipedia passages. **It's tested and compatible with Agent-lightning v0.1.x**.

## Overview

This example originally runs on a single node with four GPUs, each requiring at least 40GB of memory.

1. Prepare the RAG dataset in the wiki_retriever_mcp folder. Wiki chunks (`nq_list.pkl`) and Faiss index (`nq_hnsw_faiss_n32e40.index`) are required. (Full wiki dump files are huge, additional information will be provided later)
2. Prepare the training data in the `data` folder. Download from [here](https://drive.google.com/drive/folders/1hEqOY4EbplUB5ew-8UPFhV_5QU2j7WCN?usp=drive_link). `musique_train.parquet` and `musique_dev_128.parquet` are required.
3. Set up the environment for wiki retriever MCP: `bash wiki_retriever_install.sh`. This will install the required packages and set up the environment for the wiki retriever MCP.
4. Start the wiki retriever MCP: `python wiki_retriever_mcp.py`. This will start the wiki retriever MCP server.
5. Start Ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting Ray.
6. Run the agent: `python rag_agent.py`. This automatically launches 12 agent workers by default.
7. In another terminal, launch the training server: `bash train.sh`.

## Included Files

| File/Directory | Description |
|----------------|-------------|
| `rag_agent.py` | Entry point for running the Agent-Lightning RAG training pipeline |
| `train.sh` | Starts the GRPO training server that updates the agent |
| `utils.py` | Scoring utilities for exact match, F1, and response parsing |
| `wiki_retriever_mcp/` | Setup scripts and MCP server (`wiki_retriever_install.sh`, `wiki_retriever_mcp.py`) for Wikipedia retrieval |

## Preparing the Retrieval Corpus

To enable semantic retrieval with this mcp server, we need two files:

1. **FAISS index file** (`.index`)
2. **Chunk list file** (`.pkl`)

These two files work together: the FAISS index stores the vector embeddings and their mapping to integer IDs, while the pickle file stores the actual text chunks. The integer IDs in the index correspond exactly to the positions in the chunk list.

---

### Step 1. Collecting Text Chunks

You first need a collection of text passages (chunks). For example, you can download a Wikipedia-based dataset such as `wiki18_100w.zip` in the [FlashRAG_dataset](https://huggingface.co/datasets/FlashRAG) or use other pre-split corpora.

---

### Step 2. Creating the FAISS Index (`nq_hnsw_faiss_n32e40.index`)

- Use a sentence embedding model (e.g., `BAAI/bge-large-en-v1.5`) to encode each chunk into a vector.
- Build a FAISS index from these vectors.
- In this example, we use an **HNSW index** (Hierarchical Navigable Small World graph), which supports efficient approximate nearest-neighbor search.
- The index only stores embeddings and integer IDs (no raw text).

---

### Step 3. Creating the Chunk List (`nq_list.pkl`)

- Store the raw text chunks in a Python list.
- Save this list with `pickle`.
- The index ID returned by FAISS corresponds to the list index in this file. For example, if FAISS search returns `I[0][i] = 12345`, then the corresponding text chunk is `chunks[12345]`.

---

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

---

### Step 4. Code Example: Building Index and Chunk List
Warning: The following example only demonstrates a small-scale workflow. In practice, if the dataset is large, you should encode the text in batches and incrementally add them to the index.

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


## Evaluation

Results are coming soon.
