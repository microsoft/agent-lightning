# Copyright (c) Microsoft. All rights reserved.

# type: ignore

import pickle

import faiss
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer

# index = faiss.read_index("/mnt/input/agent_lightning/nq_hnsw_faiss_n32e40.index")
index = faiss.read_index("nq_hnsw_faiss_n32e40.index")
print("Index loaded successfully.")

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
print("Model loaded successfully.")

# with open('/mnt/input/agent_lightning/nq_list.pkl', 'rb') as f:
with open("nq_list.pkl", "rb") as f:
    chunks = pickle.load(f)
print("Chunks loaded successfully.")

mcp = FastMCP(name="wiki retrieval mcp")


@mcp.tool(
    name="retrieve",
    description="retrieve relevant chunks from the wikipedia",
)
def retrieve(query: str) -> list:
    """
    Retrieve relevant chunks from the Wikipedia dataset.

    Args:
        query (str): The query string to search for.

    Returns:
        list: A list of dictionaries containing the retrieved chunks and their metadata.
    """
    top_k = 4  # Number of top results to return
    embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(embedding, top_k)

    results = []
    for i in range(top_k):
        if I[0][i] != -1:
            chunk = chunks[I[0][i]]
            results.append({"chunk": chunk, "chunk_id": I[0][i], "distance": D[0][i]})
    return results


if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8099)
