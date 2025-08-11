# RAG Agent Example

This example originally runs on a single node with four GPUs, each requiring at least 40GB of memory.

1. Prepare the RAG dataset in the wiki_retriever_mcp folder. Wiki chunks (`nq_list.pkl`) and Faiss index (`nq_hnsw_faiss_n32e40.index`) are required. (Full wiki dump files are huge, additional information will be provided later)
2. Prepare the training data in the `data` folder. Download from [here](https://drive.google.com/drive/folders/1hEqOY4EbplUB5ew-8UPFhV_5QU2j7WCN?usp=drive_link). `musique_train.parquet` and `musique_dev_128.parquet` are required.
3. Set up the environment for wiki retriever MCP: `bash wiki_retriever_install.sh`. This will install the required packages and set up the environment for the wiki retriever MCP.
4. Start the wiki retriever MCP: `python wiki_retriever_mcp.py`. This will start the wiki retriever MCP server.
5. Start Ray: `bash ../../scripts/restart_ray.sh`. To use Wandb, you need to set the WANDB_API_KEY environment variable before starting Ray.
6. Run the agent: `python rag_agent.py`. This automatically launches 12 agent workers by default.
7. In another terminal, launch the training server: `bash train.sh`.

## Evaluation

Results are coming soon.
