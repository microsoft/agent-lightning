conda create -n mcp_server python=3.12 -y
conda activate mcp_server
pip install faiss-cpu==1.11.0 fastmcp==2.5.1 sentence-transformers==4.1.0
