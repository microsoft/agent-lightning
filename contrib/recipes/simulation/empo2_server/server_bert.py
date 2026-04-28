from fastapi import FastAPI, Request
import time
from pydantic import BaseModel
import uvicorn

import torch
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.set_per_process_memory_fraction(0.1, 0)

num_works = 1

app = FastAPI()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/key_cal/")
async def compress(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
    except:
        text = (await request.body()).decode("utf-8")
    
    key = model.encode(text)
    return {"key": key.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=num_works)