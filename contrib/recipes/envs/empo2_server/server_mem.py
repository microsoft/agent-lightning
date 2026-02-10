from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import time
import random
from collections import deque

num_works = 1
app = FastAPI()

mem_list = None
content_set = None

class MemRequest(BaseModel):
    key: list
    idx: int = None
    content: str = None
    score: float = None

@app.post("/mem/")
async def mem_handler(mem_req: MemRequest):
    global cnt, mem_list, content_set

    key = mem_req.key
    idx = mem_req.idx
    content = mem_req.content
    score = mem_req.score

    if content=="Reset":
        mem_list_num = idx
        content_set = {id: set() for id in range(mem_list_num)}
        mem_list = {id: [] for id in range(mem_list_num)}
        cnt = {id: 0 for id in range(mem_list_num)}
        print(f"Clean all the mem. The num of mem_list is {mem_list_num}")
        return None

    if content is not None:
        if content not in content_set[idx]:
            content_set[idx].add(content)
            mem_list[idx].append({
                "cnt": cnt[idx],
                "key": key,
                "content": content,
                "score": score,
            })
            cnt[idx] += 1
            if len(mem_list[idx]) > 1000:
                oldest_hash = mem_list[idx][0]["content"]
                content_set[idx].discard(oldest_hash)
                mem_list[idx] = mem_list[idx][-1000:]
            print("Add,", "id", idx, "cnt", cnt[idx], "content", content, "score", score)
    else:
        data = []
        for mem in mem_list[idx]:
            mem_key = mem["key"]
            sim = np.dot(key, mem_key) / (np.linalg.norm(key) * np.linalg.norm(mem_key))
            if sim > 0.5:
                data.append(mem)
        # data = random.sample(data, min(len(data), 10)) if len(data) > 0 else []
        data = sorted(data, key=lambda x: -x["score"])[:10] if len(data) > 0 else []
        data = [x["content"] for x in data]
        count = len(data)
        print("Load", count, data)
        return count, data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, workers=num_works)