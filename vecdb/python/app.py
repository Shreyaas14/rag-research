"""
FastAPI endpoints for vecdb.
"""
from fastapi import FastAPI
from vector_core import init_system, upsert, search
import os

app = FastAPI(title="Vector DB with cuVS + pgvector")

DB_URL = os.getenv("DB_URL")
GROK_API_KEY = os.getenv("GROK_API_KEY")

@app.on_event("startup")
async def startup():
    if not DB_URL or not GROK_API_KEY:
        raise RuntimeError("DB_URL or GROK_API_KEY missing")
    init_system(DB_URL, 1024)  # 1024 = Grok-beta dim

@app.post("/upsert")
async def add_item(text: str, metadata: dict = {}):
    id_ = await upsert(text, str(metadata))
    return {"id": id_, "status": "inserted"}

@app.get("/search")
async def search_query(q: str, k: int = 5, genre: str = None):
    filter_json = {"genre": genre} if genre else None
    results = await search(q, k, str(filter_json) if filter_json else None)
    return [
        {"id": int(r[0]), "score": float(r[1]), "metadata": eval(r[2])}
        for r in results
    ]

@app.get("/health")
async def health():
    return {"status": "ok", "db": DB_URL[:20] + "..."}