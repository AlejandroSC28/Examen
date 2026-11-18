"""API de recuperación SciFact desplegable en Vercel."""
from functools import lru_cache
import re
from typing import Literal

import os

os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"

import numpy as np
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI(
    title="SciFact Retrieval API",
    version="1.0.0",
    description="Búsqueda documental SciFact con TF-IDF coseno y BM25.",
)


def tokenize(text: str) -> list[str]:
    """Normalización mínima común a consultas y documentos."""
    return re.findall(r"(?u)\b\w+\b", str(text).lower())


@lru_cache(maxsize=1)
def get_index():
    """Carga documentos completos de SciFact y construye ambos índices."""
    rows = load_dataset("BeIR/scifact", "corpus", split="corpus")
    ids = [str(row["_id"]) for row in rows]
    titles = [row.get("title", "") for row in rows]
    texts = [(row.get("title", "") + " " + row.get("text", "")).strip() for row in rows]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    bm25 = BM25Okapi([tokenize(text) for text in texts], k1=1.5, b=0.75)
    return ids, titles, texts, vectorizer, matrix, bm25


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    model: Literal["bm25", "tfidf"] = "bm25"
    top_k: int = Field(default=10, ge=1, le=100)


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "corpus": "SciFact", "index_loaded": get_index.cache_info().currsize > 0}


@app.post("/search", tags=["retrieval"])
def search(request: SearchRequest):
    if not tokenize(request.query):
        raise HTTPException(status_code=400, detail="query debe contener términos alfanuméricos")
    ids, titles, texts, vectorizer, matrix, bm25 = get_index()
    if request.model == "tfidf":
        query_vector = vectorizer.transform([request.query])
        scores = (matrix @ query_vector.T).toarray().ravel()
    else:
        scores = np.asarray(bm25.get_scores(tokenize(request.query)))
    order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))
    order = [i for i in order if scores[i] > 0][:request.top_k]
    return {
        "model": request.model,
        "query": request.query,
        "top_k": request.top_k,
        "results": [
            {"rank": rank, "document_id": ids[i], "score": float(scores[i]),
             "title": titles[i], "snippet": texts[i][:500]}
            for rank, i in enumerate(order, 1)
        ],
    }
