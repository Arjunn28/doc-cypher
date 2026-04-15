import os
import json
import httpx
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import chromadb

from backend.ingest import (
    get_chroma_client,
    get_collection,
    BM25_PATH,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TOP_K_EACH = 10
RRF_K = 60

# GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
# GROQ_EMBED_MODEL = "nomic-embed-text-v1_5"


# ─────────────────────────────────────────────
# Hugging Face Query Embedding
# ─────────────────────────────────────────────

# HF_EMBED_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_EMBED_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"


def get_query_embedding(query: str) -> List[float]:
    """Embeds query via HuggingFace Inference API."""
    response = httpx.post(
        HF_EMBED_URL,
        json={"inputs": [query], "options": {"wait_for_model": True}},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()[0]

# ─────────────────────────────────────────────
# Load BM25 index from disk
# ─────────────────────────────────────────────

def load_bm25_index() -> Tuple[BM25Okapi, List[Dict]]:
    corpus_path = os.path.join(BM25_PATH, "corpus.json")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            "BM25 index not found. Please ingest at least one PDF first."
        )

    with open(corpus_path, "r") as f:
        saved = json.load(f)

    corpus = saved["corpus"]
    chunks = saved["chunks"]
    bm25 = BM25Okapi(corpus)

    return bm25, chunks

# ─────────────────────────────────────────────
# BM25 Search
# ─────────────────────────────────────────────

def bm25_search(query: str, top_k: int = TOP_K_EACH, filename_filter: list = None) -> List[Dict]:
    bm25, chunks = load_bm25_index()

    if filename_filter:
        filtered_chunks = [c for c in chunks if c["filename"] in filename_filter]
        if not filtered_chunks:
            return []
        filtered_corpus = [c["text"].lower().split() for c in filtered_chunks]
        bm25_filtered = BM25Okapi(filtered_corpus)
        scores = bm25_filtered.get_scores(query.lower().split())
        active_chunks = filtered_chunks
    else:
        scores = bm25.get_scores(query.lower().split())
        active_chunks = chunks

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:
            results.append({
                "chunk_id": active_chunks[idx]["chunk_id"],
                "text": active_chunks[idx]["text"],
                "filename": active_chunks[idx]["filename"],
                "page_num": active_chunks[idx]["page_num"],
                "chunk_index": active_chunks[idx]["chunk_index"],
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank + 1,
                "source": "bm25",
            })

    return results

# ─────────────────────────────────────────────
# ChromaDB Vector Search
# ─────────────────────────────────────────────

def vector_search(query: str, top_k: int = TOP_K_EACH, filename_filter: list = None) -> List[Dict]:
    client = get_chroma_client()
    collection = get_collection(client)

    if collection.count() == 0:
        return []

    query_embedding = get_query_embedding(query)

    if filename_filter and len(filename_filter) == 1:
        where = {"filename": filename_filter[0]}
    elif filename_filter and len(filename_filter) > 1:
        where = {"filename": {"$in": filename_filter}}
    else:
        where = None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
        where=where,
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for rank, (id_, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            similarity = 1 - (dist / 2)
            chunks.append({
                "chunk_id": id_,
                "text": doc,
                "filename": meta["filename"],
                "page_num": meta["page_num"],
                "chunk_index": meta["chunk_index"],
                "vector_score": float(similarity),
                "vector_rank": rank + 1,
                "source": "vector",
            })

    return chunks

# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: List[Dict],
    vector_results: List[Dict],
    k: int = RRF_K,
) -> List[Dict]:
    fused = {}

    for result in bm25_results:
        chunk_id = result["chunk_id"]
        rank = result["bm25_rank"]
        rrf_score = 1 / (k + rank)

        if chunk_id not in fused:
            fused[chunk_id] = {**result, "rrf_score": 0.0, "sources": []}

        fused[chunk_id]["rrf_score"] += rrf_score
        fused[chunk_id]["sources"].append("bm25")
        fused[chunk_id]["bm25_rank"] = rank
        fused[chunk_id]["bm25_score"] = result.get("bm25_score", 0)

    for result in vector_results:
        chunk_id = result["chunk_id"]
        rank = result["vector_rank"]
        rrf_score = 1 / (k + rank)

        if chunk_id not in fused:
            fused[chunk_id] = {**result, "rrf_score": 0.0, "sources": []}

        fused[chunk_id]["rrf_score"] += rrf_score
        fused[chunk_id]["sources"].append("vector")
        fused[chunk_id]["vector_rank"] = rank
        fused[chunk_id]["vector_score"] = result.get("vector_score", 0)

    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    for chunk in ranked:
        chunk["found_by_both"] = len(chunk["sources"]) == 2

    return ranked

# ─────────────────────────────────────────────
# Main hybrid search function
# ─────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 20, filename_filter: list = None) -> List[Dict]:
    filter_msg = f" (filtered to: {filename_filter})" if filename_filter else " (all documents)"
    print(f"\n>> Hybrid search for: '{query}'{filter_msg}")

    bm25_results = bm25_search(query, top_k=TOP_K_EACH, filename_filter=filename_filter)
    vector_results = vector_search(query, top_k=TOP_K_EACH, filename_filter=filename_filter)

    print(f"   BM25 returned: {len(bm25_results)} results")
    print(f"   Vector returned: {len(vector_results)} results")

    fused = reciprocal_rank_fusion(bm25_results, vector_results)
    top_results = fused[:top_k]
    both_count = sum(1 for r in top_results if r.get("found_by_both"))
    print(f"   After RRF fusion: {len(top_results)} results ({both_count} found by both)")

    return top_results