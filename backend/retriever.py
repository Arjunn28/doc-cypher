# retriever.py
# The hybrid retrieval engine — the heart of DocCypher.
# Runs BM25 (keyword) and ChromaDB (semantic) search in parallel,
# then combines results using Reciprocal Rank Fusion (RRF).
#
# Why hybrid? Two failure modes we're solving:
# 1. Pure vector search misses exact keyword matches ("Section 4.2", product codes)
# 2. Pure BM25 misses semantic similarity ("car" won't match "automobile")
# Hybrid covers both failure modes simultaneously.

import os
import json
import numpy as np
from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

from backend.ingest import (
    get_chroma_client,
    get_collection,
    get_embedding_model,
    BM25_PATH,
    EMBEDDING_MODEL,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# How many results to fetch from each retriever before fusion
# Fetch more than you need — reranker will cut it down to top 5
TOP_K_EACH = 10

# RRF smoothing constant — 60 is the empirically validated default
RRF_K = 60

# ─────────────────────────────────────────────
# Load BM25 index from disk
# ─────────────────────────────────────────────

def load_bm25_index() -> Tuple[BM25Okapi, List[Dict]]:
    """
    Loads the persisted BM25 corpus and chunk metadata from disk.
    Returns the BM25 index object and the list of chunk dicts.
    
    Why persist to disk? So the index survives server restarts.
    BM25 is rebuilt from the saved corpus on every load — fast operation.
    """
    corpus_path = os.path.join(BM25_PATH, "corpus.json")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            "BM25 index not found. Please ingest at least one PDF first."
        )

    with open(corpus_path, "r") as f:
        saved = json.load(f)

    corpus = saved["corpus"]      # list of tokenized documents
    chunks = saved["chunks"]      # list of chunk metadata dicts

    # Rebuild BM25 index from corpus
    # BM25Okapi is the standard variant — Okapi BM25 with k1=1.5, b=0.75
    bm25 = BM25Okapi(corpus)

    return bm25, chunks

# ─────────────────────────────────────────────
# BM25 Search
# ─────────────────────────────────────────────

def bm25_search(query: str, top_k: int = TOP_K_EACH) -> List[Dict]:
    """
    Searches the BM25 index for chunks matching the query keywords.
    
    Returns ranked list of chunks with their BM25 scores.
    BM25 is particularly good at:
    - Exact keyword matches
    - Named entities (people, companies, product names)
    - Technical terms, codes, section numbers
    """
    bm25, chunks = load_bm25_index()

    # Tokenize query the same way we tokenized the corpus
    tokenized_query = query.lower().split()

    # Get BM25 scores for all chunks
    scores = bm25.get_scores(tokenized_query)

    # Get top K indices sorted by score descending
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices):
        if scores[idx] > 0:  # only include chunks with non-zero score
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "text": chunks[idx]["text"],
                "filename": chunks[idx]["filename"],
                "page_num": chunks[idx]["page_num"],
                "chunk_index": chunks[idx]["chunk_index"],
                "bm25_score": float(scores[idx]),
                "bm25_rank": rank + 1,
                "source": "bm25",
            })

    return results

# ─────────────────────────────────────────────
# ChromaDB Vector Search
# ─────────────────────────────────────────────

def vector_search(query: str, top_k: int = TOP_K_EACH) -> List[Dict]:
    """
    Searches ChromaDB for semantically similar chunks using embeddings.
    
    Process:
    1. Embed the query using the same model used for documents
    2. ChromaDB finds chunks with highest cosine similarity to query embedding
    3. Returns ranked results with similarity scores
    
    Vector search is particularly good at:
    - Semantic similarity ("automobile" matches "car")
    - Paraphrased questions
    - Conceptual queries
    """
    model = get_embedding_model()
    client = get_chroma_client()
    collection = get_collection(client)

    if collection.count() == 0:
        return []

    # Embed the query
    query_embedding = model.encode([query]).tolist()[0]

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["ids"] and results["ids"][0]:
        for rank, (id_, doc, meta, dist) in enumerate(zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity score (higher = better)
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
    """
    Combines BM25 and vector search results using Reciprocal Rank Fusion.
    
    RRF formula: score(d) = Σ 1/(k + rank(d))
    where rank(d) is the document's rank in each list.
    
    Why RRF instead of score normalization?
    BM25 scores and cosine similarities are on completely different scales.
    Normalizing them requires knowing the distribution of all scores upfront.
    RRF sidesteps this entirely by using only rank positions — elegant and robust.
    
    k=60 is the empirically validated constant from the original RRF paper
    (Cormack, Clarke & Buettcher, 2009).
    """
    # Build a dict of chunk_id → merged chunk data + RRF score
    fused = {}

    # Score BM25 results
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

    # Score vector results
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

    # Sort by RRF score descending
    ranked = sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    # Tag chunks found by both retrievers — these are the most reliable
    for chunk in ranked:
        chunk["found_by_both"] = len(chunk["sources"]) == 2

    return ranked

# ─────────────────────────────────────────────
# Main hybrid search function
# ─────────────────────────────────────────────

def hybrid_search(query: str, top_k: int = 20) -> List[Dict]:
    """
    The main entry point for retrieval.
    Runs BM25 and vector search in parallel, fuses with RRF.
    Returns top_k results for the reranker to process.
    
    We return more than needed (default 20) because the reranker
    will cut this down to the final top 5 most relevant chunks.
    """
    print(f"\n>> Hybrid search for: '{query}'")

    # Run both searches
    bm25_results = bm25_search(query, top_k=TOP_K_EACH)
    vector_results = vector_search(query, top_k=TOP_K_EACH)

    print(f"   BM25 returned: {len(bm25_results)} results")
    print(f"   Vector returned: {len(vector_results)} results")

    # Fuse with RRF
    fused = reciprocal_rank_fusion(bm25_results, vector_results)

    # Take top_k for reranking
    top_results = fused[:top_k]

    # Count how many were found by both
    both_count = sum(1 for r in top_results if r.get("found_by_both"))
    print(f"   After RRF fusion: {len(top_results)} results ({both_count} found by both retrievers)")

    return top_results