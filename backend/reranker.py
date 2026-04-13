# reranker.py
# Cross-encoder reranker — the precision layer of DocCypher.
#
# After hybrid retrieval gives us 20 candidate chunks,
# the reranker scores each chunk against the query jointly.
# This is fundamentally more accurate than bi-encoder similarity
# because it sees the query and document together, not separately.
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
# - Free, open source, runs locally
# - Trained on MS MARCO — Microsoft's massive passage ranking dataset
# - 22MB model, fast inference
# - Industry standard for RAG reranking

import os
from typing import List, Dict
from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# This model was trained specifically for passage relevance ranking.
# It outputs a score: higher = more relevant to the query.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many chunks to keep after reranking — these go to the LLM
TOP_N_AFTER_RERANK = 5

# Singleton — load model once, reuse across requests
# Loading a transformer model takes 2-3 seconds, so we cache it
_reranker_model = None

def get_reranker():
    """
    Loads the cross-encoder model once and caches it.
    Why singleton? Model loading is expensive — 2-3 seconds.
    Subsequent calls return the cached model instantly.
    """
    global _reranker_model
    if _reranker_model is None:
        print(">> Loading cross-encoder reranker model...")
        _reranker_model = CrossEncoder(RERANKER_MODEL)
        print(">> Reranker loaded.")
    return _reranker_model

# ─────────────────────────────────────────────
# Reranking
# ─────────────────────────────────────────────

def rerank(query: str, chunks: List[Dict], top_n: int = TOP_N_AFTER_RERANK) -> List[Dict]:
    """
    Reranks candidate chunks using a cross-encoder model.
    
    Process:
    1. Create (query, chunk_text) pairs for each candidate
    2. Cross-encoder scores each pair jointly
    3. Sort by score descending
    4. Return top N chunks for the LLM
    
    The cross-encoder sees query and document together —
    it can detect nuanced relevance that bi-encoders miss.
    
    Args:
        query: the user's question
        chunks: candidate chunks from hybrid retrieval (typically 20)
        top_n: how many to return after reranking (typically 5)
    
    Returns:
        Top N chunks sorted by reranker score, with scores attached
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Build (query, passage) pairs — this is the cross-encoder input format
    pairs = [(query, chunk["text"]) for chunk in chunks]

    print(f">> Reranking {len(pairs)} candidates...")

    # Score all pairs
    # predict() returns raw logits — higher = more relevant
    scores = reranker.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = float(score)

    # Sort by reranker score descending
    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)

    # Take top N
    top_chunks = reranked[:top_n]

    print(f">> Reranking complete. Top {top_n} chunks selected.")
    for i, chunk in enumerate(top_chunks):
        print(f"   #{i+1} | Score: {chunk['reranker_score']:.4f} | "
              f"{chunk['filename']} p{chunk['page_num']} | "
              f"Found by both: {chunk.get('found_by_both', False)}")

    return top_chunks

# ─────────────────────────────────────────────
# Format citations
# ─────────────────────────────────────────────

def format_citations(chunks: List[Dict]) -> List[Dict]:
    """
    Formats the top chunks into clean citation objects.
    Every chunk becomes a citation that the LLM must reference.
    
    Citation format:
    {
        "citation_id": "[1]",
        "filename": "Claude.pdf",
        "page_num": 3,
        "text": "the actual chunk text...",
        "reranker_score": 0.94,
        "found_by_both": True
    }
    
    Why enforce citations? Enterprise RAG systems require auditability.
    Every claim in the answer must be traceable to a source chunk.
    This prevents hallucination and lets users verify answers.
    """
    citations = []
    for i, chunk in enumerate(chunks):
        citations.append({
            "citation_id": f"[{i+1}]",
            "filename": chunk["filename"],
            "page_num": chunk["page_num"],
            "text": chunk["text"],
            "reranker_score": round(chunk["reranker_score"], 4),
            "found_by_both": chunk.get("found_by_both", False),
            "chunk_index": chunk.get("chunk_index", 0),
        })
    return citations