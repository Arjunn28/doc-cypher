# reranker.py — lightweight reranker (no torch, no cross-encoder)
# Uses TF-IDF-style keyword overlap scoring instead of a neural cross-encoder.
# Saves ~100MB RAM with minimal quality loss for most queries.

from typing import List, Dict

TOP_N_AFTER_RERANK = 5

def rerank(query: str, chunks: List[Dict], top_n: int = TOP_N_AFTER_RERANK) -> List[Dict]:
    """
    Scores chunks by keyword overlap with the query.
    Fast, zero-memory, no model required.
    """
    if not chunks:
        return []

    query_terms = set(query.lower().split())

    for chunk in chunks:
        text_terms = set(chunk["text"].lower().split())
        overlap = len(query_terms & text_terms)
        # Combine keyword overlap with the RRF score from retrieval
        chunk["reranker_score"] = float(overlap) + chunk.get("rrf_score", 0.0)

    reranked = sorted(chunks, key=lambda x: x["reranker_score"], reverse=True)
    top_chunks = reranked[:top_n]

    print(f">> Reranking complete. Top {top_n} chunks selected.")
    for i, chunk in enumerate(top_chunks):
        print(f"   #{i+1} | Score: {chunk['reranker_score']:.4f} | "
              f"{chunk['filename']} p{chunk['page_num']}")

    return top_chunks


def format_citations(chunks: List[Dict]) -> List[Dict]:
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