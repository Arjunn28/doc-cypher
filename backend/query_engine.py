# query_engine.py
# The orchestrator — connects retrieval → reranking → LLM.
# Takes a user query, runs the full pipeline, and returns
# a streamed answer with enforced citations.
#
# Citation enforcement works by:
# 1. Numbering each source chunk [1], [2], [3]...
# 2. Telling the LLM it MUST cite sources inline
# 3. Parsing the response to extract which citations were used
#
# Streaming works by yielding LLM tokens as they arrive —
# the answer appears word by word like ChatGPT.

import os
import json
from typing import List, Dict, Generator
from dotenv import load_dotenv
from groq import Groq

from backend.retriever import hybrid_search
from backend.reranker import rerank, format_citations

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TOP_K_RETRIEVE = 20   # candidates from hybrid search
TOP_N_RERANK = 5      # chunks after reranking → sent to LLM

# ─────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────

def build_prompt(query: str, citations: List[Dict]) -> str:
    """
    Builds the citation-enforced prompt for the LLM.

    Key design decisions:
    1. Each source chunk is numbered [1], [2], etc.
    2. LLM is explicitly told to cite inline — no citation = no answer
    3. LLM is told NOT to use outside knowledge — only the provided chunks
    4. This prevents hallucination by grounding the answer in retrieved text

    Why enforce citations? Enterprise RAG requires auditability.
    Every claim must be traceable to a source document and page.
    """
    # Build the context block with numbered sources
    context_parts = []
    for citation in citations:
        context_parts.append(
            f"SOURCE {citation['citation_id']} "
            f"(File: {citation['filename']}, Page: {citation['page_num']}):\n"
            f"{citation['text']}\n"
        )
    context = "\n---\n".join(context_parts)

    prompt = f"""You are DocCypher, an intelligent document assistant.
Answer the user's question using ONLY the provided source chunks below.

STRICT RULES:
1. Cite sources inline using [1], [2], [3] etc. after every claim
2. Only use information from the provided sources — never from outside knowledge
3. If the sources don't contain enough information, say "The provided documents don't contain enough information to answer this fully"
4. Be concise but complete — aim for 3-5 sentences unless the question requires more
5. Always end with a "Sources used:" section listing which citations you referenced

SOURCES:
{context}

USER QUESTION: {query}

ANSWER (with inline citations):"""

    return prompt

# ─────────────────────────────────────────────
# Streaming query
# ─────────────────────────────────────────────

def stream_answer(query: str, filename_filter: str = None) -> Generator[str, None, None]:
    """
    Full RAG pipeline with streaming output.

    Pipeline:
    1. Hybrid retrieval (BM25 + vector, RRF fusion)
    2. Cross-encoder reranking
    3. Citation formatting
    4. LLM streaming response

    Yields:
    - First yields a JSON metadata block with citations
    - Then yields LLM tokens one by one as they stream
    - Finally yields a JSON done block

    Why streaming? Production UX — users see the answer
    forming in real time rather than waiting 5-10 seconds
    for the full response.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    # Step 1: Hybrid retrieval
    filter_msg = f" (scoped to: {filename_filter})" if filename_filter else " (all documents)"
    print(f"\n{'='*50}")
    print(f"QUERY: {query}{filter_msg}")
    print(f"{'='*50}")

    candidates = hybrid_search(query, top_k=TOP_K_RETRIEVE)

    if not candidates:
        yield json.dumps({"type": "error", "message": "No documents found. Please upload a PDF first."})
        return

    # Step 2: Rerank
    top_chunks = rerank(query, candidates, top_n=TOP_N_RERANK)

    # Step 3: Format citations
    citations = format_citations(top_chunks)

    # Yield citations metadata first so frontend can display sources
    # before the answer starts streaming
    yield json.dumps({
        "type": "citations",
        "citations": [
            {
                "citation_id": c["citation_id"],
                "filename": c["filename"],
                "page_num": c["page_num"],
                "reranker_score": c["reranker_score"],
                "found_by_both": c["found_by_both"],
                "preview": c["text"][:200],
            }
            for c in citations
        ]
    }) + "\n"

    # Step 4: Build prompt and stream from LLM
    prompt = build_prompt(query, citations)

    print(f"\n>> Streaming from LLM...")

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # low temperature for factual, consistent answers
        max_tokens=1024,
        stream=True,      # this is what enables word-by-word streaming
    )

    # Stream tokens as they arrive
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield json.dumps({
                "type": "token",
                "content": delta.content
            }) + "\n"

    # Signal completion
    yield json.dumps({"type": "done"}) + "\n"
    print(">> Stream complete.")

# ─────────────────────────────────────────────
# Non-streaming query (for testing)
# ─────────────────────────────────────────────
def answer_query(query: str, filename_filter: str = None) -> Dict:
    """
    Non-streaming version — collects the full answer at once.
    Used for testing and for the /query endpoint fallback.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    candidates = hybrid_search(query, top_k=TOP_K_RETRIEVE, filename_filter=filename_filter)
    if not candidates:
        return {"error": "No documents found. Please upload a PDF first."}

    top_chunks = rerank(query, candidates, top_n=TOP_N_RERANK)
    citations = format_citations(top_chunks)
    prompt = build_prompt(query, citations)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "citations": [
            {
                "citation_id": c["citation_id"],
                "filename": c["filename"],
                "page_num": c["page_num"],
                "reranker_score": c["reranker_score"],
                "preview": c["text"][:200],
            }
            for c in citations
        ],
        "chunks_retrieved": len(candidates),
        "chunks_after_rerank": len(top_chunks),
    }