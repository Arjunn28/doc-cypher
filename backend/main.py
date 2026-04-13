# main.py
# FastAPI server — exposes DocCypher as a REST API.
# Four endpoints:
# POST /upload    → ingest a PDF
# POST /query     → ask a question (non-streaming)
# GET  /stream    → ask a question (streaming, SSE)
# GET  /documents → list all ingested documents
# GET  /health    → server status

import os
import json
import glob
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from backend.ingest import ingest_pdf, UPLOAD_PATH, BM25_PATH
from backend.query_engine import answer_query, stream_answer

load_dotenv()

app = FastAPI(
    title="DocCypher — Intelligent Document Intelligence",
    description="Hybrid RAG with BM25 + vector search, cross-encoder reranking, and citation enforcement",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(UPLOAD_PATH, exist_ok=True)

# ─────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Server status check."""
    return {
        "status": "online",
        "service": "DocCypher",
        "version": "1.0.0",
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and ingest a PDF.
    Saves the file, runs the full ingestion pipeline:
    parse → chunk → embed → store in ChromaDB + BM25.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file
    filepath = os.path.join(UPLOAD_PATH, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    # Run ingestion pipeline
    try:
        result = ingest_pdf(filepath)
        if "error" in result:
            raise HTTPException(status_code=422, detail=result["error"])
        return {
            "status": "success",
            "message": f"Successfully ingested {file.filename}",
            "details": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def query(request: QueryRequest):
    """
    Ask a question — non-streaming response.
    Returns full answer with citations in one JSON response.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = answer_query(request.query)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result

@app.get("/stream")
def stream(query: str):
    """
    Ask a question — streaming response using Server-Sent Events.
    Returns tokens as they arrive from the LLM.
    Frontend receives:
    1. A citations JSON block first
    2. Token by token content
    3. A done signal
    """
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        stream_answer(query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering for streaming
        },
    )

@app.get("/documents")
def list_documents():
    """
    Lists all ingested documents.
    Reads from the BM25 corpus to get document metadata.
    """
    corpus_path = os.path.join(BM25_PATH, "corpus.json")

    if not os.path.exists(corpus_path):
        return {"documents": [], "total_chunks": 0}

    with open(corpus_path, "r") as f:
        saved = json.load(f)

    chunks = saved.get("chunks", [])

    # Group by filename
    docs = {}
    for chunk in chunks:
        fname = chunk["filename"]
        if fname not in docs:
            docs[fname] = {
                "filename": fname,
                "chunks": 0,
                "pages": set(),
            }
        docs[fname]["chunks"] += 1
        docs[fname]["pages"].add(chunk["page_num"])

    # Convert sets to counts for JSON serialization
    doc_list = [
        {
            "filename": d["filename"],
            "chunks": d["chunks"],
            "page_count": len(d["pages"]),
        }
        for d in docs.values()
    ]

    return {
        "documents": doc_list,
        "total_chunks": len(chunks),
    }