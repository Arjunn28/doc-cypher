# main.py

import os
import json
import glob
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
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

# Removed startup_event entirely — models load lazily on first use

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
    filename_filter: list = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "online",
        "service": "DocCypher",
        "version": "1.0.0",
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    filepath = os.path.join(UPLOAD_PATH, file.filename)
    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

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
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = answer_query(request.query, filename_filter=request.filename_filter)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return result

@app.get("/stream")
def stream(query: str, filename_filter: str = None):
    """filename_filter is comma-separated: 'doc1.pdf,doc2.pdf'"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    filter_list = [f.strip() for f in filename_filter.split(",")] if filename_filter else None

    return StreamingResponse(
        stream_answer(query, filename_filter=filter_list),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        },
    )

@app.get("/documents")
def list_documents():
    corpus_path = os.path.join(BM25_PATH, "corpus.json")

    if not os.path.exists(corpus_path):
        return {"documents": [], "total_chunks": 0}

    with open(corpus_path, "r") as f:
        saved = json.load(f)

    chunks = saved.get("chunks", [])

    docs = {}
    for chunk in chunks:
        fname = chunk["filename"]
        if fname not in docs:
            docs[fname] = {"filename": fname, "chunks": 0, "pages": set()}
        docs[fname]["chunks"] += 1
        docs[fname]["pages"].add(chunk["page_num"])

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

# ─────────────────────────────────────────────
# These were accidentally nested inside list_documents — now fixed
# ─────────────────────────────────────────────

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    errors = []

    try:
        from backend.ingest import get_chroma_client, get_collection
        client = get_chroma_client()
        collection = get_collection(client)
        existing = collection.get(where={"filename": filename})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])
            print(f"Deleted {len(existing['ids'])} chunks from ChromaDB for {filename}")
    except Exception as e:
        errors.append(f"ChromaDB: {str(e)}")

    try:
        corpus_path = os.path.join(BM25_PATH, "corpus.json")
        if os.path.exists(corpus_path):
            with open(corpus_path, "r") as f:
                saved = json.load(f)

            kept_chunks = [c for c in saved["chunks"] if c["filename"] != filename]
            kept_corpus = [saved["corpus"][i] for i, c in enumerate(saved["chunks"]) if c["filename"] != filename]

            with open(corpus_path, "w") as f:
                json.dump({"chunks": kept_chunks, "corpus": kept_corpus}, f)

            removed = len(saved["chunks"]) - len(kept_chunks)
            print(f"Removed {removed} chunks from BM25 for {filename}")
    except Exception as e:
        errors.append(f"BM25: {str(e)}")

    try:
        filepath = os.path.join(UPLOAD_PATH, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        errors.append(f"File: {str(e)}")

    if errors:
        raise HTTPException(status_code=500, detail=f"Partial delete errors: {errors}")

    return {"status": "success", "message": f"{filename} deleted successfully"}


@app.get("/documents/{filename}/download")
def download_document(filename: str):
    filepath = os.path.join(UPLOAD_PATH, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        filename=filename,
    )