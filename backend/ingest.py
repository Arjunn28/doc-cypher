import os
import json
import fitz
import chromadb
import httpx
from rank_bm25 import BM25Okapi
from typing import List, Dict

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma")
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), "../uploads")
BM25_PATH = os.path.join(os.path.dirname(__file__), "../data/bm25")
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

GROQ_EMBED_URL = "https://api.groq.com/openai/v1/embeddings"
GROQ_EMBED_MODEL = "nomic-embed-text-v1_5"

# ─────────────────────────────────────────────
# Memory logging
# ─────────────────────────────────────────────

def log_memory(label):
    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[MEM] {label}: {mem / 1024:.1f} MB")
    except:
        pass

# ─────────────────────────────────────────────
# Groq Embeddings — no local model, no RAM cost
# ─────────────────────────────────────────────

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Calls Groq embedding API. No local model loaded, no RAM spike."""
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        GROQ_EMBED_URL,
        headers=headers,
        json={"model": GROQ_EMBED_MODEL, "input": texts},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

# ─────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────

def get_chroma_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_collection(client, collection_name: str = "doccypher"):
    from chromadb.utils.embedding_functions import EmbeddingFunction

    class NoOpEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input):
            return []

    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=NoOpEmbeddingFunction(),
    )

# ─────────────────────────────────────────────
# Step 1: PDF Parsing
# ─────────────────────────────────────────────

def parse_pdf(filepath: str) -> List[Dict]:
    doc = fitz.open(filepath)
    filename = os.path.basename(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip() and len(text.strip()) > 50:
            pages.append({
                "page_num": page_num + 1,
                "text": text.strip(),
                "filename": filename,
            })

    doc.close()
    print(f"Parsed {len(pages)} pages from {filename}")
    return pages

# ─────────────────────────────────────────────
# Step 2: Chunking
# ─────────────────────────────────────────────

def chunk_pages(pages: List[Dict]) -> List[Dict]:
    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            if chunk_text.strip() and len(chunk_text.strip()) > 30:
                chunks.append({
                    "chunk_id": f"{page['filename']}_p{page['page_num']}_c{chunk_index}",
                    "text": chunk_text.strip(),
                    "filename": page["filename"],
                    "page_num": page["page_num"],
                    "chunk_index": chunk_index,
                    "start_char": start,
                })
                chunk_index += 1

            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

# ─────────────────────────────────────────────
# Step 3: Store in ChromaDB
# ─────────────────────────────────────────────

def store_in_chroma(chunks: List[Dict], collection) -> None:
    batch_size = 20  # Groq API processes up to 20 texts per request

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "filename": c["filename"],
                "page_num": c["page_num"],
                "chunk_index": c["chunk_index"],
            }
            for c in batch
        ]

        batch_embeddings = get_embeddings(texts)

        existing = collection.get(ids=ids)
        existing_ids = set(existing["ids"])
        new_indices = [j for j, id_ in enumerate(ids) if id_ not in existing_ids]

        if new_indices:
            collection.add(
                ids=[ids[j] for j in new_indices],
                documents=[texts[j] for j in new_indices],
                embeddings=[batch_embeddings[j] for j in new_indices],
                metadatas=[metadatas[j] for j in new_indices],
            )

        print(f"  Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")

    print(f"Stored {len(chunks)} chunks in ChromaDB")

# ─────────────────────────────────────────────
# Step 4: BM25 index
# ─────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict]) -> None:
    os.makedirs(BM25_PATH, exist_ok=True)
    corpus_path = os.path.join(BM25_PATH, "corpus.json")
    existing_corpus = []
    existing_chunks = []

    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            saved = json.load(f)
            existing_corpus = saved.get("corpus", [])
            existing_chunks = saved.get("chunks", [])

    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("BM25 index already up to date.")
        return

    new_tokenized = [chunk["text"].lower().split() for chunk in new_chunks]
    all_corpus = existing_corpus + new_tokenized
    all_chunks = existing_chunks + new_chunks

    with open(corpus_path, "w") as f:
        json.dump({"corpus": all_corpus, "chunks": all_chunks}, f)

    print(f"BM25 index built with {len(all_corpus)} chunks total")

# ─────────────────────────────────────────────
# Main ingestion function
# ─────────────────────────────────────────────

def ingest_pdf(filepath: str) -> Dict:
    print(f"\n{'='*50}")
    print(f"INGESTING: {os.path.basename(filepath)}")
    print(f"{'='*50}\n")

    log_memory("start")
    pages = parse_pdf(filepath)
    log_memory("after parse_pdf")
    if not pages:
        return {"error": "No readable text found in PDF"}

    chunks = chunk_pages(pages)
    log_memory("after chunk_pages")
    if not chunks:
        return {"error": "Could not create chunks from PDF"}

    client = get_chroma_client()
    log_memory("after get_chroma_client")

    collection = get_collection(client)
    log_memory("after get_collection")

    store_in_chroma(chunks, collection)
    log_memory("after store_in_chroma")

    build_bm25_index(chunks)
    log_memory("after build_bm25_index")

    result = {
        "filename": os.path.basename(filepath),
        "pages_parsed": len(pages),
        "chunks_created": len(chunks),
        "status": "success",
    }

    print(f"\nIngestion complete: {result}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = ingest_pdf(sys.argv[1])
        print(result)
    else:
        print("Usage: python -m backend.ingest <path_to_pdf>")