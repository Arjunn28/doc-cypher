import os
import json
import time
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

# ─────────────────────────────────────────────
# Memory logging
# ─────────────────────────────────────────────

def log_memory(label):
    try:
        import resource
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"[MEM] {label}: {mem / 1024:.1f} MB")
    except Exception:
        pass

# ─────────────────────────────────────────────
# Gemini Embeddings — no local model, no RAM cost
#
# History:
#   1. sentence-transformers (local) -> OOM on Render's 512MB free tier
#   2. HuggingFace Inference API     -> free credits exhausted, 402 on every upload (Oct 2026)
#   3. Gemini Embedding API          -> current; free tier is rate-limited, not credit-based
#
# Model verified live: Oct 2026. If uploads start failing with a 404,
# check https://ai.google.dev/gemini-api/docs/embeddings for the current model ID.
# ─────────────────────────────────────────────

GEMINI_EMBED_MODEL = "gemini-embedding-2"
GEMINI_EMBED_DIM = 768
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _gemini_headers() -> Dict[str, str]:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set on the server.")
    return {"Content-Type": "application/json", "x-goog-api-key": key}


# Free tier limits (AI Studio, Oct 2026): 100 RPM, 30K TPM, 1K RPD.
# Each text in a batch counts as one request, and a 1500-char chunk is roughly
# 375 tokens, so the 30K TPM cap is hit at about 80 chunks per minute (~30 PDF
# pages). Larger PDFs need to wait out the minute window instead of failing.

MAX_RATE_LIMIT_RETRIES = 4
DEFAULT_RETRY_WAIT_SECONDS = 30


def _retry_wait_seconds(response: httpx.Response) -> float:
    """Reads Gemini's suggested wait (e.g. "retryDelay": "34s"), else a default."""
    try:
        for detail in response.json().get("error", {}).get("details", []):
            delay = detail.get("retryDelay")
            if delay:
                return float(delay.rstrip("s")) + 1
    except Exception:
        pass
    return DEFAULT_RETRY_WAIT_SECONDS


def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Sends one batch request to Gemini and returns one vector per input text."""
    payload = {
        "requests": [
            {
                "model": f"models/{GEMINI_EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "output_dimensionality": GEMINI_EMBED_DIM,
            }
            for t in texts
        ]
    }

    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        response = httpx.post(
            f"{GEMINI_BASE}/{GEMINI_EMBED_MODEL}:batchEmbedContents",
            headers=_gemini_headers(),
            json=payload,
            timeout=60,
        )

        if response.status_code == 200:
            return [e["values"] for e in response.json()["embeddings"]]

        if response.status_code == 429:
            # Daily quota exhausted: waiting a minute will not help, fail fast.
            if "PerDay" in response.text:
                raise RuntimeError(
                    "Daily embedding limit reached. The quota resets at midnight Pacific time."
                )
            if attempt < MAX_RATE_LIMIT_RETRIES:
                wait = _retry_wait_seconds(response)
                print(f"  Rate limited (per-minute). Waiting {wait:.0f}s before retry "
                      f"{attempt + 1}/{MAX_RATE_LIMIT_RETRIES}...")
                time.sleep(wait)
                continue
            raise RuntimeError("Embedding rate limit reached. Please wait a minute and try again.")

        raise RuntimeError(f"Gemini embedding error {response.status_code}: {response.text[:300]}")


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Embeds document chunks for storage."""
    return _embed_batch([f"title: none | text: {t}" for t in texts])


def get_query_embedding(query: str) -> List[float]:
    """Embeds a user query for retrieval."""
    return _embed_batch([f"task: search result | query: {query}"])[0]

# ─────────────────────────────────────────────
# ChromaDB
# ─────────────────────────────────────────────

def get_chroma_client():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

# "doccypher_v2" holds 768-dim Gemini vectors. The old "doccypher" collection
# held 384-dim MiniLM vectors, and mixing the two in one collection would fail.
def get_collection(client, collection_name: str = "doccypher_v2"):
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
    batch_size = 20

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

        try:
            batch_embeddings = get_embeddings(texts)
            print(f"  Embeddings received: {len(batch_embeddings)} vectors")
        except Exception as e:
            print(f"  ERROR getting embeddings: {e}")
            raise

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
