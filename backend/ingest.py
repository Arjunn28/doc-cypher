# ingest.py
# The ingestion pipeline — the entry point for every PDF uploaded to DocCypher.
# Takes a raw PDF, extracts text page by page, splits into overlapping chunks,
# creates embeddings, and stores in both ChromaDB (vector) and BM25 (keyword).
# Why two indexes? Vector search finds semantically similar content.
# BM25 finds exact keyword matches. Together they cover each other's blind spots.

import os
import json
import hashlib
import fitz  # PyMuPDF — best PDF parser available, handles complex layouts
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

# Where ChromaDB persists its data between restarts
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "../data/chroma")

# Where uploaded PDFs are stored
UPLOAD_PATH = os.path.join(os.path.dirname(__file__), "../uploads")

# Where we persist the BM25 index and chunk metadata
BM25_PATH = os.path.join(os.path.dirname(__file__), "../data/bm25")

# The embedding model — free, runs locally, no API key needed.
# all-MiniLM-L6-v2 is the industry standard for RAG embeddings:
# fast, small (80MB), and produces 384-dimensional embeddings.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk size in characters. 1000 chars ≈ 200-250 tokens.
# Too small = not enough context. Too large = too much noise.
CHUNK_SIZE = 1000

# Overlap between consecutive chunks.
# Why overlap? So sentences at chunk boundaries don't lose context.
CHUNK_OVERLAP = 200

# ─────────────────────────────────────────────
# Initialize components
# ─────────────────────────────────────────────

def get_chroma_client():
    """Returns a persistent ChromaDB client."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_PATH)

# def get_embedding_model():
#     """Loads the sentence transformer model."""
#     return SentenceTransformer(EMBEDDING_MODEL)

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(">> Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(">> Embedding model loaded.")
    return _embedding_model

def get_collection(client, collection_name: str = "doccypher"):
    """
    Gets or creates the ChromaDB collection.
    A collection is like a table — stores embeddings + metadata together.
    """
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}  # cosine similarity for text
    )

# ─────────────────────────────────────────────
# Step 1: PDF Parsing
# ─────────────────────────────────────────────

def parse_pdf(filepath: str) -> List[Dict]:
    """
    Extracts text from each page of a PDF.
    Returns a list of dicts: {page_num, text, filename}
    Why page by page? So we can cite the exact page in every answer.
    """
    doc = fitz.open(filepath)
    filename = os.path.basename(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")  # extract as plain text

        # Skip pages with no meaningful content
        if text.strip() and len(text.strip()) > 50:
            pages.append({
                "page_num": page_num + 1,  # 1-indexed for human readability
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
    """
    Splits each page's text into overlapping chunks.
    
    Why overlapping chunks? If a sentence spans a chunk boundary,
    overlap ensures neither chunk loses the full context of that sentence.
    
    Each chunk keeps track of: which document, which page, which position.
    This is what enables citations — every chunk knows exactly where it came from.
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        text = page["text"]
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]

            # Only keep chunks with meaningful content
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

            # Move forward by (CHUNK_SIZE - CHUNK_OVERLAP) for overlap
            start += CHUNK_SIZE - CHUNK_OVERLAP

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")
    return chunks

# ─────────────────────────────────────────────
# Step 3: Store in ChromaDB (vector search)
# ─────────────────────────────────────────────

def store_in_chroma(chunks: List[Dict], collection) -> None:
    """
    Embeds each chunk and stores in ChromaDB.
    
    ChromaDB stores:
    - The embedding vector (for similarity search)
    - The original text (returned with results)
    - Metadata (filename, page, chunk index — for citations)
    
    Why ChromaDB? It's free, runs locally, persists to disk,
    and has a simple Python API. Production alternative: Pinecone, Weaviate.
    """
    model = get_embedding_model()

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {
            "filename": chunk["filename"],
            "page_num": chunk["page_num"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    print(f"Embedding {len(texts)} chunks (this takes 10-30 seconds)...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    # Add to ChromaDB in batches to avoid memory issues with large PDFs
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch_ids = ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size]

        # Check for existing IDs to avoid duplicates
        existing = collection.get(ids=batch_ids)
        existing_ids = set(existing["ids"])
        
        new_indices = [
            j for j, id_ in enumerate(batch_ids)
            if id_ not in existing_ids
        ]

        if new_indices:
            collection.add(
                ids=[batch_ids[j] for j in new_indices],
                documents=[batch_texts[j] for j in new_indices],
                embeddings=[batch_embeddings[j] for j in new_indices],
                metadatas=[batch_metadatas[j] for j in new_indices],
            )

    print(f"Stored {len(chunks)} chunks in ChromaDB")

# ─────────────────────────────────────────────
# Step 4: Store in BM25 (keyword search)
# ─────────────────────────────────────────────

def build_bm25_index(chunks: List[Dict]) -> None:
    """
    Builds a BM25 keyword search index from all chunks.
    
    BM25 (Best Match 25) is a classical IR algorithm used by 
    Elasticsearch and Solr. It scores documents based on term 
    frequency and inverse document frequency.
    
    Why BM25 alongside vectors? If someone searches for "Section 4.2" 
    or a specific product code, vector search might miss it because 
    there's no semantic meaning. BM25 finds exact matches reliably.
    
    We persist the corpus and metadata to disk so it survives restarts.
    """
    os.makedirs(BM25_PATH, exist_ok=True)

    # Load existing index if it exists
    corpus_path = os.path.join(BM25_PATH, "corpus.json")
    existing_corpus = []
    existing_chunks = []

    if os.path.exists(corpus_path):
        with open(corpus_path, "r") as f:
            saved = json.load(f)
            existing_corpus = saved.get("corpus", [])
            existing_chunks = saved.get("chunks", [])

    # Add new chunks (avoid duplicates by chunk_id)
    existing_ids = {c["chunk_id"] for c in existing_chunks}
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        print("BM25 index already up to date.")
        return

    # Tokenize: split on whitespace and lowercase
    # Why simple tokenization? Fast, works well for BM25,
    # and avoids dependency on NLTK or spaCy
    new_tokenized = [chunk["text"].lower().split() for chunk in new_chunks]

    all_corpus = existing_corpus + new_tokenized
    all_chunks = existing_chunks + new_chunks

    # Save to disk
    with open(corpus_path, "w") as f:
        json.dump({
            "corpus": all_corpus,
            "chunks": all_chunks,
        }, f)

    print(f"BM25 index built with {len(all_corpus)} chunks total")

# ─────────────────────────────────────────────
# Main ingestion function — called by the API
# ─────────────────────────────────────────────

def ingest_pdf(filepath: str) -> Dict:
    """
    Full ingestion pipeline for a single PDF.
    Called by the FastAPI endpoint when a user uploads a document.
    
    Returns a summary of what was ingested.
    """
    print(f"\n{'='*50}")
    print(f"INGESTING: {os.path.basename(filepath)}")
    print(f"{'='*50}\n")

    # Step 1: Parse
    pages = parse_pdf(filepath)
    if not pages:
        return {"error": "No readable text found in PDF"}

    # Step 2: Chunk
    chunks = chunk_pages(pages)
    if not chunks:
        return {"error": "Could not create chunks from PDF"}

    # Step 3: Store in ChromaDB
    client = get_chroma_client()
    collection = get_collection(client)
    store_in_chroma(chunks, collection)

    # Step 4: Build BM25 index
    build_bm25_index(chunks)

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