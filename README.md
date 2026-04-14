# DocCypher: Hybrid RAG Document Intelligence System

> A document Q&A system built on a hybrid retrieval pipeline: BM25 keyword search + dense vector search, fused with Reciprocal Rank Fusion and reranked by a cross-encoder model. Upload any PDF and interrogate it in plain English with grounded, citation-enforced answers.

**Live Demo:** <!-- add link here -->  
**Backend API:** <!-- add link here -->  
**Author:** Arjun A N

---

## What is this?

Most RAG systems run a single vector similarity search, pass the top chunks to an LLM, and hope the retrieval was good enough. The failure modes are well known: exact keyword matches get missed, semantically adjacent but irrelevant chunks rank high, and the LLM has no way to know the retrieval was bad.

DocCypher addresses this at every stage of the pipeline:

- **Hybrid retrieval** covers two distinct failure modes simultaneously
- **Reciprocal Rank Fusion** merges ranked lists without requiring score normalization across incompatible spaces
- **Cross-encoder reranking** scores each candidate chunk against the query jointly, not independently
- **Citation enforcement** in the prompt means every claim in the answer is traceable to an exact page

The result is a system where answers are grounded in the most precisely relevant passages, not just the most semantically similar ones.

---

## Project Preview

<!-- Add screenshots from assets/ folder here -->

![DocCypher Interface](assets/doccypher_main.png)

---

## The retrieval pipeline

### Stage 1: Hybrid search

Two retrievers run in parallel on every query:

**BM25 (Okapi BM25)** is a sparse keyword retrieval method. It tokenizes both the query and the corpus and scores documents by term frequency and inverse document frequency. It excels at exact matches: product codes, author names, section numbers, specific terminology.

**Vector search (ChromaDB + sentence-transformers)** encodes the query and all corpus chunks into a shared dense embedding space. Similarity is computed via cosine distance. It captures semantic relationships: "car" matches "automobile", "reduce latency" matches "improve response time".

Neither retriever is sufficient alone. BM25 fails on paraphrase and semantic variation. Vector search fails on exact keyword requirements. Running both in parallel covers both failure modes simultaneously.

### Stage 2: Reciprocal Rank Fusion

After retrieval, two ranked lists of up to 10 candidates each are merged using Reciprocal Rank Fusion (RRF):

```
score(d) = Σ 1 / (k + rank(d))
```

where `k = 60` is the empirically validated smoothing constant from the original RRF paper (Cormack, Clarke & Buettcher, 2009).

RRF uses only rank position, not raw scores. This sidesteps the fundamental incompatibility between BM25's term frequency scores and cosine similarity scores, which exist on completely different scales. Chunks found by both retrievers independently receive additive RRF contributions and are flagged with a confidence marker in the UI.

### Stage 3: Cross-encoder reranking

The fused candidate pool of up to 20 chunks is passed to a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for reranking.

The architectural difference matters here. Bi-encoder models (used for vector search) encode the query and document separately and compare their embeddings. This is fast but loses cross-attention signal between the two.

A cross-encoder sees the query and candidate chunk concatenated in a single forward pass. It can attend across both simultaneously, producing a relevance score that captures fine-grained interaction between query tokens and document tokens. This is significantly more accurate than bi-encoder similarity at retrieval time.

Only the top 5 chunks after reranking are sent to the LLM. This keeps the context window tight and the answer grounded.

### Stage 4: Citation-enforced generation

The top 5 reranked chunks are numbered and injected into a structured prompt with strict citation rules. The LLM is instructed to cite inline after every claim using `[1]`, `[2]`, `[3]` notation and to end with a "Sources used:" section. It is explicitly prohibited from using outside knowledge.

When a filename filter is active (user has scoped the query to specific documents), the filter is applied at two layers: at retrieval time, so only chunks from the selected files enter the pipeline; and at the prompt level, with explicit scope instructions to the LLM. This dual enforcement prevents cross-document contamination.

Answers stream token by token via FastAPI's `StreamingResponse`, so the response appears in real time rather than after a 5-10 second wait.

---

## Multi-document scoping

DocCypher supports querying across all ingested PDFs simultaneously or scoping to one or more specific documents. The scope filter is passed as a comma-separated query parameter (`filename_filter`) and applied at the BM25 corpus level (filtered corpus rebuild), at the ChromaDB query level (using `$in` operator on metadata), and as a post-rerank safety filter before prompt construction.

This means a user researching five papers can ask a question scoped to a single paper and get an answer grounded only in that document's content.

---

## Document ingestion pipeline

When a PDF is uploaded:

1. `pdfplumber` extracts text page by page
2. Text is chunked into overlapping windows (512 tokens, 50-token stride) to preserve context across chunk boundaries
3. Each chunk is embedded using `all-MiniLM-L6-v2` and stored in ChromaDB with filename and page number metadata
4. The full tokenized corpus is serialized to disk as `corpus.json` for BM25 index reconstruction on server restart

Chunk metadata (filename, page number, chunk index) is stored alongside embeddings, enabling precise citation to the exact source page.

---

## Tech stack

| Layer | Technology | Role |
| --- | --- | --- |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Dense chunk and query encoding |
| Vector store | ChromaDB | Persistent vector index with metadata filtering |
| Keyword search | BM25Okapi (rank_bm25) | Sparse term-frequency retrieval |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | Joint query-document relevance scoring |
| Fusion | Reciprocal Rank Fusion | Score-agnostic ranked list merging |
| LLM | Llama 3.3 70B (Groq API) | Citation-enforced answer generation |
| Backend | FastAPI | REST API + streaming response |
| Frontend | React + Vite | Real-time chat interface |
| PDF parsing | pdfplumber | Text extraction with page-level metadata |
| Backend hosting | Render | Always-on API server |
| Frontend hosting | Vercel | Global CDN deployment |

**Total infrastructure cost: $0**

---

## Running locally

**Prerequisites:** Python 3.11+, Node.js 18+

```bash
# 1. Clone the repo
git clone https://github.com/Arjunn28/doc-cypher.git
cd doc-cypher

# 2. Set up Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Configure environment variables
cp .env.example .env
# Add your GROQ_API_KEY to .env
```

`.env` file:

```
GROQ_API_KEY=your_groq_api_key
```

```bash
# 4. Start the backend
uvicorn backend.main:app --reload --port 8000

# 5. Start the frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` to use the interface.  
Open `http://localhost:8000/docs` for the auto-generated API documentation.

---

## API endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/ingest` | Upload and ingest a PDF into the retrieval index |
| GET | `/stream` | Stream an answer with citations (supports `filename_filter`) |
| GET | `/documents` | List all ingested documents with chunk counts |
| DELETE | `/documents/{filename}` | Remove a document from the index |

---

## Project structure

```
doc-cypher/
├── backend/
│   ├── main.py           # FastAPI app, routing, streaming endpoint
│   ├── ingest.py         # PDF parsing, chunking, embedding, indexing
│   ├── retriever.py      # BM25 search, vector search, RRF fusion
│   ├── reranker.py       # Cross-encoder reranking, citation formatting
│   └── query_engine.py   # Pipeline orchestration, prompt builder, LLM streaming
├── frontend/
│   └── src/              # React + Vite interface
├── assets/               # Screenshots for README
├── requirements.txt
└── render.yaml
```

---

## What makes this RAG engineering, not just RAG

| Typical RAG project | DocCypher |
| --- | --- |
| Single vector similarity search | Hybrid BM25 + vector retrieval in parallel |
| Raw similarity scores merged directly | Reciprocal Rank Fusion on ranked positions |
| Bi-encoder scoring throughout | Cross-encoder reranking on top candidates |
| LLM may answer from outside knowledge | Citation enforcement via prompt + source injection |
| No document scoping | Per-query filename filter applied at retrieval + prompt layer |
| Answer returned after full generation | Token-by-token streaming via SSE |
| Single document support | Multi-document ingestion with selective scoping |

---

## Limitations

- Scanned PDFs without embedded text (image-only PDFs) will not extract correctly without an OCR preprocessing step
- First load takes 10-20 seconds while the cross-encoder model initializes
- Retrieval quality is bounded by chunk quality: very short or fragmented PDF text produces noisy chunks

---

## Author

**Arjun A N**  
[GitHub](https://github.com/Arjunn28) · [Live Demo](#) · [LinkedIn](https://www.linkedin.com/in/arjun-an/)

---

> Note on hosting: Backend runs on Render's free tier, which spins down after 15 minutes of inactivity. The first request after sleep takes around 60 seconds to wake up. This is a hosting constraint, not an application one.
