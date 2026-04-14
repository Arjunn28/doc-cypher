import { useState, useEffect, useRef } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SUGGESTED = [
  "What is this document about?",
  "Summarize the key findings",
  "Who are the key people mentioned?",
  "What are the main conclusions?",
];

function InfoPanel({ onClose }) {
  return (
    <div style={{
      position: "fixed", inset: 0, background: "rgba(0,0,0,0.4)",
      zIndex: 500, display: "flex", justifyContent: "flex-end"
    }} onClick={onClose}>
      <div style={{
        width: "min(480px, 100%)", background: "white", height: "100%",
        overflowY: "auto", padding: "2rem", boxShadow: "-4px 0 24px rgba(0,0,0,0.15)"
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
          <h2 style={{ fontSize: "1.3rem", fontWeight: 700, color: "#1e2235" }}>What is DocCypher?</h2>
          <button onClick={onClose} style={{ background: "none", border: "none", fontSize: "1.5rem", cursor: "pointer", color: "#6b7280" }}>✕</button>
        </div>

        {[
          {
            title: "Overview",
            content: "DocCypher is an intelligent document Q&A system. Upload any PDF — research papers, reports, manuals — and ask questions in plain English. DocCypher finds the most relevant passages and generates a grounded answer with citations pointing to exact pages."
          },
          {
            title: "How to use",
            content: "1. Upload one or more PDFs using the sidebar.\n2. Optionally select which document(s) to search — or leave it on 'All documents' to search across everything.\n3. Type your question or click a suggested query.\n4. The answer streams in real time with inline citations like [1] [2].\n5. Click any citation chip to preview the exact source text."
          },
          {
            title: "The retrieval pipeline",
            content: "Most RAG systems use only vector (semantic) search. DocCypher uses hybrid retrieval:\n\n• BM25 keyword search catches exact matches — product codes, names, section numbers.\n• Vector search catches semantic similarity — 'car' matches 'automobile'.\n• Reciprocal Rank Fusion (RRF) combines both result lists using rank position, not raw scores.\n\nThis covers both failure modes simultaneously."
          },
          {
            title: "Cross-encoder reranking",
            content: "After retrieval returns 20 candidate chunks, a cross-encoder model scores each chunk against the query jointly — seeing both together, not separately. This is significantly more accurate than the initial retrieval. Only the top 5 chunks after reranking are sent to the LLM. Result: answers grounded in the most precise passages, not just the most similar ones."
          },
          {
            title: "Citation enforcement",
            content: "Every answer must cite its sources inline. The LLM is prompted with numbered source chunks and strict rules to cite after every claim. Citations marked ★ were found by both BM25 and vector search independently — these carry the highest confidence."
          },
          {
            title: "What to expect",
            content: "• Answers are grounded — no hallucination from outside knowledge.\n• First load takes 10-20 seconds while models initialize.\n• Quality depends on PDF text clarity — scanned image PDFs without OCR won't work well.\n• For best results, ask specific questions rather than very broad ones."
          },
          {
            title: "Tech stack",
            content: "• Embeddings: sentence-transformers (all-MiniLM-L6-v2)\n• Vector DB: ChromaDB\n• Keyword search: BM25Okapi (rank_bm25)\n• Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2\n• LLM: Llama 3.3 70B via Groq API\n• Backend: FastAPI\n• Frontend: React + Vite"
          },
        ].map((section, i) => (
          <div key={i} style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "13px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.06em", color: "#9ca3af", marginBottom: "6px" }}>
              {section.title}
            </h3>
            <p style={{ fontSize: "14px", color: "#374151", lineHeight: 1.75, whiteSpace: "pre-wrap" }}>
              {section.content}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth]                   = useState(null);
  const [documents, setDocuments]             = useState([]);
  const [selectedDocs, setSelectedDocs]       = useState([]);  // array now
  const [selectedFile, setSelectedFile]       = useState(null);
  const [uploading, setUploading]             = useState(false);
  const [uploadMsg, setUploadMsg]             = useState(null);
  const [query, setQuery]                     = useState("");
  const [answer, setAnswer]                   = useState("");
  const [streaming, setStreaming]             = useState(false);
  const [citations, setCitations]             = useState([]);
  const [activeCitation, setActiveCitation]   = useState(null);
  const [error, setError]                     = useState(null);
  const [dragging, setDragging]               = useState(false);
  const [showInfo, setShowInfo]               = useState(false);
  const fileInputRef                          = useRef(null);

  useEffect(() => { fetchHealth(); fetchDocuments(); }, []);

  async function fetchHealth() {
    try { const res = await axios.get(`${API}/health`); setHealth(res.data); }
    catch { setHealth({ status: "offline" }); }
  }

  async function fetchDocuments() {
    try { const res = await axios.get(`${API}/documents`); setDocuments(res.data.documents || []); }
    catch { setDocuments([]); }
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setUploading(true); setUploadMsg(null); setError(null);
    const formData = new FormData();
    formData.append("file", selectedFile);
    try {
      const res = await axios.post(`${API}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" }, timeout: 120000,
      });
      setUploadMsg(`✓ ${res.data.details.filename} — ${res.data.details.chunks_created} chunks indexed`);
      setSelectedFile(null);
      await fetchDocuments();
    } catch { setError("Upload failed. Check the file is a valid PDF."); }
    finally { setUploading(false); }
  }

  async function handleDelete(filename) {
    if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;
    try {
      await axios.delete(`${API}/documents/${encodeURIComponent(filename)}`);
      setSelectedDocs(prev => prev.filter(d => d !== filename));
      await fetchDocuments();
    } catch { setError(`Failed to delete ${filename}.`); }
  }

  function toggleDocSelection(filename) {
    setSelectedDocs(prev =>
      prev.includes(filename)
        ? prev.filter(d => d !== filename)
        : [...prev, filename]
    );
  }

  function selectAll() { setSelectedDocs(documents.map(d => d.filename)); }
  function clearAll() { setSelectedDocs([]); }

  async function handleQuery(q = query) {
    if (!q.trim() || streaming) return;
    setQuery(q); setAnswer(""); setCitations([]); setActiveCitation(null); setError(null); setStreaming(true);

    const params = new URLSearchParams({ query: q });
    if (selectedDocs.length > 0) params.append("filename_filter", selectedDocs.join(","));

    try {
      const response = await fetch(`${API}/stream?${params.toString()}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();
        // for (const line of lines) {
        //   if (!line.trim()) continue;
        //   try {
        //     const data = JSON.parse(line);
        //     if (data.type === "citations") setCitations(data.citations);
        //     else if (data.type === "token") setAnswer(p => p + data.content);
        //     else if (data.type === "error") setError(data.message);
        //   } catch {}
        // }
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            try {
              const data = JSON.parse(trimmed);

              console.log("STREAM:", data); // 👈 keep this for now

              if (data.type === "citations") {
                setCitations(data.citations);
              } else if (data.type === "token") {
                setAnswer(prev => prev + data.content);
              } else if (data.type === "error") {
                setError(data.message);
              } else if (data.type === "done") {
                setStreaming(false);
              }

            } catch (err) {
              console.error("JSON parse failed:", trimmed);
            }
          }
      }
    } catch { setError("Query failed. Make sure the backend is running."); }
    finally { setStreaming(false); }
  }

  function handleDrop(e) {
    e.preventDefault(); setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file?.name.endsWith(".pdf")) setSelectedFile(file);
  }

  const isOnline = health?.status === "online";

  const scopeLabel = selectedDocs.length === 0
    ? `Searching all ${documents.length} document${documents.length !== 1 ? "s" : ""}`
    : selectedDocs.length === 1
      ? `Scoped to: ${selectedDocs[0]}`
      : `Scoped to ${selectedDocs.length} documents`;

  return (
    <div className="app">
      {showInfo && <InfoPanel onClose={() => setShowInfo(false)} />}

      <nav className="topnav">
        <div className="nav-brand">
          <h1>DocCypher</h1>
          <span>Hybrid RAG · BM25 + Vector · Cross-encoder Reranking</span>
        </div>
        <button
          onClick={() => setShowInfo(true)}
          style={{
            padding: "6px 16px", background: "rgba(255,255,255,0.1)",
            color: "white", border: "1px solid rgba(255,255,255,0.2)",
            borderRadius: "20px", fontSize: "13px", cursor: "pointer",
            fontFamily: "inherit", fontWeight: 500,
          }}
        >
          What is DocCypher?
        </button>
      </nav>

      <div className="safety-banner">
        <span className="safety-icon">⚠️</span>
        <span>
          <strong>Privacy notice:</strong> Do not upload personal or sensitive documents
          (passports, bank statements, medical records). Treat this as a public demo environment.
        </span>
      </div>

      <div className="main">
        <aside className="sidebar">

          <div className={`status-pill ${isOnline ? "" : "offline"}`}>
            <div className={`status-dot ${isOnline ? "" : "offline"}`} />
            {isOnline ? "Backend online" : "Backend offline"}
          </div>

          {/* Upload */}
          <div className="sidebar-section">
            <h3>Upload PDF</h3>
            <div
              className={`upload-zone ${dragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input ref={fileInputRef} type="file" accept=".pdf"
                onChange={e => setSelectedFile(e.target.files[0])} />
              <div className="upload-zone-icon">📄</div>
              <div className="upload-zone-text">
                {selectedFile
                  ? <strong>{selectedFile.name}</strong>
                  : <><strong>Choose a PDF</strong><br />or drag and drop</>
                }
              </div>
            </div>
            {uploadMsg && <div className="success-banner">{uploadMsg}</div>}
            {error && <div className="error-banner">{error}</div>}
            <button className="upload-btn" onClick={handleUpload} disabled={!selectedFile || uploading}>
              {uploading ? <><span className="spinner" /> &nbsp;Ingesting...</> : "Ingest PDF"}
            </button>
          </div>

          {/* Documents + scope selector */}
          <div className="sidebar-section">
            <h3>Documents</h3>

            {documents.length === 0 ? (
              <div className="doc-scope-label">No documents yet. Upload a PDF above.</div>
            ) : (
              <>
                <div className="doc-scope-label">
                  {selectedDocs.length === 0
                    ? `Tick documents to scope your search. Currently searching all ${documents.length}.`
                    : `${selectedDocs.length} of ${documents.length} selected for search.`
                  }
                </div>

                {documents.length > 1 && (
                  <div style={{ display: "flex", gap: 6, marginBottom: 8 }}>
                    <button onClick={selectAll} style={{
                      flex: 1, fontSize: 11, padding: "4px 0",
                      background: "#f3f4f6", border: "1px solid #e5e7eb",
                      borderRadius: 6, cursor: "pointer", fontFamily: "inherit", color: "#374151"
                    }}>Select all</button>
                    <button onClick={clearAll} style={{
                      flex: 1, fontSize: 11, padding: "4px 0",
                      background: "#f3f4f6", border: "1px solid #e5e7eb",
                      borderRadius: 6, cursor: "pointer", fontFamily: "inherit", color: "#374151"
                    }}>Clear</button>
                  </div>
                )}

                <div className="doc-list">
                  {documents.map((doc, i) => (
                    <div key={i} className={`doc-item ${selectedDocs.includes(doc.filename) ? "active" : ""}`}>
                      {/* Checkbox for scope selection */}
                      <input
                        type="checkbox"
                        checked={selectedDocs.includes(doc.filename)}
                        onChange={() => toggleDocSelection(doc.filename)}
                        style={{ cursor: "pointer", flexShrink: 0 }}
                      />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div className="doc-name">{doc.filename}</div>
                        <div className="doc-meta">{doc.page_count}p · {doc.chunks} chunks</div>
                      </div>
                      {/* Download button */}
                      <a
                        href={`${API}/documents/${encodeURIComponent(doc.filename)}/download`}
                        download={doc.filename}
                        title="Download PDF"
                        style={{
                          fontSize: 14, textDecoration: "none", opacity: 0.5,
                          cursor: "pointer", flexShrink: 0,
                        }}
                        onClick={e => e.stopPropagation()}
                      >
                        ⬇
                      </a>
                      {/* Delete button */}
                      <button
                        onClick={e => { e.stopPropagation(); handleDelete(doc.filename); }}
                        title="Delete document"
                        style={{
                          background: "none", border: "none", cursor: "pointer",
                          fontSize: 13, opacity: 0.4, flexShrink: 0, padding: "0 2px",
                          color: "#ef4444",
                        }}
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

        </aside>

        <main className="content">

          {/* Scope bar */}
          <div className={`scope-bar ${selectedDocs.length === 0 ? "all" : ""}`}>
            <span>{selectedDocs.length === 0 ? "🔍" : "🎯"}</span>
            <span style={{ flex: 1 }}>
              {selectedDocs.length === 0
                ? `Searching all documents — tick documents in the sidebar to narrow results`
                : selectedDocs.length === 1
                  ? `Scoped to: ${selectedDocs[0]} — answers will only use this document`
                  : `Scoped to ${selectedDocs.length} documents: ${selectedDocs.map(d => d.replace(".pdf", "")).join(", ")}`
              }
            </span>
            {selectedDocs.length > 0 && (
              <span onClick={clearAll} style={{ cursor: "pointer", opacity: 0.7, fontSize: 12 }}>
                Clear ✕
              </span>
            )}
          </div>

          {/* Query */}
          <div className="query-box">
            {documents.length > 0 && (
              <div className="suggested-row">
                {SUGGESTED.map((s, i) => (
                  <button key={i} className="suggested-chip"
                    onClick={() => handleQuery(s)} disabled={streaming}>{s}</button>
                ))}
              </div>
            )}
            <div className="query-row">
              <input
                className="query-input"
                type="text"
                placeholder={
                  documents.length === 0 ? "Upload a PDF to get started..." :
                  selectedDocs.length === 1 ? `Ask anything about ${selectedDocs[0]}...` :
                  selectedDocs.length > 1 ? `Ask anything across ${selectedDocs.length} selected documents...` :
                  "Ask anything across all your documents..."
                }
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleQuery()}
                disabled={streaming || documents.length === 0}
              />
              <button className="ask-btn"
                onClick={() => handleQuery()}
                disabled={streaming || !query.trim() || documents.length === 0}>
                {streaming ? <><span className="spinner" /> Thinking</> : "Ask →"}
              </button>
            </div>
          </div>

          {/* Answer */}
          <div className="answer-panel">
            <div className="panel-label">Answer</div>
            {!answer && !streaming && (
              <div className="answer-placeholder">
                {documents.length === 0
                  ? "Upload a PDF on the left to begin."
                  : "Your answer will appear here with inline citations like [1] [2]."
                }
              </div>
            )}
            {(answer || streaming) && (
              <div className="answer-text">
                {answer}
                {streaming && <span className="cursor" />}
              </div>
            )}
          </div>

          {/* Citations */}
          {citations.length > 0 && (
            <div className="citations-panel">
              <div className="panel-label">Sources — click any to preview</div>
              <div className="citation-chips">
                {citations.map((c, i) => (
                  <div key={i}
                    className={`citation-chip ${c.found_by_both ? "both" : ""} ${activeCitation?.citation_id === c.citation_id ? "active" : ""}`}
                    onClick={() => setActiveCitation(activeCitation?.citation_id === c.citation_id ? null : c)}
                  >
                    <span className="citation-num">{c.citation_id}</span>
                    <span>{c.filename.replace(".pdf", "")}</span>
                    <span style={{ opacity: 0.5 }}>p{c.page_num}</span>
                    <span style={{ opacity: 0.4, fontSize: 11 }}>{c.reranker_score.toFixed(2)}</span>
                    {c.found_by_both && <span title="Found by both retrievers">★</span>}
                  </div>
                ))}
              </div>
              {activeCitation && (
                <div className="citation-detail">
                  <strong>
                    {activeCitation.citation_id} · {activeCitation.filename}, page {activeCitation.page_num}
                    {activeCitation.found_by_both && " · ★ found by both retrievers"}
                  </strong>
                  <br /><br />
                  {activeCitation.preview}...
                </div>
              )}
              <div className="retrieval-tags">
                <span className="rtag">Hybrid BM25 + Vector</span>
                <span className="rtag">RRF fusion</span>
                <span className="rtag green">Cross-encoder reranked</span>
                <span className="rtag green">
                  {citations.filter(c => c.found_by_both).length} found by both retrievers
                </span>
              </div>
            </div>
          )}

        </main>
      </div>

      <div className="footer">DocCypher · Built by Arjun · 2026</div>
    </div>
  );
}