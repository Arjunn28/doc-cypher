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
        width: "min(440px, 100%)", background: "white", height: "100%",
        overflowY: "auto", padding: "2rem", boxShadow: "-4px 0 24px rgba(0,0,0,0.15)"
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.75rem" }}>
          <h2 style={{ fontSize: "1.1rem", fontWeight: 700, color: "#0f172a" }}>DocCypher</h2>
          <button onClick={onClose} style={{ background: "none", border: "none", fontSize: "1.5rem", cursor: "pointer", color: "#9ca3af" }}>✕</button>
        </div>

        {[
          {
            title: "What it does",
            content: "Upload a PDF and ask questions in plain English. DocCypher retrieves the most relevant passages and returns a grounded answer with page-level citations. No hallucination from outside knowledge."
          },
          {
            title: "How to use",
            content: "1. Drop a PDF into the sidebar.\n2. Tick documents to scope your search, or leave all selected.\n3. Type a question or pick a suggestion.\n4. Citations like [1] [2] link back to exact pages."
          },
          {
            title: "Why hybrid retrieval",
            content: "Most RAG systems use only vector search. DocCypher runs BM25 (keyword) and vector (semantic) search in parallel, then fuses results using Reciprocal Rank Fusion. BM25 catches exact matches. Vector catches meaning. Together they cover each other's blind spots."
          },
          {
            title: "Citation confidence",
            content: "Sources marked ★ were returned by both BM25 and vector search independently. These carry the highest confidence since two different retrieval methods agreed."
          },
          {
            title: "Tech stack",
            content: "Embeddings: HuggingFace all-MiniLM-L6-v2\nVector DB: ChromaDB\nKeyword search: BM25Okapi\nLLM: Llama 3.3 70B via Groq\nBackend: FastAPI\nFrontend: React + Vite"
          },
          {
            title: "Heads up",
            content: "This is a public demo. Avoid uploading sensitive documents. Uploaded files may be cleared on server restart."
          },
        ].map((section, i) => (
          <div key={i} style={{ marginBottom: "1.5rem", paddingBottom: "1.5rem", borderBottom: i < 5 ? "1px solid #f3f4f6" : "none" }}>
            <h3 style={{ fontSize: "11px", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.08em", color: "#94a3b8", marginBottom: "8px" }}>
              {section.title}
            </h3>
            <p style={{ fontSize: "14px", color: "#374151", lineHeight: 1.8, whiteSpace: "pre-wrap" }}>
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

      {/* <div className="safety-banner">
        <span className="safety-icon">⚠️</span>
        <span>
          <strong>Privacy notice:</strong> Do not upload personal or sensitive documents
          (passports, bank statements, medical records). Treat this as a public demo environment.
        </span>
      </div> */}

      {/* Persistence warning */}
      {/* <div style={{
        background: "#eff6ff",
        borderBottom: "1px solid #bfdbfe",
        padding: "8px 1.5rem",
        fontSize: "12.5px",
        color: "#1d4ed8",
        display: "flex",
        gap: 8,
        alignItems: "center",
      }}>
        <span>ℹ️</span>
        <span>
          <strong>Demo environment:</strong> Uploaded documents are stored temporarily.
          They may be cleared when the server restarts. Re-upload if documents disappear.
        </span>
      </div> */}

      <div className="main">
        <aside className="sidebar">

          <div className={`status-pill ${isOnline ? "" : "offline"}`}>
            <div className={`status-dot ${isOnline ? "" : "offline"}`} />
            {/* {isOnline ? "ONLINE" : "Backend offline"} */}
            {isOnline ? "● ONLINE" : "● OFFLINE"}
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