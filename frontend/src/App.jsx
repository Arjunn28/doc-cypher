import { useState, useEffect, useRef } from "react";
import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const SUGGESTED = [
  "What is this document about?",
  "Summarize the key points",
  "What are the main findings?",
  "Who are the key people mentioned?",
];

export default function App() {
  const [health, setHealth] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState(null);
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [citations, setCitations] = useState([]);
  const [selectedCitation, setSelectedCitation] = useState(null);
  const [stats, setStats] = useState(null);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);
  const answerRef = useRef(null);

  useEffect(() => {
    fetchHealth();
    fetchDocuments();
  }, []);

  useEffect(() => {
    if (answerRef.current) {
      answerRef.current.scrollTop = answerRef.current.scrollHeight;
    }
  }, [answer]);

  async function fetchHealth() {
    try {
      const res = await axios.get(`${API}/health`);
      setHealth(res.data);
    } catch {
      setHealth({ status: "offline" });
    }
  }

  async function fetchDocuments() {
    try {
      const res = await axios.get(`${API}/documents`);
      setDocuments(res.data.documents || []);
    } catch {
      setDocuments([]);
    }
  }

  async function handleUpload() {
    if (!selectedFile) return;
    setUploading(true);
    setUploadMsg(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await axios.post(`${API}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 120000,
      });
      setUploadMsg(`✓ ${res.data.details.filename} ingested — ${res.data.details.chunks_created} chunks created`);
      setSelectedFile(null);
      await fetchDocuments();
    } catch (e) {
      setError("Upload failed. Make sure the file is a valid PDF.");
    } finally {
      setUploading(false);
    }
  }

  async function handleQuery(q = query) {
    if (!q.trim() || streaming) return;
    setQuery(q);
    setAnswer("");
    setCitations([]);
    setSelectedCitation(null);
    setStats(null);
    setError(null);
    setStreaming(true);

    try {
      const response = await fetch(`${API}/stream?query=${encodeURIComponent(q)}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const data = JSON.parse(line);

            if (data.type === "citations") {
              setCitations(data.citations);
              setStats({
                retrieved: data.citations.length,
                reranked: data.citations.length,
              });
            } else if (data.type === "token") {
              setAnswer(prev => prev + data.content);
            } else if (data.type === "error") {
              setError(data.message);
            }
          } catch {
            // incomplete JSON line, skip
          }
        }
      }
    } catch (e) {
      setError("Query failed. Make sure the backend is running.");
    } finally {
      setStreaming(false);
    }
  }

  function handleDrop(e) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file && file.name.endsWith(".pdf")) {
      setSelectedFile(file);
    }
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>DocCypher</h1>
          <p>Hybrid RAG · BM25 + Vector · Cross-encoder Reranking</p>
        </div>
        <div className="status-badge">
          <div className={`status-dot ${health?.status === "online" ? "" : "offline"}`} />
          {health?.status === "online" ? "Online" : "Offline"}
        </div>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="layout">
        {/* Left sidebar */}
        <div>
          {/* Upload */}
          <div className="card">
            <h2>Upload PDF</h2>

            <div
              className={`upload-area ${dragging ? "dragging" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={e => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={e => setSelectedFile(e.target.files[0])}
              />
              <div className="upload-icon">📄</div>
              <div className="upload-text">
                {selectedFile
                  ? <strong>{selectedFile.name}</strong>
                  : <><strong>Choose a PDF</strong> or drag it here</>
                }
              </div>
            </div>

            {uploadMsg && <div className="success-banner">{uploadMsg}</div>}

            <button
              className="upload-btn"
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
            >
              {uploading
                ? <><span className="spinner" />Ingesting...</>
                : "Ingest PDF"
              }
            </button>
          </div>

          {/* Document list */}
          <div className="card">
            <h2>Ingested documents ({documents.length})</h2>
            {documents.length === 0 ? (
              <p style={{ fontSize: 13, color: "#aaa", fontStyle: "italic" }}>
                No documents yet. Upload a PDF to get started.
              </p>
            ) : (
              <ul className="doc-list">
                {documents.map((doc, i) => (
                  <li key={i} className="doc-item">
                    <span className="doc-icon">📑</span>
                    <div className="doc-info">
                      <div className="doc-name">{doc.filename}</div>
                      <div className="doc-meta">
                        {doc.page_count} pages · {doc.chunks} chunks
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        {/* Right — query + answer */}
        <div>
          <div className="card query-area">
            <h2>Ask a question</h2>

            {documents.length > 0 && (
              <div className="suggested-queries">
                {SUGGESTED.map((s, i) => (
                  <button
                    key={i}
                    className="suggested-btn"
                    onClick={() => handleQuery(s)}
                    disabled={streaming}
                  >
                    {s}
                  </button>
                ))}
              </div>
            )}

            <div className="query-input-row">
              <input
                className="query-input"
                type="text"
                placeholder={documents.length === 0
                  ? "Upload a PDF first..."
                  : "Ask anything about your documents..."
                }
                value={query}
                onChange={e => setQuery(e.target.value)}
                onKeyDown={e => e.key === "Enter" && handleQuery()}
                disabled={streaming || documents.length === 0}
              />
              <button
                className="ask-btn"
                onClick={() => handleQuery()}
                disabled={streaming || !query.trim() || documents.length === 0}
              >
                {streaming
                  ? <><span className="spinner" />Thinking</>
                  : "Ask"
                }
              </button>
            </div>
          </div>

          {/* Answer */}
          <div className="card answer-card">
            <h2>Answer</h2>

            {!answer && !streaming && (
              <div className="answer-placeholder">
                {documents.length === 0
                  ? "Upload a PDF and ask a question to get started."
                  : "Your answer will appear here with inline citations."
                }
              </div>
            )}

            {(answer || streaming) && (
              <div ref={answerRef} className="answer-text">
                {answer}
                {streaming && <span className="cursor" />}
              </div>
            )}

            {/* Citations */}
            {citations.length > 0 && (
              <div className="citations-section">
                <div className="citations-label">Sources retrieved</div>

                <div className="citation-chips">
                  {citations.map((c, i) => (
                    <div
                      key={i}
                      className={`citation-chip ${c.found_by_both ? "both" : ""}`}
                      onClick={() => setSelectedCitation(
                        selectedCitation?.citation_id === c.citation_id ? null : c
                      )}
                      title={c.found_by_both ? "Found by both BM25 and vector search" : ""}
                    >
                      <span className="citation-num">{c.citation_id}</span>
                      <span className="citation-file">
                        {c.filename.replace(".pdf", "")}
                      </span>
                      <span className="citation-page">p{c.page_num}</span>
                      <span className="citation-score">
                        {c.reranker_score.toFixed(2)}
                      </span>
                    </div>
                  ))}
                </div>

                {selectedCitation && (
                  <div className="citation-detail">
                    <strong>
                      {selectedCitation.citation_id} — {selectedCitation.filename}, page {selectedCitation.page_num}
                      {selectedCitation.found_by_both && " · found by both retrievers"}
                    </strong>
                    <br /><br />
                    {selectedCitation.preview}...
                  </div>
                )}

                <div className="retrieval-stats">
                  <span className="stat-pill">
                    Hybrid: BM25 + Vector
                  </span>
                  <span className="stat-pill">
                    RRF fusion
                  </span>
                  <span className="stat-pill highlight">
                    Cross-encoder reranked
                  </span>
                  <span className="stat-pill">
                    {citations.filter(c => c.found_by_both).length} found by both retrievers
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="footer">
        DocCypher · Built by Arjun · 2026
      </div>
    </div>
  );
}