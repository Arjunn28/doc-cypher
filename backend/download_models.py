# download_models.py
# Pre-downloads HuggingFace models during Render's build step.
# This prevents cold-start timeouts when the first request arrives.

print("Downloading embedding model...")
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model ready.")

print("Downloading reranker model...")
from sentence_transformers import CrossEncoder
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("Reranker model ready.")

print("All models downloaded.")