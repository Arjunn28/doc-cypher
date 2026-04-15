# download_models.py
# Pre-downloads models during Render's build step.
# Prevents cold-start timeouts on first request.

print("Downloading embedding model...")
from fastembed import TextEmbedding
TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("Embedding model ready.")

print("All models downloaded.")