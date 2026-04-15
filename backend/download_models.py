# download_models.py
# Pre-downloads models during Render's BUILD step, not at runtime.

import os

print("Downloading fastembed embedding model...")
from fastembed import TextEmbedding
# This downloads and caches the model to disk
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
# Force it to actually load by running a dummy embed
list(model.embed(["warmup"]))
print("Embedding model ready.")

print("All models downloaded.")