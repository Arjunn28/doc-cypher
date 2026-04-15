print("Using Groq API for embeddings — no models to download.")



# import os

# # Must set this BEFORE importing fastembed
# os.environ["FASTEMBED_CACHE_PATH"] = "/opt/render/project/src/.fastembed_cache"

# print("Downloading fastembed embedding model...")
# from fastembed import TextEmbedding

# model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
# list(model.embed(["warmup"]))
# print("Embedding model ready.")
# print("All models downloaded.")