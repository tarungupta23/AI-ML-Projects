"""
embedding.py
Wraps the sentence-transformers all-MiniLM-L6-v2 model as a LangChain
embeddings object, used for both index building and query time.
"""

from langchain_huggingface import HuggingFaceEmbeddings

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings_instance = None


def get_embeddings():
    """Returns a cached HuggingFaceEmbeddings instance (loads model once)."""
    global _embeddings_instance
    if _embeddings_instance is None:
        print(f"[embedding] Loading model: {MODEL_NAME}")
        _embeddings_instance = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    return _embeddings_instance
