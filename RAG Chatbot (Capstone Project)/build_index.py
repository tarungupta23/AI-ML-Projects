"""
build_index.py
End-to-end index build:
  1. Load PDFs from data/ (excluding rag.pdf)
  2. Split into chunks
  3. Embed with all-MiniLM-L6-v2
  4. Save a FAISS index + chunks.pkl into vector_db/
"""

import os
import pickle

from utils.loader import load_documents
from utils.splitter import split_documents
from utils.embedding import get_embeddings

from langchain_community.vectorstores import FAISS

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")
CHUNKS_PATH = os.path.join(VECTOR_DB_DIR, "chunks.pkl")


def main():
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    print("=== Step 1/4: Loading documents ===")
    documents = load_documents()
    if not documents:
        print("No documents found in data/ (besides any excluded reference "
              "files). Add your source PDFs to data/ and re-run.")
        return

    print("=== Step 2/4: Splitting into chunks ===")
    chunks = split_documents(documents)

    print("=== Step 3/4: Loading embedding model ===")
    embeddings = get_embeddings()

    print("=== Step 4/4: Building FAISS index ===")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nDone. FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Chunk metadata saved to: {CHUNKS_PATH}")
    print(f"Total chunks indexed: {len(chunks)}")


if __name__ == "__main__":
    main()
