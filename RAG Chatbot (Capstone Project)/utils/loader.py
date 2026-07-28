"""
loader.py
Loads all PDF documents from the data/ directory to be used as the
RAG knowledge base. rag.pdf (the tutorial/structure reference) is
explicitly excluded so it never gets chunked or embedded.
"""

import os
from langchain_community.document_loaders import PyPDFLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

# Files that should NEVER be treated as knowledge-base content.
# rag.pdf is a structural/code reference only, not a source document.
EXCLUDED_FILES = {"rag.pdf"}


def load_documents(data_dir: str = DATA_DIR):
    """
    Loads every .pdf in data_dir (case-insensitive match), skipping
    anything listed in EXCLUDED_FILES. Returns a list of LangChain
    Document objects with page-level metadata (source, page).
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents = []
    pdf_files = sorted(
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    )

    if not pdf_files:
        print(f"[loader] No PDFs found in {data_dir}")
        return documents

    for filename in pdf_files:
        if filename.lower() in {x.lower() for x in EXCLUDED_FILES}:
            print(f"[loader] Skipping excluded reference file: {filename}")
            continue

        filepath = os.path.join(data_dir, filename)
        print(f"[loader] Loading: {filename}")
        loader = PyPDFLoader(filepath)
        pages = loader.load()

        # Tag every page with a clean source name for citations later
        for page in pages:
            page.metadata["source"] = filename

        documents.extend(pages)

    print(f"[loader] Loaded {len(documents)} pages from "
          f"{len(pdf_files) - sum(1 for f in pdf_files if f.lower() in EXCLUDED_FILES)} document(s).")
    return documents


if __name__ == "__main__":
    docs = load_documents()
    for d in docs[:3]:
        print(d.metadata, "->", d.page_content[:120].replace("\n", " "), "...")
