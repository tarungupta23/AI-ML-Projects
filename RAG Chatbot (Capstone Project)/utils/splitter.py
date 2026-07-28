"""
splitter.py
Splits loaded documents into overlapping chunks suitable for embedding.
chunk_size=600 / chunk_overlap=100 sits in the 500-800 char "PDF" band
from the reference tutorial (rag.pdf), scaled here in characters via
RecursiveCharacterTextSplitter (LangChain's default unit).
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    """
    Splits a list of LangChain Document objects into smaller chunks,
    preserving metadata (source filename, page number) on each chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"[splitter] Split {len(documents)} pages into {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


if __name__ == "__main__":
    from loader import load_documents
    docs = load_documents()
    chunks = split_documents(docs)
    for c in chunks[:3]:
        print(c.metadata, "->", c.page_content[:120].replace("\n", " "), "...")
