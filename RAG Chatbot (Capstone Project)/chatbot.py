"""
chatbot.py
Retrieval-augmented answer generation:
  - Loads the persisted FAISS index
  - Retrieves top-k relevant chunks for a user question
  - Calls a chat model (Groq or OpenAI) with the retrieved context
  - Returns an answer plus the source citations (filename + page)

Provider selection is automatic and key-driven, not hardcoded:
  - If GROQ_API_KEY is set, use Groq (free tier, OpenAI-compatible API).
  - Else if OPENAI_API_KEY is set, use OpenAI.
  - Else raise a clear error.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.vectorstores import FAISS

from utils.embedding import get_embeddings

# Load variables from a .env file in the project root (if present) into
# the environment, so GROQ_API_KEY / OPENAI_API_KEY set there are picked up.
load_dotenv()

VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "vector_db")
FAISS_INDEX_PATH = os.path.join(VECTOR_DB_DIR, "faiss_index")

TOP_K = 4

# Default chat models per provider (override with CHAT_MODEL env var if needed)
GROQ_MODEL = "llama-3.1-8b-instant"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the "
    "provided context excerpts. If the answer isn't in the context, say "
    "you don't have enough information rather than guessing. "
    "Always ground your answer in the given excerpts."
)


def _resolve_provider(groq_key: str = None, openai_key: str = None):
    
    groq_key = groq_key or os.environ.get("GROQ_API_KEY")
    openai_key = openai_key or os.environ.get("OPENAI_API_KEY")

    if groq_key:
        client = OpenAI(api_key=groq_key, base_url="https://api.groq.com/openai/v1")
        model = os.environ.get("CHAT_MODEL", GROQ_MODEL)
        return client, model, "Groq"

    if openai_key:
        client = OpenAI(api_key=openai_key)
        model = os.environ.get("CHAT_MODEL", OPENAI_MODEL)
        return client, model, "OpenAI"

    raise EnvironmentError(
        "No API key found. Set GROQ_API_KEY (free — https://console.groq.com) "
        "or OPENAI_API_KEY in your .env file or environment."
    )


class RagChatbot:
    def __init__(self, groq_api_key: str = None, openai_api_key: str = None):
        self.client, self.model, self.provider = _resolve_provider(
            groq_key=groq_api_key, openai_key=openai_api_key
        )
        self.embeddings = get_embeddings()
        self.vector_store = self._load_index()

    def _load_index(self):
        if not os.path.isdir(FAISS_INDEX_PATH):
            raise FileNotFoundError(
                f"No FAISS index found at {FAISS_INDEX_PATH}. "
                "Run `python build_index.py` first."
            )
        return FAISS.load_local(
            FAISS_INDEX_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def retrieve(self, query: str, k: int = TOP_K):
        return self.vector_store.similarity_search(query, k=k)

    def _build_context(self, chunks):
        blocks = []
        for i, c in enumerate(chunks, start=1):
            src = c.metadata.get("source", "unknown")
            page = c.metadata.get("page", "?")
            blocks.append(f"[{i}] (source: {src}, page: {page})\n{c.page_content}")
        return "\n\n---\n\n".join(blocks)

    def _citations(self, chunks):
        seen = []
        for c in chunks:
            src = c.metadata.get("source", "unknown")
            page = c.metadata.get("page", "?")
            label = f"{src} (page {page})"
            if label not in seen:
                seen.append(label)
        return seen

    def ask(self, question: str, k: int = TOP_K):
        chunks = self.retrieve(question, k=k)
        context = self._build_context(chunks)

        user_prompt = (
            f"Context excerpts:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using only the context above, and mention "
            "which excerpt number(s) you used."
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": self._citations(chunks),
        }


if __name__ == "__main__":
    bot = RagChatbot()
    print(f"Using provider: {bot.provider} (model: {bot.model})")
    while True:
        q = input("\nAsk a question (or 'quit'): ")
        if q.lower() in {"quit", "exit"}:
            break
        result = bot.ask(q)
        print("\nAnswer:", result["answer"])
        print("Sources:", ", ".join(result["sources"]))
