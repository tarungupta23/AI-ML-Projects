# RAG Chatbot

A retrieval-augmented chatbot built with LangChain, FAISS, `all-MiniLM-L6-v2`
embeddings, and a Streamlit chat UI. The chat model is **provider-agnostic**:
it works with either a free Groq key or an OpenAI key — see below.

## Choosing a chat provider (free vs. paid)

`chatbot.py` auto-detects which key is available and picks the provider
accordingly — no code changes needed either way:

| Env var          | Provider | Cost                         | Model used              |
|-------------------|----------|-------------------------------|--------------------------|
| `GROQ_API_KEY`    | Groq     | Free tier (no card required)  | `llama-3.1-8b-instant`  |
| `OPENAI_API_KEY`  | OpenAI   | Paid                          | `gpt-4o-mini`            |

If both are set, **Groq is preferred**. Get a free Groq key at
https://console.groq.com — no credit card needed.

This means you can develop and test entirely on the free Groq tier, and
anyone else (e.g. a reviewer) can drop their own `OPENAI_API_KEY` into
`.env` instead and it'll just work — nobody needs to touch the code or be
told which provider is in use.

## Current knowledge base

- `data/Culture_of_India.pdf` ✅ added
- `data/History_of_India.pdf` ✅ added

Knowledge base is complete (2/2 documents). Drop any additional PDFs
straight into `data/` and re-run `build_index.py` to extend it further.

## Setup

```bash
cd RAG_Chatbot
pip install -r requirements.txt

# Pick ONE:
export GROQ_API_KEY="gsk_..."      # free — https://console.groq.com
# export OPENAI_API_KEY="sk-..."   # paid alternative
```

## 1. Build the index

Run this once, and again any time you add/change files in `data/`:

```bash
python build_index.py
```

This will:
1. Load every PDF in `data/` except `rag.pdf`
2. Split them into ~600-character chunks (100-char overlap)
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2`
4. Save a FAISS index to `vector_db/faiss_index` and chunk metadata to
   `vector_db/chunks.pkl`

## 2. Run the chatbot

CLI:
```bash
python chatbot.py
```

Streamlit UI:
```bash
streamlit run streamlit_app.py
```

## Project structure

```
RAG_Chatbot/
├── data/                  # source PDFs (knowledge base) — rag.pdf excluded
├── vector_db/             # generated: FAISS index + chunks.pkl
├── utils/
│   ├── loader.py          # loads PDFs, excludes rag.pdf
│   ├── splitter.py        # RecursiveCharacterTextSplitter (600/100)
│   └── embedding.py       # all-MiniLM-L6-v2 embeddings
├── build_index.py         # builds and saves the FAISS index
├── chatbot.py             # retrieval + OpenAI chat completion + citations
├── streamlit_app.py       # chat UI
├── requirements.txt
└── README.md
```

## Notes

- Answers are grounded strictly in retrieved chunks; the model is instructed
  to say when it doesn't have enough information.
- Every answer includes source citations (filename + page number).
- To add another excluded reference file, add its filename to
  `EXCLUDED_FILES` in `utils/loader.py`.
