import os
import streamlit as st

from chatbot import RagChatbot

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 RAG Chatbot")
st.caption("Ask questions about the documents in `data/` (FAISS + all-MiniLM-L6-v2 + Groq/OpenAI)")

# --- API key handling ---
# Auto-detect from environment first (either provider). 
groq_key = os.environ.get("GROQ_API_KEY")
openai_key = os.environ.get("OPENAI_API_KEY")

if not groq_key and not openai_key:
    st.sidebar.markdown("**No API key found in environment.**")
    provider_choice = st.sidebar.radio("Provider", ["Groq (free)", "OpenAI"])
    entered_key = st.sidebar.text_input(
        f"{provider_choice.split(' ')[0]} API Key", type="password"
    )
    if provider_choice.startswith("Groq"):
        groq_key = entered_key
    else:
        openai_key = entered_key

if not groq_key and not openai_key:
    st.info(
        "Enter an API key in the sidebar to get started. "
        "Groq offers a free tier: https://console.groq.com"
    )
    st.stop()

# --- Load chatbot (cached across reruns) ---
@st.cache_resource(show_spinner="Loading index and embedding model...")
def load_bot(g_key, o_key):
    return RagChatbot(groq_api_key=g_key, openai_api_key=o_key)

try:
    bot = load_bot(groq_key, openai_key)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except EnvironmentError as e:
    st.error(str(e))
    st.stop()

st.sidebar.success(f"Using {bot.provider} ({bot.model})")

# --- Chat history state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + ", ".join(msg["sources"]))

# --- Chat input ---
if prompt := st.chat_input("Ask something about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = bot.ask(prompt)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })
