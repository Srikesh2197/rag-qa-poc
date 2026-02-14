import os
import streamlit as st


from ingest import ingest
from rag import ask
from config import OPENAI_API_KEY


st.set_page_config(page_title="RAG Q&A POC", layout="wide")
st.title("RAG Q&A - POC")

with st.sidebar:
    st.header("Index")

    # model_provider = st.selectbox("Model", ["OpenAI", "Ollama"], index=0)
    model_options = ["OpenAI", "Ollama"]
    default_index = 0 if OPENAI_API_KEY else 1
    model_provider = st.selectbox("Model", model_options, index=default_index)

    if model_provider == "OpenAI" and not OPENAI_API_KEY:
        st.warning("OpenAI API key not found. Falling back to Ollama.")

    all_uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if all_uploaded_files:
        os.makedirs("data/raw", exist_ok=True)
        saved = 0
        for file in all_uploaded_files:
            save_path = os.path.join("data/raw", file.name)
            with open(save_path, "wb") as f:
                f.write(file.getbuffer())
            saved += 1
        st.success(f"Saved {saved} file(s) to data/raw. Now click 'Rebuild Index'.")

    if st.button("Rebuild Index", type="primary"):
        with st.spinner("Rebuilding index from data/raw ..."):
            n_chunks = ingest(reset_db=True)
        st.success(f"Indexed {n_chunks} chunks into Chroma.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    pairs = []
    user_buf = None
    for m in st.session_state.messages:
        if m["role"] == "user":
            user_buf = m["content"]
        elif m["role"] == "assistant" and user_buf is not None:
            pairs.append((user_buf, m["content"]))
            user_buf = None

    pairs = pairs[-5:]

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Retrieving with history + generating answer..."):
        result = ask(question, chat_history=pairs, model_provider=model_provider)

    answer = result.get("answer", "")
    sources = result.get("source_documents", [])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

        st.subheader("Sources (Top 3)")
        if not sources:
            st.info("No sources returned.")
        else:
            for i, doc in enumerate(sources[:3], start=1):
                meta = doc.metadata or {}
                source = meta.get("source", "unknown")
                page = meta.get("page", None)

                label = f"{i}. {source}"
                if page is not None:
                    label += f" (page {page})"

                with st.expander(label):
                    st.write(doc.page_content)