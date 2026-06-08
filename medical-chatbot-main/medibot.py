from pathlib import Path

import streamlit as st

from rag_core import (
    DB_FAISS_PATH,
    build_qa_chain,
    format_source_documents,
    load_vectorstore,
    validate_runtime_config,
)


st.set_page_config(page_title="AI Medical Chatbot with RAG", layout="centered")
st.title("AI Medical Chatbot with RAG")
st.caption("Educational answers grounded in the bundled medical knowledge base.")


@st.cache_resource(show_spinner="Loading FAISS vector store...")
def cached_vectorstore(path: str):
    return load_vectorstore(Path(path))


with st.sidebar:
    st.subheader("Retrieval")
    top_k = st.slider("Sources", min_value=2, max_value=6, value=4)
    st.write(f"Vector store: `{DB_FAISS_PATH}`")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

config_issues = validate_runtime_config()
if config_issues:
    for issue in config_issues:
        st.error(issue)
    st.info("Configure secrets and build the FAISS index before deploying.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Ask a medical education question. I will answer only from the indexed source material.",
        }
    ]

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask a question about the indexed medical reference")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        vectorstore = cached_vectorstore(str(DB_FAISS_PATH))
        qa_chain = build_qa_chain(vectorstore, k=top_k)
        response = qa_chain.invoke({"query": prompt})

        answer = response.get("result", "I do not know based on the available sources.")
        sources = format_source_documents(response.get("source_documents", []))
        assistant_message = f"{answer}\n\n**Sources**\n{sources}"

    except Exception as exc:
        assistant_message = (
            "The chatbot could not complete this request. "
            "Check the deployment logs for model, key, or vector-store errors."
        )
        st.exception(exc)

    st.chat_message("assistant").markdown(assistant_message)
    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
