import streamlit as st

from src.ingestion.bootstrap import build_sample_vectorstore, has_persisted_vectorstore
from src.ui.chat import answer_question

st.set_page_config(page_title="challenge_epam_LLM", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Preparing the document index...")
def ensure_vectorstore() -> None:
    if not has_persisted_vectorstore() and not build_sample_vectorstore():
        raise RuntimeError("The sample document index could not be created.")


ensure_vectorstore()

st.title("challenge_epam_LLM - Colombian & Global Socioeconomic Impact RAG Agent")
st.caption("Strict RAG mode: answers are grounded only in retrieved documents with citations.")

question = st.text_input("Ask a question about your indexed documents")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating grounded answer..."):
        result = answer_question(question)

    st.subheader("Answer")
    st.text(result["answer"])

    st.subheader("Citations")
    if result["citations"]:
        for idx, source in enumerate(result["citations"], start=1):
            st.text(f"Source {idx}: {source}")
    else:
        st.info("No supporting citation available. The assistant abstained.")
