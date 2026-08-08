import streamlit as st

from src.ui.chat import answer_question

st.set_page_config(page_title="challenge_epam_LLM", page_icon="📚", layout="wide")
st.title("challenge_epam_LLM - Colombian & Global Socioeconomic Impact RAG Agent")
st.caption("Strict RAG mode: answers are grounded only in retrieved documents with citations.")

question = st.text_input("Ask a question about your indexed documents")

if st.button("Ask") and question.strip():
    with st.spinner("Retrieving context and generating grounded answer..."):
        result = answer_question(question)

    st.subheader("Answer")
    st.write(result["answer"])

    st.subheader("Citations")
    if result["citations"]:
        for idx, source in enumerate(result["citations"], start=1):
            st.markdown(f"- [Source {idx}] {source}")
    else:
        st.info("No supporting citation available. The assistant abstained.")
