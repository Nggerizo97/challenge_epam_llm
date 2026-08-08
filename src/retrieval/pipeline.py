from __future__ import annotations

from typing import List

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings


def retrieve_context(query: str) -> List[Document]:
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vector_store = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=embeddings,
    )
    return vector_store.similarity_search(query, k=settings.top_k)
