from __future__ import annotations

import logging
from typing import Dict, Any
from src.generation.pipeline import generate_answer
from src.retrieval.pipeline import retrieve_context

logger = logging.getLogger("ChatUIBackend")


def answer_question(question: str) -> Dict[str, Any]:
    """
    Retrieve context documents and generate grounded answer with citations using AWS Bedrock.
    """
    logger.info(f"Processing question: '{question}'")
    context_docs = retrieve_context(question)
    answer, citations = generate_answer(question, context_docs)
    
    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "retrieved_count": len(context_docs),
        "context_snippets": [
            {
                "source": doc.metadata.get("source") or doc.metadata.get("file_path") or f"Doc {i+1}",
                "content": doc.page_content[:300] + "..."
            }
            for i, doc in enumerate(context_docs)
        ]
    }
