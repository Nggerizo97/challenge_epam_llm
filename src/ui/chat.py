from __future__ import annotations

import logging
from typing import Dict, Any
from src.config import settings
from src.generation.pipeline import generate_answer
from src.retrieval.pipeline import retrieve_context

logger = logging.getLogger("ChatUIBackend")

ABSTAIN_MSG = "I do not know based on the provided documents."


def _empty_result(question: str, answer: str) -> Dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "citations": [],
        "retrieved_count": 0,
        "context_snippets": [],
    }


def answer_question(question: str) -> Dict[str, Any]:
    """
    Retrieve context documents and generate grounded answer with citations using AWS Bedrock.

    Applies input-length guardrails and fails closed: on any pipeline error the user
    gets a generic message, never a stack trace.
    """
    question = (question or "").strip()
    if not question:
        return _empty_result(question, ABSTAIN_MSG)

    if len(question) > settings.max_question_chars:
        logger.warning(
            "Rejected over-length question: %d chars (cap %d).",
            len(question), settings.max_question_chars,
        )
        return _empty_result(
            question[: settings.max_question_chars],
            f"Your question is too long. Please keep it under {settings.max_question_chars} characters.",
        )

    # Log a bounded, single-line preview only (avoid dumping untrusted input verbatim).
    logger.info("Processing question (%d chars): %.120r", len(question), question)
    try:
        context_docs = retrieve_context(question)
        answer, citations = generate_answer(question, context_docs)
    except Exception:
        logger.exception("Answer pipeline failed.")
        return _empty_result(
            question,
            "Sorry, something went wrong while processing your request. Please try again.",
        )

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
