from __future__ import annotations

from pathlib import Path

from src.config import settings
from src.ingestion.orchestrator import run_pipeline


def has_persisted_vectorstore(chroma_dir: str | None = None) -> bool:
    """Return whether Chroma's persistent database exists."""
    return (Path(chroma_dir or settings.chroma_dir) / "chroma.sqlite3").is_file()


def build_sample_vectorstore() -> int:
    """Index only the small, versioned sample documents for cloud deployments."""
    summary = run_pipeline(run_scrapers=False, target_dir="./data/raw_sample")
    return summary["indexed_chunks"]