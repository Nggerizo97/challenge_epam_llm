"""
Project Ingestion Entrypoint
============================
Unified single command to run automated scrapers and build/update ChromaDB vector store.

Usage:
  python ingest.py             # Run all scrapers + build vector store
  python ingest.py --no-scrape # Skip scraping, re-index existing data/raw files
"""

import sys
from src.ingestion.orchestrator import run_pipeline

if __name__ == "__main__":
    should_scrape = "--no-scrape" not in sys.argv
    summary = run_pipeline(run_scrapers=should_scrape)
    print(f"\nResult: {summary}")
