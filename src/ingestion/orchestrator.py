"""
Ingestion Master Orchestrator
=============================
Orchestrates end-to-end data pipeline execution:
1. Discovers and runs all registered scrapers in ACTIVE_SCRAPERS.
2. Builds/updates the local ChromaDB vector store.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.ingestion.scrapers import ACTIVE_SCRAPERS
from src.ingestion.vectorstore import build_vector_store

# Ensure UTF-8 console output for Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("IngestionOrchestrator")


def run_pipeline(run_scrapers: bool = True) -> Dict[str, Any]:
    """
    Run full end-to-end ingestion pipeline.
    
    Args:
        run_scrapers: If True, executes all registered scrapers before vector store indexing.
    """
    logger.info("==================================================")
    logger.info("       STARTING INGESTION PIPELINE ORCHESTRATOR    ")
    logger.info("==================================================")

    all_scraped_items: List[Dict[str, Any]] = []

    if run_scrapers:
        logger.info(f"Registered scrapers to run: {len(ACTIVE_SCRAPERS)}")
        for scraper_cls in ACTIVE_SCRAPERS:
            scraper = scraper_cls()
            try:
                scraped_items = scraper.run()
                all_scraped_items.extend(scraped_items)
                logger.info(f"Scraper '{scraper.name}' finished. Fetched {len(scraped_items)} items.")
            except Exception as e:
                logger.error(f"Error running scraper '{scraper.name}': {e}")
    else:
        logger.info("Skipping scraping phase. Proceeding directly to VectorStore indexing.")

    # Build Vector Store
    logger.info("\n--- Building / Updating ChromaDB Vector Store ---")
    indexed_chunks = build_vector_store()

    summary = {
        "scrapers_executed": len(ACTIVE_SCRAPERS) if run_scrapers else 0,
        "scraped_files_count": len(all_scraped_items),
        "indexed_chunks": indexed_chunks,
        "status": "SUCCESS"
    }

    logger.info("==================================================")
    logger.info(f" PIPELINE COMPLETE: Indexed {indexed_chunks} chunks.")
    logger.info("==================================================")
    return summary


if __name__ == "__main__":
    run_pipeline(run_scrapers=True)
