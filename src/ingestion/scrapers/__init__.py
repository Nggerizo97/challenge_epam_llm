"""
Scraper Registry Package
========================
Exports active scraper classes for automated execution.
"""

from typing import List, Type
from src.ingestion.scrapers.base import BaseScraper
from src.ingestion.scrapers.datos_gov import DatosGovScraper

# Add any new scrapers to this list in the future
ACTIVE_SCRAPERS: List[Type[BaseScraper]] = [
    DatosGovScraper,
]
