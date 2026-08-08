"""
Base Scraper Interface
======================
Defines abstract BaseScraper class for registering modular scrapers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseScraper(ABC):
    """Abstract base class for all data and document scrapers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the scraper."""
        pass

    @abstractmethod
    def run(self) -> List[Dict[str, Any]]:
        """
        Execute scraping & downloading pipeline.
        Returns a list of metadata dictionaries for all downloaded files.
        """
        pass
