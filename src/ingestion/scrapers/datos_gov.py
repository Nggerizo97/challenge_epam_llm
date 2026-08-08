"""
Colombia Open Data & Document Scraper Module
============================================
Dynamic multi-format scraper for discovering and downloading CSV, PDF, and Excel (.xlsx/.xls)
documents from Colombian open data portals (datos.gov.co, DANE, Banco de la República, ANM, MinAmbiente).
"""

import sys
import os
import re
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
from urllib.parse import urljoin, urlparse

import urllib3
import requests

from src.ingestion.scrapers.base import BaseScraper

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

logger = logging.getLogger("DatosGovScraper")

DOWNLOAD_DIR = Path("./data/raw")
MAX_CSV_SIZE_MB = 25.0
MAX_PDF_SIZE_MB = 35.0
MAX_XLSX_SIZE_MB = 35.0

TARGET_CATEGORIES = {
    "Comercio Exterior & Exportaciones": {
        "socrata_queries": ["Exportaciones", "Comercio Exterior", "Cafe", "Flores", "Aguacate"],
        "crawl_urls": [
            "https://www.dane.gov.co/index.php/estadisticas-por-tema/comercio-internacional/exportaciones",
            "https://www.dane.gov.co/index.php/estadisticas-por-tema/agropecuario/sistema-de-informacion-de-precios-sipsa"
        ]
    },
    "Energía & Minería": {
        "socrata_queries": ["Carbon", "Hidrocarburos", "Petroleo", "Esmeraldas", "Produccion Minera"],
        "crawl_urls": [
            "https://www.anm.gov.co/"
        ]
    },
    "Biodiversidad & Medio Ambiente": {
        "socrata_queries": ["Deforestacion", "Parques Nacionales", "Biodiversidad", "Bosques"],
        "crawl_urls": [
            "https://www.minambiente.gov.co/"
        ]
    },
    "Macroeconomía, Bancos & Remesas": {
        "socrata_queries": ["Tasas de interes", "Inflacion", "Remesas", "Credito"],
        "crawl_urls": [
            "https://www.banrep.gov.co/es/estadisticas/inflacion-total-y-meta"
        ]
    },
    "Turismo, Cultura & Migración": {
        "socrata_queries": ["Turismo", "Viajeros Internacionales", "Migracion", "PPT"],
        "crawl_urls": []
    }
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8"
}


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r'[^\w\s-]', '', name).strip().lower()
    return re.sub(r'[-\s]+', '_', cleaned)[:65]


class DatosGovScraper(BaseScraper):
    """Scraper implementation for Colombian Open Data portals and government entities."""

    @property
    def name(self) -> str:
        return "Colombia Open Data & Document Scraper (datos.gov.co / DANE / BanRep)"

    def _download_file(self, url: str, dest_filename: str, max_mb: float = 30.0) -> Optional[Path]:
        target_path = DOWNLOAD_DIR / dest_filename
        max_bytes = int(max_mb * 1024 * 1024)

        logger.info(f"Downloading [{dest_filename}] from: {url}")
        try:
            r = requests.get(url, headers=HEADERS, verify=False, timeout=30, stream=True)
            if r.status_code == 200:
                total_bytes = 0
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=32768):
                        if not chunk:
                            continue
                        f.write(chunk)
                        total_bytes += len(chunk)
                        if total_bytes >= max_bytes:
                            logger.info(f"File reached max size limit ({max_mb:.0f} MB). Truncating download safely.")
                            break

                file_kb = target_path.stat().st_size / 1024.0
                if file_kb < 1.5:
                    logger.warning(f"File size too small ({file_kb:.1f} KB). Removing corrupted file: {target_path}")
                    target_path.unlink(missing_ok=True)
                    return None

                logger.info(f"SUCCESS: Saved {dest_filename} ({file_kb:.1f} KB)")
                return target_path
            else:
                logger.warning(f"HTTP {r.status_code} for URL: {url}")
                return None
        except Exception as e:
            logger.error(f"Failed download for {url}: {e}")
            return None

    def _discover_socrata_datasets(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        catalog_url = "https://www.datos.gov.co/api/catalog/v1"
        params = {"q": query, "type": "dataset", "limit": limit, "sortBy": "relevance"}
        
        found = []
        try:
            r = requests.get(catalog_url, params=params, headers=HEADERS, verify=False, timeout=15)
            if r.status_code == 200:
                results = r.json().get("results", [])
                for item in results:
                    res = item.get("resource", {})
                    ds_id = res.get("id")
                    title = res.get("name")
                    if ds_id and title:
                        found.append({
                            "id": ds_id,
                            "title": title,
                            "download_url": f"https://www.datos.gov.co/api/views/{ds_id}/rows.csv?accessType=DOWNLOAD",
                            "format": "csv"
                        })
        except Exception as e:
            logger.error(f"Socrata discovery error for '{query}': {e}")
        
        return found

    def _crawl_page_for_documents(self, page_url: str) -> List[Dict[str, Any]]:
        logger.info(f"Crawling portal page for live documents: {page_url}")
        documents = []

        try:
            r = requests.get(page_url, headers=HEADERS, verify=False, timeout=20)
            if r.status_code == 200:
                html = r.text
                matches = re.findall(r'href=["\']([^"\']+\.(?:pdf|xlsx?))["\']', html, re.IGNORECASE)
                
                for rel_link in set(matches):
                    abs_url = urljoin(page_url, rel_link)
                    parsed = urlparse(abs_url)
                    filename = Path(parsed.path).name
                    
                    if filename and len(filename) > 5 and not any(x in filename.lower() for x in ['logo', 'icon', 'manual_identidad', 'template']):
                        ext = filename.split('.')[-1].lower()
                        documents.append({
                            "id": filename.split('.')[0],
                            "title": filename,
                            "download_url": abs_url,
                            "format": ext
                        })
            else:
                logger.warning(f"Crawling status {r.status_code} for page: {page_url}")
        except Exception as e:
            logger.error(f"Error crawling page {page_url}: {e}")

        return documents

    def run(self) -> List[Dict[str, Any]]:
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"=== Running Scraper: {self.name} ===")

        downloaded_urls: Set[str] = set()
        manifest: List[Dict[str, Any]] = []

        for category, config in TARGET_CATEGORIES.items():
            logger.info(f"\nTarget Category: [{category}]")

            # 1. Socrata Datasets
            for query in config["socrata_queries"]:
                datasets = self._discover_socrata_datasets(query, limit=2)
                for ds in datasets:
                    d_url = ds["download_url"]
                    if d_url in downloaded_urls:
                        continue

                    fname = f"csv_{ds['id']}_{sanitize_name(ds['title'])}.csv"
                    saved = self._download_file(d_url, fname, max_mb=MAX_CSV_SIZE_MB)
                    if saved:
                        downloaded_urls.add(d_url)
                        manifest.append({
                            "title": ds["title"],
                            "category": category,
                            "format": "csv",
                            "filename": fname,
                            "source_url": d_url,
                            "file_size_kb": saved.stat().st_size / 1024.0,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        })

            # 2. Dynamic Web Crawling for PDF/XLSX
            for portal_url in config["crawl_urls"]:
                docs = self._crawl_page_for_documents(portal_url)
                for doc in docs[:3]:
                    d_url = doc["download_url"]
                    if d_url in downloaded_urls:
                        continue

                    fmt = doc["format"]
                    max_size = MAX_PDF_SIZE_MB if fmt == "pdf" else MAX_XLSX_SIZE_MB
                    fname = f"{fmt}_{sanitize_name(doc['title'])}.{fmt}"
                    
                    saved = self._download_file(d_url, fname, max_mb=max_size)
                    if saved:
                        downloaded_urls.add(d_url)
                        manifest.append({
                            "title": doc["title"],
                            "category": category,
                            "format": fmt,
                            "filename": fname,
                            "source_url": d_url,
                            "file_size_kb": saved.stat().st_size / 1024.0,
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ")
                        })

        return manifest
