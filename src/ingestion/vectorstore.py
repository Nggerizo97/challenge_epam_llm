"""
Vector Store Builder Module
===========================
Loads plain text (.txt), PDF (.pdf), and CSV (.csv) documents from data/raw/,
chunks content using RecursiveCharacterTextSplitter, embeds vectors, and persists to ChromaDB.
"""

import sys
import logging
from pathlib import Path

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger("VectorStoreBuilder")


def build_vector_store(target_dir: str = None) -> int:
    """Build and persist local ChromaDB vector index from raw data files."""
    selected_dir = target_dir or settings.data_dir
    data_path = Path(selected_dir)

    if not data_path.exists() or not list(data_path.glob("**/*")):
        fallback_sample = Path("./data/raw_sample")
        if fallback_sample.exists():
            logger.info(f"Directory {data_path} empty or missing. Falling back to sample directory: {fallback_sample.resolve()}")
            data_path = fallback_sample

    logger.info(f"Scanning raw documents in: {data_path.resolve()}")
    docs = []


    # 1. Plain Text Files (.txt)
    txt_loader = DirectoryLoader(
        str(data_path),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=False,
    )
    txt_docs = txt_loader.load()
    docs.extend(txt_docs)
    logger.info(f"Loaded {len(txt_docs)} text document chunks.")


    # 2. PDF Documents (.pdf)
    try:
        from langchain_community.document_loaders import PyPDFLoader
        pdf_files = list(data_path.glob("**/*.pdf"))
        loaded_pdfs = 0
        for pdf_file in pdf_files:
            try:
                loader = PyPDFLoader(str(pdf_file))
                pdf_chunks = loader.load()
                docs.extend(pdf_chunks)
                loaded_pdfs += len(pdf_chunks)
            except Exception as e:
                logger.warning(f"Failed to load PDF {pdf_file.name}: {e}")
        logger.info(f"Loaded {loaded_pdfs} pages across {len(pdf_files)} PDF files.")
    except ImportError:
        logger.warning("PyPDFLoader not available. Skipping PDF files.")

    # 3. CSV Datasets (.csv)
    try:
        from langchain_community.document_loaders import CSVLoader
        csv_files = list(data_path.glob("**/*.csv"))
        loaded_csvs = 0
        for csv_file in csv_files:
            # Skip massive CSV files (> 50 MB) to prevent OOM
            if csv_file.stat().st_size > 50 * 1024 * 1024:
                continue
            try:
                loader = CSVLoader(str(csv_file), encoding="utf-8")
                csv_chunks = loader.load()
                docs.extend(csv_chunks)
                loaded_csvs += len(csv_chunks)
            except Exception as e:
                logger.warning(f"Failed to load CSV {csv_file.name}: {e}")
        logger.info(f"Loaded {loaded_csvs} records across CSV files.")
    except ImportError:
        logger.warning("CSVLoader not available. Skipping CSV files.")

    if not docs:
        logger.warning("No valid documents loaded for indexing.")
        return 0

    logger.info(f"Splitting {len(docs)} total document pages/chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    logger.info(f"Embedding {len(chunks)} text chunks into ChromaDB at: {settings.chroma_dir}")
    embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.chroma_dir,
    )
    
    logger.info("ChromaDB VectorStore indexing complete!")
    return len(chunks)
