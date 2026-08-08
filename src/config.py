from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # AWS Bedrock Credentials & Settings
    aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    aws_region: str = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_REGION", "us-east-1"))
    bedrock_model_id: str = os.getenv("AWS_BEDROCK_MODEL_ID", os.getenv("BEDROCK_MODEL_ID", "meta.llama3-70b-instruct-v1:0"))
    bedrock_fallback_model_id: str = os.getenv("BEDROCK_FALLBACK_MODEL_ID", "amazon.titan-text-express-v1")
    
    # Cost & Generation Guardrail Parameters
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    # Reject over-length questions before embedding/LLM to cap cost and abuse.
    max_question_chars: int = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
    # Per-process provider invocation budget. It can be reduced, never raised above 20.
    max_api_requests: int = min(max(int(os.getenv("MAX_API_REQUESTS", "5")), 1), 20)
    # When configured, DynamoDB enforces the same quota across restarts and replicas.
    dynamodb_quota_table: str = os.getenv("DYNAMODB_QUOTA_TABLE", "")

    # Groq Fallback Settings
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    # Embedding & Vector Database Settings
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    data_dir: str = os.getenv("DATA_DIR", "./data/raw")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")


settings = Settings()
