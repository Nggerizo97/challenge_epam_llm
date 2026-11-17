from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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

    # LLM provider: "bedrock" (Bedrock chain, then Groq if a key is set) or "groq".
    # The AWS Emulator has no language model, so use "groq" when running against it.
    llm_provider: str = os.getenv("LLM_PROVIDER", "bedrock").strip().lower()

    # AWS Emulator / endpoint settings. Cognito and DynamoDB clients use this endpoint.
    # Credentials and the CA bundle come from AWS_SHARED_CREDENTIALS_FILE / AWS_CA_BUNDLE.
    aws_endpoint_url: str = os.getenv("AWS_ENDPOINT_URL", "")
    # Create missing tables and the Cognito user pool at startup (meant for the emulator,
    # which keeps everything in memory and starts empty on every run).
    aws_auto_create_resources: bool = _env_bool("AWS_AUTO_CREATE_RESOURCES", False)

    # Amazon Cognito (user sign-up / sign-in)
    cognito_user_pool_id: str = os.getenv("COGNITO_USER_POOL_ID", "")
    cognito_client_id: str = os.getenv("COGNITO_CLIENT_ID", "")
    cognito_user_pool_name: str = os.getenv("COGNITO_USER_POOL_NAME", "challenge-epam-llm-users")
    # Confirm new accounts server-side (AdminConfirmSignUp). The emulator does not send
    # verification e-mails or SMS, so this is required there to be able to sign in.
    cognito_auto_confirm: bool = _env_bool("COGNITO_AUTO_CONFIRM", False)

    # Sessions stored in DynamoDB, each with its own LLM token budget.
    sessions_table: str = os.getenv("SESSIONS_TABLE", "challenge-epam-llm-sessions")
    session_token_limit: int = max(int(os.getenv("SESSION_TOKEN_LIMIT", "10000")), 1)
    # Never longer than the Cognito access token lifetime.
    session_ttl_seconds: int = max(int(os.getenv("SESSION_TTL_SECONDS", "3600")), 60)

    # Per-IP limit on account creation (fixed window), to stop mass sign-ups.
    rate_limit_table: str = os.getenv("RATE_LIMIT_TABLE", "challenge-epam-llm-rate-limits")
    signup_limit_per_ip: int = max(int(os.getenv("SIGNUP_LIMIT_PER_IP", "3")), 1)
    signup_window_seconds: int = max(int(os.getenv("SIGNUP_WINDOW_SECONDS", "86400")), 60)
    # Number of reverse proxies in front of the app whose X-Forwarded-For can be trusted.
    # 0 = use the socket peer address only (X-Forwarded-For is ignored: it is spoofable).
    trusted_proxy_hops: int = max(int(os.getenv("TRUSTED_PROXY_HOPS", "0")), 0)

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
