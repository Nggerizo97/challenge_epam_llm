"""boto3 client factory shared by Cognito, DynamoDB and the quota code.

When AWS_ENDPOINT_URL is set (for example ``https://127.0.0.1:9297`` for the AWS
Emulator), clients are sent there. Credentials are resolved by the normal boto3 chain,
so with the emulator they come from ``AWS_SHARED_CREDENTIALS_FILE`` and the emulator
certificate from ``AWS_CA_BUNDLE``.
"""
from __future__ import annotations

from functools import lru_cache

from src.config import settings


@lru_cache(maxsize=None)
def aws_client(service_name: str):
    import boto3

    return boto3.client(
        service_name,
        region_name=settings.aws_region,
        endpoint_url=settings.aws_endpoint_url or None,
        aws_access_key_id=settings.aws_access_key_id or None,
        aws_secret_access_key=settings.aws_secret_access_key or None,
    )
