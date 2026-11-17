"""Create the DynamoDB tables and Cognito user pool in the AWS Emulator.

The emulator keeps resources in memory, so run this after every emulator start
(the app also does it by itself when AWS_AUTO_CREATE_RESOURCES=true):

    source scripts/emulator_env.sh
    python -m scripts.setup_emulator
"""
from __future__ import annotations

import os

os.environ.setdefault("AWS_AUTO_CREATE_RESOURCES", "true")

from src.aws.resources import ensure_resources  # noqa: E402
from src.config import settings  # noqa: E402


def main() -> None:
    if not settings.aws_endpoint_url:
        raise SystemExit("AWS_ENDPOINT_URL is not set. Run: source scripts/emulator_env.sh")
    ids = ensure_resources()
    print(f"Endpoint:            {settings.aws_endpoint_url}")
    print(f"Sessions table:      {settings.sessions_table}")
    print(f"Rate-limit table:    {settings.rate_limit_table}")
    if settings.dynamodb_quota_table:
        print(f"Quota table:         {settings.dynamodb_quota_table}")
    print(f"COGNITO_USER_POOL_ID={ids.user_pool_id}")
    print(f"COGNITO_CLIENT_ID={ids.client_id}")


if __name__ == "__main__":
    main()
