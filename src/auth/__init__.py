"""Cognito authentication, DynamoDB sessions with token budgets and per-IP limits."""
from __future__ import annotations

from src.config import settings


def build_auth_service():
    """Wire the auth service to the configured endpoint (e.g. the AWS Emulator).

    Creates the tables and the user pool first when AWS_AUTO_CREATE_RESOURCES is on.
    """
    from src.auth.cognito import CognitoAuth
    from src.auth.rate_limit import DynamoRateLimiter
    from src.auth.service import AuthPolicy, AuthService
    from src.auth.sessions import DynamoSessionStore, TokenLedger, TokenLimits
    from src.aws.clients import aws_client
    from src.aws.resources import ensure_resources

    ids = ensure_resources()
    dynamodb = aws_client("dynamodb")
    return AuthService(
        cognito=CognitoAuth(aws_client("cognito-idp"), ids.user_pool_id, ids.client_id),
        sessions=DynamoSessionStore(dynamodb, settings.sessions_table),
        rate_limiter=DynamoRateLimiter(dynamodb, settings.rate_limit_table),
        ledger=TokenLedger(
            dynamodb,
            settings.sessions_table,
            settings.rate_limit_table,
            TokenLimits(user_daily=settings.user_daily_token_limit, ip_daily=settings.ip_daily_token_limit),
        ),
        policy=AuthPolicy(
            session_token_limit=settings.session_token_limit,
            session_ttl_seconds=settings.session_ttl_seconds,
            signup_limit_per_ip=settings.signup_limit_per_ip,
            signup_window_seconds=settings.signup_window_seconds,
            auto_confirm=settings.cognito_auto_confirm,
            login_max_failures_per_ip=settings.login_max_failures_per_ip,
            login_max_failures_per_account=settings.login_max_failures_per_account,
            login_failure_window_seconds=settings.login_failure_window_seconds,
        ),
    )
