"""Idempotent creation of the DynamoDB tables and Cognito user pool the app needs.

The AWS Emulator keeps every resource in memory, so they disappear each time it stops.
With ``AWS_AUTO_CREATE_RESOURCES=true`` the app recreates them at startup; otherwise it
only looks them up, and fails with a clear error if they are missing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from botocore.exceptions import ClientError

from src.aws.clients import aws_client
from src.config import settings

logger = logging.getLogger("AwsResources")

APP_CLIENT_NAME = "streamlit-app"


class ResourceConfigError(RuntimeError):
    """Raised when a required AWS resource is missing and auto-creation is off."""


@dataclass(frozen=True)
class CognitoIds:
    user_pool_id: str
    client_id: str


def _ensure_table(name: str, hash_key: str) -> None:
    dynamodb = aws_client("dynamodb")
    try:
        dynamodb.describe_table(TableName=name)
        return
    except ClientError as error:
        if error.response["Error"].get("Code") != "ResourceNotFoundException":
            raise
    if not settings.aws_auto_create_resources:
        raise ResourceConfigError(
            f"DynamoDB table '{name}' does not exist. Create it or set AWS_AUTO_CREATE_RESOURCES=true."
        )

    logger.info("Creating DynamoDB table %s", name)
    try:
        dynamodb.create_table(
            TableName=name,
            BillingMode="PAY_PER_REQUEST",
            AttributeDefinitions=[{"AttributeName": hash_key, "AttributeType": "S"}],
            KeySchema=[{"AttributeName": hash_key, "KeyType": "HASH"}],
        )
    except ClientError as error:
        # Another process created it between describe and create.
        if error.response["Error"].get("Code") != "ResourceInUseException":
            raise
    dynamodb.get_waiter("table_exists").wait(TableName=name)
    dynamodb.update_time_to_live(
        TableName=name,
        TimeToLiveSpecification={"Enabled": True, "AttributeName": "expires_at"},
    )


def ensure_tables() -> None:
    _ensure_table(settings.sessions_table, "session_id")
    _ensure_table(settings.rate_limit_table, "limit_key")
    if settings.dynamodb_quota_table:
        _ensure_table(settings.dynamodb_quota_table, "quota_key")


def _find_user_pool(name: str) -> str | None:
    paginator = aws_client("cognito-idp").get_paginator("list_user_pools")
    for page in paginator.paginate(MaxResults=60):
        for pool in page.get("UserPools", []):
            if pool["Name"] == name:
                return pool["Id"]
    return None


def _find_app_client(user_pool_id: str, name: str) -> str | None:
    paginator = aws_client("cognito-idp").get_paginator("list_user_pool_clients")
    for page in paginator.paginate(UserPoolId=user_pool_id, MaxResults=60):
        for client in page.get("UserPoolClients", []):
            if client["ClientName"] == name:
                return client["ClientId"]
    return None


def _create_user_pool(name: str) -> str:
    logger.info("Creating Cognito user pool %s", name)
    pool = aws_client("cognito-idp").create_user_pool(
        PoolName=name,
        # Users sign up with an e-mail address or a phone number as their username.
        UsernameAttributes=["email", "phone_number"],
        AutoVerifiedAttributes=["email"],
        Policies={
            "PasswordPolicy": {
                "MinimumLength": 10,
                "RequireUppercase": True,
                "RequireLowercase": True,
                "RequireNumbers": True,
                "RequireSymbols": False,
            }
        },
        AccountRecoverySetting={
            "RecoveryMechanisms": [{"Priority": 1, "Name": "verified_email"}]
        },
    )
    return pool["UserPool"]["Id"]


def _create_app_client(user_pool_id: str) -> str:
    logger.info("Creating Cognito app client %s", APP_CLIENT_NAME)
    client = aws_client("cognito-idp").create_user_pool_client(
        UserPoolId=user_pool_id,
        ClientName=APP_CLIENT_NAME,
        GenerateSecret=False,
        ExplicitAuthFlows=["ALLOW_USER_PASSWORD_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"],
        PreventUserExistenceErrors="ENABLED",
        AccessTokenValidity=1,
        IdTokenValidity=1,
        TokenValidityUnits={"AccessToken": "hours", "IdToken": "hours"},
    )
    return client["UserPoolClient"]["ClientId"]


def ensure_cognito() -> CognitoIds:
    if settings.cognito_user_pool_id and settings.cognito_client_id:
        return CognitoIds(settings.cognito_user_pool_id, settings.cognito_client_id)

    pool_id = settings.cognito_user_pool_id or _find_user_pool(settings.cognito_user_pool_name)
    if not pool_id:
        if not settings.aws_auto_create_resources:
            raise ResourceConfigError(
                f"Cognito user pool '{settings.cognito_user_pool_name}' was not found. Set "
                "COGNITO_USER_POOL_ID and COGNITO_CLIENT_ID, or AWS_AUTO_CREATE_RESOURCES=true."
            )
        pool_id = _create_user_pool(settings.cognito_user_pool_name)

    client_id = settings.cognito_client_id or _find_app_client(pool_id, APP_CLIENT_NAME)
    if not client_id:
        if not settings.aws_auto_create_resources:
            raise ResourceConfigError(
                f"Cognito app client '{APP_CLIENT_NAME}' was not found in pool {pool_id}."
            )
        client_id = _create_app_client(pool_id)

    return CognitoIds(pool_id, client_id)


def ensure_resources() -> CognitoIds:
    ensure_tables()
    return ensure_cognito()
