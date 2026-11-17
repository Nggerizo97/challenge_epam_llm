"""Per-client-IP limits backed by DynamoDB (atomic fixed-window counters).

Used to cap how many accounts one IP can create, so a single person cannot mint many
e-mail / phone accounts to multiply their LLM token budget.
"""
from __future__ import annotations

import ipaddress
import time
from dataclasses import dataclass

from botocore.exceptions import ClientError


class RateLimitExceeded(Exception):
    def __init__(self, retry_after_seconds: int):
        super().__init__(f"Rate limit exceeded; retry in {retry_after_seconds} s.")
        self.retry_after_seconds = retry_after_seconds


def ip_bucket(ip: str | None) -> str:
    """Normalise a client IP into the unit that is rate limited.

    IPv6 users usually control a whole /64, so it is limited as one block. Values that
    are not IP addresses (or a missing IP) share a single "unknown" bucket, which fails
    closed instead of letting them bypass the limit.
    """
    if not ip:
        return "unknown"
    try:
        address = ipaddress.ip_address(ip.strip())
    except ValueError:
        return "unknown"
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped:
            return str(address.ipv4_mapped)
        return str(ipaddress.IPv6Network(f"{address}/64", strict=False))
    return str(address)


@dataclass(frozen=True)
class RateLimitHit:
    key: str


class DynamoRateLimiter:
    def __init__(self, dynamodb_client, table_name: str, clock=time.time):
        self._db = dynamodb_client
        self._table = table_name
        self._clock = clock

    def hit(self, scope: str, subject: str, limit: int, window_seconds: int) -> RateLimitHit:
        """Count one event for ``subject``; raise RateLimitExceeded if over ``limit``."""
        now = int(self._clock())
        window_start = now - now % window_seconds
        key = f"{scope}#{subject}#{window_start}"
        try:
            self._db.update_item(
                TableName=self._table,
                Key={"limit_key": {"S": key}},
                UpdateExpression="ADD hits :one SET expires_at = if_not_exists(expires_at, :expires_at)",
                ConditionExpression="attribute_not_exists(hits) OR hits < :limit",
                ExpressionAttributeValues={
                    ":one": {"N": "1"},
                    ":limit": {"N": str(limit)},
                    ":expires_at": {"N": str(window_start + window_seconds * 2)},
                },
            )
        except ClientError as error:
            if error.response["Error"].get("Code") == "ConditionalCheckFailedException":
                raise RateLimitExceeded(window_start + window_seconds - now) from error
            raise
        return RateLimitHit(key)

    def release(self, hit: RateLimitHit) -> None:
        """Give back one event, e.g. when the sign-up was rejected as invalid input."""
        self._db.update_item(
            TableName=self._table,
            Key={"limit_key": {"S": hit.key}},
            UpdateExpression="ADD hits :minus_one",
            ConditionExpression="attribute_exists(hits) AND hits > :zero",
            ExpressionAttributeValues={":minus_one": {"N": "-1"}, ":zero": {"N": "0"}},
        )
