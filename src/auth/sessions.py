"""User sessions stored in DynamoDB, each with an LLM token budget.

A session is created after a successful Cognito sign-in. Its record holds the Cognito
user id, the client IP, an expiry and a ``tokens_used`` counter. Before every LLM call
the worst case (prompt estimate + max output tokens) is reserved with a conditional
update, so concurrent requests can never push a session over its limit. After the call
the reservation is corrected to the tokens the provider actually reported.

Cognito tokens are never written to DynamoDB; only the opaque session id is stored.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

from botocore.exceptions import ClientError

logger = logging.getLogger("SessionStore")


class SessionInvalid(Exception):
    """The session does not exist, was revoked or has expired."""


class SessionTokenLimitExceeded(Exception):
    """The session has no token budget left for this request."""


@dataclass(frozen=True)
class Session:
    session_id: str
    user_sub: str
    username: str
    client_ip: str
    created_at: int
    expires_at: int
    tokens_used: int
    token_limit: int
    status: str

    @property
    def tokens_remaining(self) -> int:
        return max(self.token_limit - self.tokens_used, 0)


def _to_session(item: dict) -> Session:
    return Session(
        session_id=item["session_id"]["S"],
        user_sub=item["user_sub"]["S"],
        username=item["username"]["S"],
        client_ip=item["client_ip"]["S"],
        created_at=int(item["created_at"]["N"]),
        expires_at=int(item["expires_at"]["N"]),
        tokens_used=int(item["tokens_used"]["N"]),
        token_limit=int(item["token_limit"]["N"]),
        status=item["status"]["S"],
    )


class DynamoSessionStore:
    def __init__(self, dynamodb_client, table_name: str, clock=time.time):
        self._db = dynamodb_client
        self._table = table_name
        self._clock = clock

    def create(self, *, user_sub: str, username: str, client_ip: str, ttl_seconds: int, token_limit: int) -> Session:
        now = int(self._clock())
        session = Session(
            session_id=secrets.token_urlsafe(32),
            user_sub=user_sub,
            username=username,
            client_ip=client_ip,
            created_at=now,
            expires_at=now + ttl_seconds,
            tokens_used=0,
            token_limit=token_limit,
            status="active",
        )
        self._db.put_item(
            TableName=self._table,
            Item={
                "session_id": {"S": session.session_id},
                "user_sub": {"S": session.user_sub},
                "username": {"S": session.username},
                "client_ip": {"S": session.client_ip},
                "created_at": {"N": str(session.created_at)},
                "expires_at": {"N": str(session.expires_at)},
                "tokens_used": {"N": "0"},
                "token_limit": {"N": str(session.token_limit)},
                "status": {"S": session.status},
            },
            ConditionExpression="attribute_not_exists(session_id)",
        )
        return session

    def get_active(self, session_id: str) -> Session:
        """Return the session, or raise SessionInvalid if it is unknown, revoked or expired."""
        if not session_id:
            raise SessionInvalid("No session.")
        item = self._db.get_item(
            TableName=self._table, Key={"session_id": {"S": session_id}}, ConsistentRead=True
        ).get("Item")
        if not item:
            raise SessionInvalid("Unknown session.")
        session = _to_session(item)
        # DynamoDB TTL deletes lazily, so expiry is always checked explicitly.
        if session.status != "active" or session.expires_at <= int(self._clock()):
            raise SessionInvalid("Session revoked or expired.")
        return session

    def revoke(self, session_id: str) -> None:
        self._db.update_item(
            TableName=self._table,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET #status = :revoked",
            ConditionExpression="attribute_exists(session_id)",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":revoked": {"S": "revoked"}},
        )

    def reserve_tokens(self, session_id: str, tokens: int) -> None:
        """Atomically add ``tokens`` to the session, only if it stays within its limit."""
        tokens = max(int(tokens), 0)
        limit = self._limit_of(session_id)
        try:
            self._db.update_item(
                TableName=self._table,
                Key={"session_id": {"S": session_id}},
                UpdateExpression="ADD tokens_used :tokens",
                # Conditions cannot do arithmetic, so "used + tokens <= limit" is written
                # as "used <= limit - tokens". token_limit = :limit pins the value read.
                ConditionExpression=(
                    "attribute_exists(session_id) AND #status = :active AND expires_at > :now "
                    "AND token_limit = :limit AND tokens_used <= :max_before"
                ),
                ExpressionAttributeNames={"#status": "status"},
                ExpressionAttributeValues={
                    ":tokens": {"N": str(tokens)},
                    ":active": {"S": "active"},
                    ":now": {"N": str(int(self._clock()))},
                    ":limit": {"N": str(limit)},
                    ":max_before": {"N": str(limit - tokens)},
                },
            )
        except ClientError as error:
            if error.response["Error"].get("Code") == "ConditionalCheckFailedException":
                raise SessionTokenLimitExceeded("The session token limit has been reached.") from error
            raise

    def adjust_tokens(self, session_id: str, delta: int) -> None:
        """Correct a reservation once the real usage is known (delta is usually negative)."""
        if delta == 0:
            return
        self._db.update_item(
            TableName=self._table,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="ADD tokens_used :delta",
            ConditionExpression="attribute_exists(session_id)",
            ExpressionAttributeValues={":delta": {"N": str(int(delta))}},
        )

    def _limit_of(self, session_id: str) -> int:
        item = self._db.get_item(
            TableName=self._table,
            Key={"session_id": {"S": session_id}},
            ProjectionExpression="token_limit",
            ConsistentRead=True,
        ).get("Item")
        if not item:
            raise SessionTokenLimitExceeded("Unknown session.")
        return int(item["token_limit"]["N"])


class SessionTokenBudget:
    """Token budget of one session, handed to the generation pipeline."""

    def __init__(self, store: DynamoSessionStore, session_id: str):
        self._store = store
        self.session_id = session_id

    def reserve(self, tokens: int) -> None:
        self._store.reserve_tokens(self.session_id, tokens)

    def settle(self, reserved: int, actual: int) -> None:
        try:
            self._store.adjust_tokens(self.session_id, int(actual) - int(reserved))
        except ClientError:
            # The reservation stays charged in full: failing here never gives tokens away.
            logger.exception("Could not settle token usage for a session.")
