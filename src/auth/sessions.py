"""User sessions in DynamoDB, and the token ledger that limits LLM usage.

A session is created after a successful Cognito sign-in. Its record holds the Cognito
user id, a hashed identity key, the client IP, an expiry and a ``tokens_used`` counter.

Every LLM call is charged to three budgets at once, in a single DynamoDB transaction:

* the session          (``SESSION_TOKEN_LIMIT``)
* the user, per UTC day (``USER_DAILY_TOKEN_LIMIT``) - keyed by the canonical e-mail /
  phone, so signing in again, or deleting the account and creating it again, does not
  give the same person a fresh budget;
* the client IP, per UTC day (``IP_DAILY_TOKEN_LIMIT``) - so several accounts created
  from one network cannot multiply the budget.

Before a call the worst case (prompt estimate + max output tokens) is reserved; the
transaction fails as a whole if any budget would be exceeded. After the call the
reservation is corrected to the tokens the provider reported.

Cognito tokens are never written to DynamoDB; only the opaque session id is stored.
"""
from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

from botocore.exceptions import ClientError

logger = logging.getLogger("SessionStore")

DAY_SECONDS = 86400


class SessionInvalid(Exception):
    """The session does not exist, was revoked or has expired."""


class TokenLimitExceeded(Exception):
    """A token budget has no room for this request. The message is user-facing."""


@dataclass(frozen=True)
class Session:
    session_id: str
    user_sub: str
    username: str
    usage_key: str
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
        usage_key=item["usage_key"]["S"],
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
        self.table = table_name
        self._clock = clock

    def create(
        self,
        *,
        user_sub: str,
        username: str,
        usage_key: str,
        client_ip: str,
        ttl_seconds: int,
        token_limit: int,
    ) -> Session:
        now = int(self._clock())
        session = Session(
            session_id=secrets.token_urlsafe(32),
            user_sub=user_sub,
            username=username,
            usage_key=usage_key,
            client_ip=client_ip,
            created_at=now,
            expires_at=now + ttl_seconds,
            tokens_used=0,
            token_limit=token_limit,
            status="active",
        )
        self._db.put_item(
            TableName=self.table,
            Item={
                "session_id": {"S": session.session_id},
                "user_sub": {"S": session.user_sub},
                "username": {"S": session.username},
                "usage_key": {"S": session.usage_key},
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
            TableName=self.table, Key={"session_id": {"S": session_id}}, ConsistentRead=True
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
            TableName=self.table,
            Key={"session_id": {"S": session_id}},
            UpdateExpression="SET #status = :revoked",
            ConditionExpression="attribute_exists(session_id)",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={":revoked": {"S": "revoked"}},
        )


@dataclass(frozen=True)
class TokenLimits:
    user_daily: int
    ip_daily: int


class TokenLedger:
    """Charges LLM tokens to the session, user-day and IP-day budgets atomically."""

    def __init__(self, dynamodb_client, sessions_table: str, counters_table: str, limits: TokenLimits, clock=time.time):
        self._db = dynamodb_client
        self._sessions_table = sessions_table
        self._counters_table = counters_table
        self.limits = limits
        self._clock = clock

    def _day_keys(self, session: Session) -> tuple[str, str, int]:
        now = int(self._clock())
        day = now - now % DAY_SECONDS
        return f"tokens-user#{session.usage_key}#{day}", f"tokens-ip#{session.client_ip}#{day}", day

    def usage_today(self, session: Session) -> tuple[int, int]:
        """Tokens used today by this user and by this IP."""
        user_key, ip_key, _ = self._day_keys(session)
        totals = []
        for key in (user_key, ip_key):
            item = self._db.get_item(
                TableName=self._counters_table, Key={"limit_key": {"S": key}}, ConsistentRead=True
            ).get("Item")
            totals.append(int(item["tokens"]["N"]) if item and "tokens" in item else 0)
        return totals[0], totals[1]

    def _counter_update(self, key: str, tokens: int, limit: int, expires_at: int) -> dict:
        return {
            "Update": {
                "TableName": self._counters_table,
                "Key": {"limit_key": {"S": key}},
                "UpdateExpression": "ADD tokens :tokens SET expires_at = if_not_exists(expires_at, :expires_at)",
                "ConditionExpression": "attribute_not_exists(tokens) OR tokens <= :max_before",
                "ExpressionAttributeValues": {
                    ":tokens": {"N": str(tokens)},
                    ":max_before": {"N": str(limit - tokens)},
                    ":expires_at": {"N": str(expires_at)},
                },
            }
        }

    def reserve(self, session: Session, tokens: int) -> None:
        tokens = max(int(tokens), 0)
        # A brand-new counter passes "attribute_not_exists", so oversize requests are
        # refused here rather than relying on the condition.
        if tokens > session.token_limit:
            raise TokenLimitExceeded("This question needs more tokens than a session allows.")
        if tokens > self.limits.user_daily:
            raise TokenLimitExceeded("This question needs more tokens than your daily limit allows.")
        if tokens > self.limits.ip_daily:
            raise TokenLimitExceeded("This question needs more tokens than the daily limit allows.")

        user_key, ip_key, day = self._day_keys(session)
        expires_at = day + 2 * DAY_SECONDS
        items = [
            {
                "Update": {
                    "TableName": self._sessions_table,
                    "Key": {"session_id": {"S": session.session_id}},
                    "UpdateExpression": "ADD tokens_used :tokens",
                    # Conditions cannot add two values, so "used + tokens <= limit" is
                    # written as "used <= limit - tokens"; token_limit pins the value used.
                    "ConditionExpression": (
                        "attribute_exists(session_id) AND #status = :active AND expires_at > :now "
                        "AND token_limit = :limit AND tokens_used <= :max_before"
                    ),
                    "ExpressionAttributeNames": {"#status": "status"},
                    "ExpressionAttributeValues": {
                        ":tokens": {"N": str(tokens)},
                        ":active": {"S": "active"},
                        ":now": {"N": str(int(self._clock()))},
                        ":limit": {"N": str(session.token_limit)},
                        ":max_before": {"N": str(session.token_limit - tokens)},
                    },
                }
            },
            self._counter_update(user_key, tokens, self.limits.user_daily, expires_at),
            self._counter_update(ip_key, tokens, self.limits.ip_daily, expires_at),
        ]
        try:
            self._db.transact_write_items(TransactItems=items)
        except ClientError as error:
            if error.response["Error"].get("Code") != "TransactionCanceledException":
                raise
            reasons = [r.get("Code") for r in error.response.get("CancellationReasons", [])]
            messages = (
                "You have used all the tokens available for this session.",
                "You have reached your daily token limit. It resets at 00:00 UTC.",
                "The daily token limit for your network has been reached. It resets at 00:00 UTC.",
            )
            for reason, message in zip(reasons, messages):
                if reason == "ConditionalCheckFailed":
                    logger.info("Token reservation refused: %s", message)
                    raise TokenLimitExceeded(message) from error
            raise  # cancelled for another reason (e.g. a conflicting write): fail closed

    def settle(self, session: Session, delta: int) -> None:
        """Correct a reservation once the real usage is known (delta is usually negative)."""
        if delta == 0:
            return
        user_key, ip_key, _ = self._day_keys(session)
        add = lambda table, key_name, key: {  # noqa: E731
            "Update": {
                "TableName": table,
                "Key": {key_name: {"S": key}},
                "UpdateExpression": "ADD #counter :delta",
                "ConditionExpression": f"attribute_exists({key_name})",
                "ExpressionAttributeNames": {"#counter": "tokens_used" if table == self._sessions_table else "tokens"},
                "ExpressionAttributeValues": {":delta": {"N": str(int(delta))}},
            }
        }
        self._db.transact_write_items(
            TransactItems=[
                add(self._sessions_table, "session_id", session.session_id),
                add(self._counters_table, "limit_key", user_key),
                add(self._counters_table, "limit_key", ip_key),
            ]
        )


class SessionTokenBudget:
    """Token budget of one signed-in session, handed to the generation pipeline."""

    def __init__(self, ledger: TokenLedger, session: Session):
        self._ledger = ledger
        self.session = session

    def reserve(self, tokens: int) -> None:
        self._ledger.reserve(self.session, tokens)

    def settle(self, reserved: int, actual: int) -> None:
        try:
            self._ledger.settle(self.session, int(actual) - int(reserved))
        except ClientError:
            # The reservation stays charged in full: failing here never gives tokens away.
            logger.exception("Could not settle token usage.")
