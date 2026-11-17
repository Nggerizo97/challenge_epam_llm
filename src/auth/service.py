"""Account and session workflow: Cognito identity + DynamoDB sessions + IP limits."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.auth.cognito import AuthError, CognitoAuth, parse_identifier
from src.auth.rate_limit import DynamoRateLimiter, RateLimitExceeded, ip_bucket
from src.auth.sessions import DynamoSessionStore, Session, SessionInvalid, SessionTokenBudget

logger = logging.getLogger("AuthService")

SIGNUP_SCOPE = "signup-ip"


@dataclass(frozen=True)
class AuthPolicy:
    session_token_limit: int
    session_ttl_seconds: int
    signup_limit_per_ip: int
    signup_window_seconds: int
    auto_confirm: bool


@dataclass(frozen=True)
class RegisterResult:
    confirmed: bool
    delivery_destination: str | None


@dataclass(frozen=True)
class LoginResult:
    session: Session
    access_token: str


class AuthService:
    def __init__(
        self,
        cognito: CognitoAuth,
        sessions: DynamoSessionStore,
        rate_limiter: DynamoRateLimiter,
        policy: AuthPolicy,
    ):
        self.cognito = cognito
        self.sessions = sessions
        self.rate_limiter = rate_limiter
        self.policy = policy

    def register(self, raw_identifier: str, password: str, client_ip: str | None) -> RegisterResult:
        identifier = parse_identifier(raw_identifier)
        bucket = ip_bucket(client_ip)
        try:
            # Reserved before calling Cognito so parallel sign-ups cannot slip past it.
            hit = self.rate_limiter.hit(
                SIGNUP_SCOPE, bucket, self.policy.signup_limit_per_ip, self.policy.signup_window_seconds
            )
        except RateLimitExceeded as error:
            logger.warning("Sign-up blocked by the per-IP limit for %s.", bucket)
            hours = max(round(error.retry_after_seconds / 3600), 1)
            raise AuthError(
                f"Too many accounts were created from your network. Try again in about {hours} h."
            ) from error

        try:
            result = self.cognito.sign_up(identifier, password)
        except AuthError:
            # Input rejected (weak password, duplicate...): no account was created, so the
            # attempt does not count against the IP.
            self.rate_limiter.release(hit)
            raise

        confirmed = result.confirmed
        if not confirmed and self.policy.auto_confirm:
            self.cognito.admin_confirm(identifier)
            confirmed = True
        logger.info("Account created (%s) from %s.", identifier.attribute, bucket)
        return RegisterResult(confirmed=confirmed, delivery_destination=result.delivery_destination)

    def confirm(self, raw_identifier: str, code: str) -> None:
        self.cognito.confirm(parse_identifier(raw_identifier), code)

    def login(self, raw_identifier: str, password: str, client_ip: str | None) -> LoginResult:
        identifier = parse_identifier(raw_identifier)
        signed_in = self.cognito.sign_in(identifier, password)
        user_sub = self.cognito.user_sub(signed_in.access_token)
        session = self.sessions.create(
            user_sub=user_sub,
            username=identifier.username,
            client_ip=ip_bucket(client_ip),
            # The session never outlives the Cognito access token it is tied to.
            ttl_seconds=min(self.policy.session_ttl_seconds, signed_in.expires_in),
            token_limit=self.policy.session_token_limit,
        )
        return LoginResult(session=session, access_token=signed_in.access_token)

    def validate(self, session_id: str, access_token: str) -> Session:
        """Check both the DynamoDB session and the Cognito token, and that they match."""
        try:
            session = self.sessions.get_active(session_id)
        except SessionInvalid as error:
            raise AuthError("Your session has expired. Please sign in again.") from error
        if self.cognito.user_sub(access_token) != session.user_sub:
            raise AuthError("Your session could not be verified. Please sign in again.")
        return session

    def token_budget(self, session: Session) -> SessionTokenBudget:
        return SessionTokenBudget(self.sessions, session.session_id)

    def logout(self, session_id: str, access_token: str) -> None:
        try:
            self.sessions.revoke(session_id)
        finally:
            self.cognito.sign_out(access_token)
