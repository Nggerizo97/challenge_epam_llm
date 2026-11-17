"""Account and session workflow: Cognito identity, DynamoDB sessions and abuse limits."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.auth.cognito import AuthError, CognitoAuth, Identifier, InvalidCredentials, identity_key, parse_identifier
from src.auth.rate_limit import DynamoRateLimiter, RateLimitExceeded, ip_bucket
from src.auth.sessions import DynamoSessionStore, Session, SessionInvalid, SessionTokenBudget, TokenLedger

logger = logging.getLogger("AuthService")

SIGNUP_SCOPE = "signup-ip"
LOGIN_FAIL_IP_SCOPE = "login-fail-ip"
LOGIN_FAIL_ACCOUNT_SCOPE = "login-fail-account"


@dataclass(frozen=True)
class AuthPolicy:
    session_token_limit: int
    session_ttl_seconds: int
    signup_limit_per_ip: int
    signup_window_seconds: int
    auto_confirm: bool
    login_max_failures_per_ip: int = 20
    login_max_failures_per_account: int = 5
    login_failure_window_seconds: int = 900


@dataclass(frozen=True)
class RegisterResult:
    confirmed: bool
    delivery_destination: str | None


@dataclass(frozen=True)
class LoginResult:
    session: Session
    access_token: str


@dataclass(frozen=True)
class Usage:
    session_remaining: int
    session_limit: int
    user_today_remaining: int
    user_daily_limit: int


def _minutes(seconds: int) -> int:
    return max((seconds + 59) // 60, 1)


class AuthService:
    def __init__(
        self,
        cognito: CognitoAuth,
        sessions: DynamoSessionStore,
        rate_limiter: DynamoRateLimiter,
        ledger: TokenLedger,
        policy: AuthPolicy,
    ):
        self.cognito = cognito
        self.sessions = sessions
        self.rate_limiter = rate_limiter
        self.ledger = ledger
        self.policy = policy

    # --- accounts -------------------------------------------------------------------

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

    def delete_account(self, session_id: str, access_token: str, password: str, client_ip: str | None) -> None:
        """Delete the signed-in user's account after re-checking their password.

        Daily token usage and the per-IP sign-up count are kept on purpose: they are
        keyed by identity and IP, so deleting and re-creating an account resets nothing.
        """
        session = self.validate(session_id, access_token)
        self._authenticate(parse_identifier(session.username), password, ip_bucket(client_ip))
        # Cognito invalidates every token of the user, so their other sessions fail
        # validation from now on as well.
        self.cognito.delete_user(access_token)
        self.sessions.revoke(session_id)
        logger.info("Account deleted from %s.", ip_bucket(client_ip))

    # --- sessions -------------------------------------------------------------------

    def _authenticate(self, identifier: Identifier, password: str, bucket: str):
        """Cognito sign-in, refused while too many recent attempts have failed."""
        account = identity_key(identifier)
        window = self.policy.login_failure_window_seconds
        ip_failures, ip_reset = self.rate_limiter.count(LOGIN_FAIL_IP_SCOPE, bucket, window)
        account_failures, account_reset = self.rate_limiter.count(LOGIN_FAIL_ACCOUNT_SCOPE, account, window)
        if ip_failures >= self.policy.login_max_failures_per_ip:
            logger.warning("Sign-in blocked: too many failures from %s.", bucket)
            raise AuthError(f"Too many failed sign-in attempts. Try again in {_minutes(ip_reset)} min.")
        if account_failures >= self.policy.login_max_failures_per_account:
            logger.warning("Sign-in blocked: too many failures for one account (from %s).", bucket)
            raise AuthError(f"Too many failed sign-in attempts. Try again in {_minutes(account_reset)} min.")

        try:
            return self.cognito.sign_in(identifier, password)
        except InvalidCredentials:
            self.rate_limiter.add(LOGIN_FAIL_IP_SCOPE, bucket, window)
            self.rate_limiter.add(LOGIN_FAIL_ACCOUNT_SCOPE, account, window)
            raise

    def login(self, raw_identifier: str, password: str, client_ip: str | None) -> LoginResult:
        identifier = parse_identifier(raw_identifier)
        bucket = ip_bucket(client_ip)
        signed_in = self._authenticate(identifier, password, bucket)
        user_sub = self.cognito.user_sub(signed_in.access_token)
        session = self.sessions.create(
            user_sub=user_sub,
            username=identifier.username,
            usage_key=identity_key(identifier),
            client_ip=bucket,
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
        return SessionTokenBudget(self.ledger, session)

    def usage(self, session: Session) -> Usage:
        user_today, _ = self.ledger.usage_today(session)
        daily = self.ledger.limits.user_daily
        return Usage(
            session_remaining=session.tokens_remaining,
            session_limit=session.token_limit,
            user_today_remaining=max(daily - user_today, 0),
            user_daily_limit=daily,
        )

    def logout(self, session_id: str, access_token: str) -> None:
        try:
            self.sessions.revoke(session_id)
        finally:
            self.cognito.sign_out(access_token)
