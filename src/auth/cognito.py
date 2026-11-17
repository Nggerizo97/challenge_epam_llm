"""Thin wrapper over the Cognito user pool API (sign-up, sign-in, token checks)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from botocore.exceptions import ClientError

_EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]{1,253}\.[A-Za-z]{2,}$")
# E.164: "+", country code, up to 15 digits in total.
_PHONE_RE = re.compile(r"^\+[1-9]\d{6,14}$")


class AuthError(Exception):
    """An error whose message is safe to show to the user."""


class InvalidCredentials(AuthError):
    """Wrong username or password (counted as a failed sign-in)."""


@dataclass(frozen=True)
class Identifier:
    username: str
    attribute: str  # "email" or "phone_number"


def parse_identifier(raw: str) -> Identifier:
    """Accept an e-mail address or a phone number in international format."""
    value = (raw or "").strip()
    if "@" in value:
        email = value.lower()
        if not _EMAIL_RE.match(email):
            raise AuthError("Enter a valid e-mail address.")
        return Identifier(email, "email")
    phone = re.sub(r"[\s().-]", "", value)
    if not _PHONE_RE.match(phone):
        raise AuthError("Enter a valid e-mail or a phone number in international format, e.g. +573001234567.")
    return Identifier(phone, "phone_number")


# Providers that ignore dots in the local part, so "a.b@gmail.com" is "ab@gmail.com".
_DOT_INSENSITIVE_DOMAINS = {"gmail.com": "gmail.com", "googlemail.com": "gmail.com"}


def identity_key(identifier: Identifier) -> str:
    """Stable, hashed key for the person behind an identifier, used for usage limits.

    E-mail aliases that reach the same inbox ("ana+1@x.com", "a.na@gmail.com") map to
    the same key, so they share one daily token budget. The key is hashed so counters
    in DynamoDB do not contain e-mail addresses or phone numbers.
    """
    value = identifier.username
    if identifier.attribute == "email":
        local, _, domain = value.partition("@")
        local = local.split("+", 1)[0]
        if domain in _DOT_INSENSITIVE_DOMAINS:
            domain = _DOT_INSENSITIVE_DOMAINS[domain]
            local = local.replace(".", "")
        value = f"{local}@{domain}"
    return hashlib.sha256(f"{identifier.attribute}:{value}".encode()).hexdigest()


@dataclass(frozen=True)
class SignUpResult:
    user_sub: str
    confirmed: bool
    delivery_destination: str | None


@dataclass(frozen=True)
class SignInResult:
    access_token: str
    expires_in: int


def _code(error: ClientError) -> str:
    return error.response["Error"].get("Code", "")


class CognitoAuth:
    def __init__(self, cognito_client, user_pool_id: str, client_id: str):
        self._idp = cognito_client
        self.user_pool_id = user_pool_id
        self.client_id = client_id

    def sign_up(self, identifier: Identifier, password: str) -> SignUpResult:
        try:
            response = self._idp.sign_up(
                ClientId=self.client_id,
                Username=identifier.username,
                Password=password,
                UserAttributes=[{"Name": identifier.attribute, "Value": identifier.username}],
            )
        except ClientError as error:
            code = _code(error)
            if code == "UsernameExistsException":
                raise AuthError("An account with this e-mail or phone already exists.") from error
            if code == "InvalidPasswordException":
                # Cognito's message states which password-policy rule failed.
                message = error.response["Error"].get("Message") or "The password is too weak."
                raise AuthError(message) from error
            if code == "InvalidParameterException":
                raise AuthError("The e-mail or phone number is not valid.") from error
            raise
        delivery = response.get("CodeDeliveryDetails") or {}
        return SignUpResult(
            user_sub=response["UserSub"],
            confirmed=bool(response.get("UserConfirmed")),
            delivery_destination=delivery.get("Destination"),
        )

    def admin_confirm(self, identifier: Identifier) -> None:
        self._idp.admin_confirm_sign_up(UserPoolId=self.user_pool_id, Username=identifier.username)

    def confirm(self, identifier: Identifier, code: str) -> None:
        try:
            self._idp.confirm_sign_up(
                ClientId=self.client_id, Username=identifier.username, ConfirmationCode=code.strip()
            )
        except ClientError as error:
            if _code(error) in {"CodeMismatchException", "ExpiredCodeException"}:
                raise AuthError("The confirmation code is wrong or has expired.") from error
            if _code(error) in {"UserNotFoundException", "NotAuthorizedException"}:
                raise AuthError("This account cannot be confirmed.") from error
            raise

    def sign_in(self, identifier: Identifier, password: str) -> SignInResult:
        try:
            response = self._idp.initiate_auth(
                ClientId=self.client_id,
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={"USERNAME": identifier.username, "PASSWORD": password},
            )
        except ClientError as error:
            code = _code(error)
            if code == "UserNotConfirmedException":
                raise AuthError("Confirm your account before signing in.") from error
            if code in {"NotAuthorizedException", "UserNotFoundException"}:
                # Same message for both, so sign-in does not reveal which accounts exist.
                raise InvalidCredentials("Incorrect e-mail/phone or password.") from error
            if code == "PasswordResetRequiredException":
                raise AuthError("A password reset is required for this account.") from error
            raise
        result = response.get("AuthenticationResult")
        if not result:
            # MFA or other challenges are not enabled on this pool.
            raise AuthError("This sign-in requires a step that the app does not support.")
        return SignInResult(access_token=result["AccessToken"], expires_in=int(result["ExpiresIn"]))

    def user_sub(self, access_token: str) -> str:
        """Validate the access token with Cognito and return the user's ``sub``.

        Cognito rejects expired, malformed and revoked (signed-out) tokens here.
        """
        try:
            response = self._idp.get_user(AccessToken=access_token)
        except ClientError as error:
            if _code(error) in {"NotAuthorizedException", "UserNotFoundException"}:
                raise AuthError("Your session has expired. Please sign in again.") from error
            raise
        for attribute in response.get("UserAttributes", []):
            if attribute["Name"] == "sub":
                return attribute["Value"]
        raise AuthError("Your session could not be verified. Please sign in again.")

    def delete_user(self, access_token: str) -> None:
        """Delete the signed-in user. Cognito invalidates all of their tokens."""
        try:
            self._idp.delete_user(AccessToken=access_token)
        except ClientError as error:
            if _code(error) in {"NotAuthorizedException", "UserNotFoundException"}:
                raise AuthError("Your session has expired. Please sign in again.") from error
            raise

    def sign_out(self, access_token: str) -> None:
        """Revoke every token issued to this user's current sign-ins."""
        try:
            self._idp.global_sign_out(AccessToken=access_token)
        except ClientError as error:
            if _code(error) != "NotAuthorizedException":  # already expired or revoked
                raise
