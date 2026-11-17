"""Integration tests against a running AWS Emulator (skipped when none is reachable).

Start the emulator and export its variables first (see README, "AWS Emulator"):

    aws-emulator
    source scripts/emulator_env.sh
    python -m pytest tests/test_auth_emulator.py
"""
import os
import time
import unittest
import uuid

from src.auth.cognito import AuthError, CognitoAuth
from src.auth.rate_limit import DynamoRateLimiter
from src.auth.service import AuthPolicy, AuthService
from src.auth.sessions import DynamoSessionStore, TokenLedger, TokenLimitExceeded, TokenLimits

PASSWORD = "Str0ngPassword!"


def _emulator_available() -> bool:
    if not os.getenv("AWS_ENDPOINT_URL"):
        return False
    try:
        import boto3

        boto3.client("sts").get_caller_identity()
        return True
    except Exception:
        return False


@unittest.skipUnless(_emulator_available(), "AWS Emulator is not running / AWS_ENDPOINT_URL is not set")
class EmulatorAuthFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import boto3

        suffix = uuid.uuid4().hex[:8]
        cls.dynamodb = boto3.client("dynamodb")
        cls.idp = boto3.client("cognito-idp")
        cls.sessions_table = f"test-sessions-{suffix}"
        cls.limits_table = f"test-limits-{suffix}"
        for name, key in ((cls.sessions_table, "session_id"), (cls.limits_table, "limit_key")):
            cls.dynamodb.create_table(
                TableName=name,
                BillingMode="PAY_PER_REQUEST",
                AttributeDefinitions=[{"AttributeName": key, "AttributeType": "S"}],
                KeySchema=[{"AttributeName": key, "KeyType": "HASH"}],
            )
            cls.dynamodb.get_waiter("table_exists").wait(TableName=name)
        pool_id = cls.idp.create_user_pool(
            PoolName=f"test-pool-{suffix}", UsernameAttributes=["email", "phone_number"]
        )["UserPool"]["Id"]
        client_id = cls.idp.create_user_pool_client(
            UserPoolId=pool_id,
            ClientName="test",
            GenerateSecret=False,
            ExplicitAuthFlows=["ALLOW_USER_PASSWORD_AUTH", "ALLOW_REFRESH_TOKEN_AUTH"],
        )["UserPoolClient"]["ClientId"]
        cls.pool_id = pool_id
        cls.cognito = CognitoAuth(cls.idp, pool_id, client_id)

    @classmethod
    def tearDownClass(cls):
        for name in (cls.sessions_table, cls.limits_table):
            cls.dynamodb.delete_table(TableName=name)
        cls.idp.delete_user_pool(UserPoolId=cls.pool_id)

    def _service(
        self,
        token_limit=5000,
        user_daily=100_000,
        ip_daily=1_000_000,
        signup_limit=100,
        max_failures_account=5,
        max_failures_ip=20,
        clock=time.time,
    ):
        return AuthService(
            cognito=self.cognito,
            sessions=DynamoSessionStore(self.dynamodb, self.sessions_table, clock=clock),
            rate_limiter=DynamoRateLimiter(self.dynamodb, self.limits_table, clock=clock),
            ledger=TokenLedger(
                self.dynamodb,
                self.sessions_table,
                self.limits_table,
                TokenLimits(user_daily=user_daily, ip_daily=ip_daily),
                clock=clock,
            ),
            policy=AuthPolicy(
                session_token_limit=token_limit,
                session_ttl_seconds=3600,
                signup_limit_per_ip=signup_limit,
                signup_window_seconds=86400,
                auto_confirm=True,
                login_max_failures_per_ip=max_failures_ip,
                login_max_failures_per_account=max_failures_account,
                login_failure_window_seconds=900,
            ),
        )

    @staticmethod
    def _email(domain="example.com"):
        return f"user{uuid.uuid4().hex[:10]}@{domain}"

    @staticmethod
    def _ip():
        return f"203.0.{uuid.uuid4().int % 250}.{uuid.uuid4().int % 250 + 1}"

    def _signed_in(self, service, identifier=None, ip=None):
        identifier, ip = identifier or self._email(), ip or self._ip()
        service.register(identifier, PASSWORD, ip)
        return service.login(identifier, PASSWORD, ip)

    # --- sign-up, sign-in and sessions ----------------------------------------------

    def test_sign_up_sign_in_creates_a_dynamodb_session(self):
        service = self._service()
        email = self._email()
        self.assertTrue(service.register(email, PASSWORD, "203.0.113.10").confirmed)

        login = service.login(email, PASSWORD, "203.0.113.10")
        session = service.validate(login.session.session_id, login.access_token)

        self.assertEqual(session.username, email)
        self.assertEqual(session.client_ip, "203.0.113.10")
        self.assertEqual(session.tokens_used, 0)
        item = self.dynamodb.get_item(
            TableName=self.sessions_table, Key={"session_id": {"S": session.session_id}}
        )["Item"]
        stored = " ".join(str(value) for value in item.values())
        self.assertNotIn(login.access_token, stored, "Cognito tokens must not be stored in DynamoDB")

    def test_phone_number_accounts_work(self):
        service = self._service()
        phone = "+57300" + str(uuid.uuid4().int)[:7]
        login = self._signed_in(service, phone)
        self.assertEqual(login.session.username, phone)

    def test_wrong_password_and_unknown_user_get_the_same_message(self):
        service = self._service()
        email = self._email()
        service.register(email, PASSWORD, "203.0.113.12")
        with self.assertRaises(AuthError) as wrong_password:
            service.login(email, "WrongPassword1!", "203.0.113.12")
        with self.assertRaises(AuthError) as unknown_user:
            service.login(self._email(), PASSWORD, "203.0.113.12")
        self.assertEqual(str(wrong_password.exception), str(unknown_user.exception))

    def test_logout_revokes_both_the_session_and_the_cognito_token(self):
        service = self._service()
        login = self._signed_in(service)
        service.logout(login.session.session_id, login.access_token)

        with self.assertRaises(AuthError):
            service.validate(login.session.session_id, login.access_token)
        with self.assertRaises(TokenLimitExceeded):
            service.token_budget(login.session).reserve(1)
        with self.assertRaises(AuthError):
            self.cognito.user_sub(login.access_token)

    def test_session_cannot_be_used_with_another_users_token(self):
        service = self._service()
        alice, bob = self._signed_in(service), self._signed_in(service)
        with self.assertRaises(AuthError):
            service.validate(alice.session.session_id, bob.access_token)

    def test_expired_session_is_rejected(self):
        now = [time.time()]
        service = self._service(clock=lambda: now[0])
        login = self._signed_in(service)
        now[0] += 3601
        with self.assertRaises(AuthError):
            service.validate(login.session.session_id, login.access_token)
        with self.assertRaises(TokenLimitExceeded):
            service.token_budget(login.session).reserve(1)

    # --- per-IP sign-up limit -----------------------------------------------------

    def test_ip_cannot_create_more_accounts_than_the_limit(self):
        service = self._service(signup_limit=2)
        ip = self._ip()
        service.register(self._email(), PASSWORD, ip)
        service.register(self._email(), PASSWORD, ip)
        with self.assertRaisesRegex(AuthError, "Too many accounts"):
            service.register(self._email(), PASSWORD, ip)
        with self.assertRaisesRegex(AuthError, "Too many accounts"):
            service.register("+573009998877", PASSWORD, ip)
        service.register(self._email(), PASSWORD, self._ip())  # another IP is not affected

    def test_rejected_sign_ups_do_not_use_up_the_ip_limit(self):
        service = self._service(signup_limit=1)
        ip = self._ip()
        with self.assertRaises(AuthError):
            service.register(self._email(), "weak", ip)
        service.register(self._email(), PASSWORD, ip)  # still allowed

    def test_ip_limit_resets_after_the_window(self):
        now = [1_800_000_000]
        service = self._service(signup_limit=1, clock=lambda: now[0])
        ip = self._ip()
        service.register(self._email(), PASSWORD, ip)
        with self.assertRaises(AuthError):
            service.register(self._email(), PASSWORD, ip)
        now[0] += 86400
        service.register(self._email(), PASSWORD, ip)

    # --- token budgets ----------------------------------------------------------------

    def test_session_token_limit_is_enforced(self):
        service = self._service(token_limit=1000)
        login = self._signed_in(service)
        budget = service.token_budget(login.session)

        budget.reserve(900)
        budget.settle(900, 600)  # provider reported 600
        budget.reserve(400)  # 600 + 400 = 1000: exactly at the limit
        with self.assertRaisesRegex(TokenLimitExceeded, "this session"):
            budget.reserve(1)

        session = service.validate(login.session.session_id, login.access_token)
        self.assertEqual((session.tokens_used, session.tokens_remaining), (1000, 0))

    def test_signing_in_again_does_not_reset_the_daily_budget(self):
        service = self._service(token_limit=1000, user_daily=1500)
        email, ip = self._email(), self._ip()
        first = self._signed_in(service, email, ip)
        service.token_budget(first.session).reserve(1000)

        second = service.login(email, PASSWORD, ip)
        self.assertEqual(second.session.tokens_used, 0)  # a new session...
        with self.assertRaisesRegex(TokenLimitExceeded, "daily token limit"):
            service.token_budget(second.session).reserve(600)  # ...but not more tokens today
        service.token_budget(second.session).reserve(500)
        self.assertEqual(service.usage(second.session).user_today_remaining, 0)

    def test_refused_reservation_charges_nothing(self):
        service = self._service(token_limit=1000, user_daily=500)
        login = self._signed_in(service)
        with self.assertRaises(TokenLimitExceeded):
            service.token_budget(login.session).reserve(600)
        session = service.validate(login.session.session_id, login.access_token)
        self.assertEqual(session.tokens_used, 0, "the transaction must be all-or-nothing")

    def test_email_aliases_share_the_daily_budget(self):
        service = self._service(token_limit=1000, user_daily=1000)
        local = f"ana{uuid.uuid4().hex[:6]}"
        ip = self._ip()
        first = self._signed_in(service, f"{local}@gmail.com", ip)
        service.token_budget(first.session).reserve(1000)
        alias = self._signed_in(service, f"{local[:2]}.{local[2:]}+x@gmail.com", ip)
        with self.assertRaisesRegex(TokenLimitExceeded, "daily token limit"):
            service.token_budget(alias.session).reserve(1)

    def test_many_accounts_from_one_ip_share_the_ip_daily_budget(self):
        service = self._service(token_limit=1000, user_daily=1000, ip_daily=1500)
        ip = self._ip()
        service.token_budget(self._signed_in(service, ip=ip).session).reserve(1000)
        second = self._signed_in(service, ip=ip)
        with self.assertRaisesRegex(TokenLimitExceeded, "your network"):
            service.token_budget(second.session).reserve(600)
        service.token_budget(self._signed_in(service).session).reserve(600)  # other IP: fine

    def test_daily_budget_resets_the_next_utc_day(self):
        now = [1_800_000_000 - 1_800_000_000 % 86400 + 3600]  # 01:00 UTC
        service = self._service(token_limit=5000, user_daily=1000, clock=lambda: now[0])
        email, ip = self._email(), self._ip()
        service.token_budget(self._signed_in(service, email, ip).session).reserve(1000)
        now[0] += 86400
        service.token_budget(service.login(email, PASSWORD, ip).session).reserve(1000)

    # --- failed sign-in lockout -------------------------------------------------------

    def test_account_is_locked_after_repeated_wrong_passwords(self):
        service = self._service(max_failures_account=3)
        email, ip = self._email(), self._ip()
        service.register(email, PASSWORD, ip)
        for _ in range(3):
            with self.assertRaisesRegex(AuthError, "Incorrect"):
                service.login(email, "WrongPassword1!", ip)
        with self.assertRaisesRegex(AuthError, "Too many failed"):
            service.login(email, PASSWORD, ip)  # even the right password, while locked
        with self.assertRaisesRegex(AuthError, "Too many failed"):
            service.login(email, PASSWORD, self._ip())  # from another IP too
        other = self._email()
        service.register(other, PASSWORD, ip)
        service.login(other, PASSWORD, ip)  # other accounts are not affected

    def test_ip_is_blocked_after_failures_across_accounts(self):
        service = self._service(max_failures_ip=3, max_failures_account=100)
        ip = self._ip()
        for _ in range(3):
            with self.assertRaises(AuthError):
                service.login(self._email(), PASSWORD, ip)  # guessing unknown accounts
        email = self._email()
        service.register(email, PASSWORD, self._ip())
        with self.assertRaisesRegex(AuthError, "Too many failed"):
            service.login(email, PASSWORD, ip)
        service.login(email, PASSWORD, self._ip())

    # --- account deletion -------------------------------------------------------------

    def test_delete_account_requires_the_password(self):
        service = self._service()
        login = self._signed_in(service)
        with self.assertRaisesRegex(AuthError, "Incorrect"):
            service.delete_account(login.session.session_id, login.access_token, "WrongPassword1!", self._ip())
        service.validate(login.session.session_id, login.access_token)  # still there

    def test_deleted_account_and_all_its_sessions_stop_working(self):
        service = self._service()
        email, ip = self._email(), self._ip()
        first = self._signed_in(service, email, ip)
        second = service.login(email, PASSWORD, ip)

        service.delete_account(first.session.session_id, first.access_token, PASSWORD, ip)

        for login in (first, second):
            with self.assertRaises(AuthError):
                service.validate(login.session.session_id, login.access_token)
        with self.assertRaisesRegex(AuthError, "Incorrect"):
            service.login(email, PASSWORD, ip)

    def test_re_creating_a_deleted_account_does_not_reset_limits(self):
        service = self._service(token_limit=1000, user_daily=1000, signup_limit=2)
        email, ip = self._email(), self._ip()
        login = self._signed_in(service, email, ip)
        service.token_budget(login.session).reserve(1000)
        service.delete_account(login.session.session_id, login.access_token, PASSWORD, ip)

        again = self._signed_in(service, email, ip)  # second sign-up from this IP
        with self.assertRaisesRegex(TokenLimitExceeded, "daily token limit"):
            service.token_budget(again.session).reserve(1)
        with self.assertRaisesRegex(AuthError, "Too many accounts"):
            service.register(self._email(), PASSWORD, ip)  # the IP slot was not given back


if __name__ == "__main__":
    unittest.main()
