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
from src.auth.sessions import DynamoSessionStore, SessionTokenLimitExceeded


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

    def _service(self, token_limit=5000, signup_limit=2, clock=time.time):
        return AuthService(
            cognito=self.cognito,
            sessions=DynamoSessionStore(self.dynamodb, self.sessions_table, clock=clock),
            rate_limiter=DynamoRateLimiter(self.dynamodb, self.limits_table, clock=clock),
            policy=AuthPolicy(
                session_token_limit=token_limit,
                session_ttl_seconds=3600,
                signup_limit_per_ip=signup_limit,
                signup_window_seconds=86400,
                auto_confirm=True,
            ),
        )

    @staticmethod
    def _email():
        return f"user-{uuid.uuid4().hex[:10]}@example.com"

    def test_sign_up_sign_in_creates_a_dynamodb_session(self):
        service = self._service()
        email = self._email()
        self.assertTrue(service.register(email, "Str0ngPassword!", "203.0.113.10").confirmed)

        login = service.login(email, "Str0ngPassword!", "203.0.113.10")
        session = service.validate(login.session.session_id, login.access_token)

        self.assertEqual(session.username, email)
        self.assertEqual(session.client_ip, "203.0.113.10")
        self.assertEqual(session.tokens_used, 0)
        item = self.dynamodb.get_item(
            TableName=self.sessions_table, Key={"session_id": {"S": session.session_id}}
        )["Item"]
        self.assertNotIn("access_token", item, "Cognito tokens must not be stored in DynamoDB")

    def test_phone_number_accounts_work(self):
        service = self._service()
        phone = "+57300" + str(uuid.uuid4().int)[:7]
        service.register(phone, "Str0ngPassword!", "203.0.113.11")
        login = service.login(phone, "Str0ngPassword!", "203.0.113.11")
        self.assertEqual(login.session.username, phone)

    def test_wrong_password_and_unknown_user_get_the_same_message(self):
        service = self._service()
        email = self._email()
        service.register(email, "Str0ngPassword!", "203.0.113.12")
        with self.assertRaises(AuthError) as wrong_password:
            service.login(email, "WrongPassword1!", "203.0.113.12")
        with self.assertRaises(AuthError) as unknown_user:
            service.login(self._email(), "Str0ngPassword!", "203.0.113.12")
        self.assertEqual(str(wrong_password.exception), str(unknown_user.exception))

    def test_ip_cannot_create_more_accounts_than_the_limit(self):
        service = self._service(signup_limit=2)
        ip = "198.51.100.20"
        service.register(self._email(), "Str0ngPassword!", ip)
        service.register(self._email(), "Str0ngPassword!", ip)
        with self.assertRaisesRegex(AuthError, "Too many accounts"):
            service.register(self._email(), "Str0ngPassword!", ip)
        with self.assertRaisesRegex(AuthError, "Too many accounts"):
            service.register("+573009998877", "Str0ngPassword!", ip)
        # Another IP is not affected.
        service.register(self._email(), "Str0ngPassword!", "198.51.100.21")

    def test_rejected_sign_ups_do_not_use_up_the_ip_limit(self):
        service = self._service(signup_limit=1)
        ip = "198.51.100.30"
        with self.assertRaises(AuthError):
            service.register(self._email(), "weak", ip)
        service.register(self._email(), "Str0ngPassword!", ip)  # still allowed

    def test_ip_limit_resets_after_the_window(self):
        now = [1_800_000_000]
        service = self._service(signup_limit=1, clock=lambda: now[0])
        ip = "198.51.100.40"
        service.register(self._email(), "Str0ngPassword!", ip)
        with self.assertRaises(AuthError):
            service.register(self._email(), "Str0ngPassword!", ip)
        now[0] += 86400
        service.register(self._email(), "Str0ngPassword!", ip)

    def test_session_token_limit_is_enforced(self):
        service = self._service(token_limit=1000)
        email = self._email()
        service.register(email, "Str0ngPassword!", "203.0.113.50")
        login = service.login(email, "Str0ngPassword!", "203.0.113.50")
        budget = service.token_budget(login.session)

        budget.reserve(900)
        budget.settle(900, 600)  # provider reported 600
        budget.reserve(400)  # 600 + 400 = 1000: exactly at the limit
        with self.assertRaises(SessionTokenLimitExceeded):
            budget.reserve(1)

        session = service.validate(login.session.session_id, login.access_token)
        self.assertEqual((session.tokens_used, session.tokens_remaining), (1000, 0))

    def test_new_session_does_not_inherit_or_share_the_budget(self):
        service = self._service(token_limit=1000)
        email = self._email()
        service.register(email, "Str0ngPassword!", "203.0.113.51")
        first = service.login(email, "Str0ngPassword!", "203.0.113.51")
        service.token_budget(first.session).reserve(1000)
        second = service.login(email, "Str0ngPassword!", "203.0.113.51")
        self.assertEqual(service.validate(second.session.session_id, second.access_token).tokens_used, 0)

    def test_logout_revokes_both_the_session_and_the_cognito_token(self):
        service = self._service()
        email = self._email()
        service.register(email, "Str0ngPassword!", "203.0.113.60")
        login = service.login(email, "Str0ngPassword!", "203.0.113.60")
        service.logout(login.session.session_id, login.access_token)

        with self.assertRaises(AuthError):
            service.validate(login.session.session_id, login.access_token)
        with self.assertRaises(SessionTokenLimitExceeded):
            service.token_budget(login.session).reserve(1)
        with self.assertRaises(AuthError):
            self.cognito.user_sub(login.access_token)

    def test_session_cannot_be_used_with_another_users_token(self):
        service = self._service()
        alice, bob = self._email(), self._email()
        for email in (alice, bob):
            service.register(email, "Str0ngPassword!", "203.0.113.70")
        alice_login = service.login(alice, "Str0ngPassword!", "203.0.113.70")
        bob_login = service.login(bob, "Str0ngPassword!", "203.0.113.70")
        with self.assertRaises(AuthError):
            service.validate(alice_login.session.session_id, bob_login.access_token)

    def test_expired_session_is_rejected(self):
        now = [time.time()]
        service = self._service(clock=lambda: now[0])
        email = self._email()
        service.register(email, "Str0ngPassword!", "203.0.113.80")
        login = service.login(email, "Str0ngPassword!", "203.0.113.80")
        now[0] += 3601
        with self.assertRaises(AuthError):
            service.validate(login.session.session_id, login.access_token)
        with self.assertRaises(SessionTokenLimitExceeded):
            service.token_budget(login.session).reserve(1)


if __name__ == "__main__":
    unittest.main()
