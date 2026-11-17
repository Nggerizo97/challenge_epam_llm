import unittest

from src.auth.cognito import AuthError, parse_identifier
from src.auth.rate_limit import ip_bucket
from src.auth.sessions import SessionTokenLimitExceeded
from src.config import settings
from src.generation import pipeline
from src.ui.client_ip import resolve_client_ip


class IdentifierTests(unittest.TestCase):
    def test_email_is_lowercased(self):
        identifier = parse_identifier("  Ana.Perez@Example.COM ")
        self.assertEqual((identifier.username, identifier.attribute), ("ana.perez@example.com", "email"))

    def test_phone_is_normalised_to_e164(self):
        identifier = parse_identifier("+57 (300) 123-4567")
        self.assertEqual((identifier.username, identifier.attribute), ("+573001234567", "phone_number"))

    def test_rejects_invalid_values(self):
        for raw in ("", "not-an-email@", "3001234567", "+0123456789", "a@b"):
            with self.assertRaises(AuthError, msg=raw):
                parse_identifier(raw)


class IpTests(unittest.TestCase):
    def test_ipv6_is_limited_per_64_block(self):
        self.assertEqual(ip_bucket("2001:db8:1:2:aaaa::1"), ip_bucket("2001:db8:1:2:bbbb::9"))
        self.assertEqual(ip_bucket("2001:db8:1:2::1"), "2001:db8:1:2::/64")

    def test_ipv4_mapped_ipv6_is_the_ipv4_address(self):
        self.assertEqual(ip_bucket("::ffff:203.0.113.7"), "203.0.113.7")

    def test_missing_or_garbage_ip_shares_one_bucket(self):
        self.assertEqual(ip_bucket(None), "unknown")
        self.assertEqual(ip_bucket("not-an-ip"), "unknown")

    def test_forwarded_for_is_ignored_without_trusted_proxies(self):
        headers = {"X-Forwarded-For": "1.1.1.1"}
        self.assertEqual(resolve_client_ip("203.0.113.7", headers, 0), "203.0.113.7")

    def test_forwarded_for_uses_the_entry_added_by_the_trusted_proxy(self):
        # The client forged "6.6.6.6"; the trusted proxy appended the real peer.
        headers = {"X-Forwarded-For": "6.6.6.6, 198.51.100.4"}
        self.assertEqual(resolve_client_ip("10.0.0.2", headers, 1), "198.51.100.4")

    def test_localhost_falls_back_to_loopback(self):
        self.assertEqual(resolve_client_ip(None, {}, 0), "127.0.0.1")


class _Budget:
    def __init__(self, limit):
        self.limit, self.used, self.calls = limit, 0, []

    def reserve(self, tokens):
        if self.used + tokens > self.limit:
            raise SessionTokenLimitExceeded()
        self.used += tokens
        self.calls.append(("reserve", tokens))

    def settle(self, reserved, actual):
        self.used += actual - reserved
        self.calls.append(("settle", reserved, actual))


class _LLM:
    def __init__(self, usage=None, fail=False):
        self.usage, self.fail, self.invoked = usage, fail, 0

    def invoke(self, prompt):
        self.invoked += 1
        if self.fail:
            raise RuntimeError("provider down")
        return type("Response", (), {"content": "ok [Source 1]", "usage_metadata": self.usage})()


class SessionBudgetInPipelineTests(unittest.TestCase):
    def setUp(self):
        self._count = pipeline._api_request_count
        pipeline._api_request_count = 0

    def tearDown(self):
        pipeline._api_request_count = self._count

    def test_charges_the_tokens_reported_by_the_provider(self):
        budget = _Budget(limit=100_000)
        pipeline._invoke_llm(_LLM(usage={"total_tokens": 321}), "x" * 300, budget)
        self.assertEqual(budget.used, 321)
        self.assertEqual(budget.calls[0], ("reserve", 100 + settings.max_tokens))

    def test_refuses_before_calling_the_provider_when_out_of_tokens(self):
        llm = _LLM(usage={"total_tokens": 10})
        with self.assertRaises(SessionTokenLimitExceeded):
            pipeline._invoke_llm(llm, "x" * 300, _Budget(limit=settings.max_tokens))
        self.assertEqual(llm.invoked, 0)
        self.assertEqual(pipeline._api_request_count, 0, "global quota must not be consumed")

    def test_refunds_the_reservation_when_the_provider_fails(self):
        budget = _Budget(limit=100_000)
        with self.assertRaises(RuntimeError):
            pipeline._invoke_llm(_LLM(fail=True), "prompt", budget)
        self.assertEqual(budget.used, 0)

    def test_refunds_the_reservation_when_the_global_quota_is_exhausted(self):
        pipeline._api_request_count = settings.max_api_requests
        budget = _Budget(limit=100_000)
        with self.assertRaises(pipeline.APIRequestLimitExceeded):
            pipeline._invoke_llm(_LLM(), "prompt", budget)
        self.assertEqual(budget.used, 0)

    def test_estimates_usage_when_the_provider_reports_none(self):
        budget = _Budget(limit=100_000)
        pipeline._invoke_llm(_LLM(usage=None), "x" * 30, budget)
        self.assertEqual(budget.used, 10 + pipeline.estimate_prompt_tokens("ok [Source 1]"))


if __name__ == "__main__":
    unittest.main()
