import unittest

from src.config import settings
from src.generation import pipeline


class GenerationQuotaTests(unittest.TestCase):
    def test_blocks_calls_after_the_configured_budget(self):
        original_count = pipeline._api_request_count
        pipeline._api_request_count = settings.max_api_requests - 1
        try:
            self.assertEqual(pipeline._invoke_llm(_SuccessfulLLM(), "prompt").content, "ok")
            with self.assertRaises(pipeline.APIRequestLimitExceeded):
                pipeline._invoke_llm(_SuccessfulLLM(), "prompt")
        finally:
            pipeline._api_request_count = original_count


class _SuccessfulLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "ok"})()


if __name__ == "__main__":
    unittest.main()