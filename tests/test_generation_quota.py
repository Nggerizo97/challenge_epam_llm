import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from botocore.exceptions import ClientError
from langchain_core.documents import Document

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

    @patch("src.generation.pipeline._get_dynamodb_client")
    def test_persistent_quota_uses_conditional_increment(self, get_client):
        now = datetime(2026, 8, 7, tzinfo=timezone.utc)

        pipeline._reserve_persistent_api_request("quota-table", 5, now)

        get_client.return_value.update_item.assert_called_once_with(
            TableName="quota-table",
            Key={"quota_key": {"S": "llm-calls#2026-08"}},
            UpdateExpression="ADD request_count :one SET expires_at = if_not_exists(expires_at, :expires_at)",
            ConditionExpression="attribute_not_exists(request_count) OR request_count < :limit",
            ExpressionAttributeValues={
                ":one": {"N": "1"},
                ":limit": {"N": "5"},
                ":expires_at": {"N": str(int((now + timedelta(days=40)).timestamp()))},
            },
        )

    @patch("src.generation.pipeline._get_dynamodb_client")
    def test_persistent_quota_rejects_an_exhausted_month(self, get_client):
        get_client.return_value.update_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException"}}, "UpdateItem"
        )

        with self.assertRaises(pipeline.APIRequestLimitExceeded):
            pipeline._reserve_persistent_api_request("quota-table", 5)

    def test_resolves_only_the_sources_cited_in_the_answer(self):
        context_docs = [
            Document(page_content="exports", metadata={"source": "exports.txt"}),
            Document(page_content="inflation", metadata={"source": "inflation.txt"}),
            Document(page_content="climate", metadata={"source": "climate.txt"}),
        ]

        citations = pipeline._resolve_citations("Coffee exports rose [Source 1].", context_docs)

        self.assertEqual(citations, ["exports.txt"])


class _SuccessfulLLM:
    def invoke(self, prompt):
        return type("Response", (), {"content": "ok"})()


if __name__ == "__main__":
    unittest.main()