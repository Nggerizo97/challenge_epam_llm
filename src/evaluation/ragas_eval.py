import sys
import types
import logging

# Compatibility shim for older Ragas versions referencing deprecated langchain_community VertexAI path
try:
    import langchain_community.chat_models.vertexai
except (ImportError, ModuleNotFoundError):
    try:
        import langchain_google_vertexai
        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = langchain_google_vertexai.ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module
    except Exception:
        pass

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.generation.pipeline import get_bedrock_llm

logger = logging.getLogger("RagasEval")


def _get_eval_llm():
    """Get verified project LLM (AWS Bedrock or Groq API fallback) for Ragas evaluation."""
    if settings.aws_access_key_id and settings.aws_secret_access_key:
        try:
            logger.info(f"Testing AWS Bedrock LLM for Ragas evaluation: {settings.bedrock_model_id}")
            bedrock_llm = get_bedrock_llm(settings.bedrock_model_id)
            # Probe invocation to ensure AWS model access is granted
            bedrock_llm.invoke("ping")
            logger.info("AWS Bedrock LLM verified successfully.")
            return bedrock_llm
        except Exception as e:
            logger.warning(f"AWS Bedrock LLM unavailable ({e}). Falling back to Groq API...")

    if settings.groq_api_key:
        logger.info(f"Initializing Groq API LLM for Ragas evaluation: {settings.groq_model}")
        from langchain_groq import ChatGroq
        return ChatGroq(model=settings.groq_model, api_key=settings.groq_api_key, temperature=0.0)

    raise ValueError("No valid active LLM credentials found (AWS Bedrock or GROQ_API_KEY required).")



def run_ragas_evaluation(samples: list[dict]) -> dict:
    dataset = Dataset.from_list(samples)
    eval_llm = _get_eval_llm()
    eval_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    logger.info("Starting Ragas evaluation metric scoring...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )
    return results.to_pandas().mean(numeric_only=True).to_dict()
