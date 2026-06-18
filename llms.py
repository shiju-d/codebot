import os
import threading

from llama_index.core import Settings
from llama_index.core.llms import LLMMetadata
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-5-20250929-v1:0")
SERVICES_CONFIG_PATH = os.getenv("SERVICES_CONFIG_PATH", "/app/services.yaml")
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "codebot-secret")
ELASTIC_URL = os.getenv("ELASTIC_URL", "")
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY", "")
ELASTIC_INDEX = os.getenv("ELASTIC_INDEX", "logs-*")

local_llm = Ollama(base_url=OLLAMA_BASE_URL, model="qwen2.5-coder:7b", request_timeout=120.0)

# Patch BedrockConverse.metadata at the class level so every internal LlamaIndex
# code path (including response synthesizers) sees the correct 200k context window.
# Without this, unrecognised Bedrock model IDs default to ~4k context, making
# num_output=4096 produce a negative chunk size: 4091 - 4096 = -5.
_BEDROCK_MAX_TOKENS = 4096
try:
    _orig_bedrock_metadata = BedrockConverse.metadata.fget
    def _patched_bedrock_metadata(self) -> LLMMetadata:
        m = _orig_bedrock_metadata(self)
        return LLMMetadata(
            context_window=200000,
            # Keep num_output modest so the response synthesizer reserves less
            # budget for output and leaves more room for retrieved context chunks.
            # Actual generation length is controlled by max_tokens on the constructor.
            num_output=2048,
            is_chat_model=m.is_chat_model,
            is_function_calling_model=m.is_function_calling_model,
            model_name=m.model_name,
        )
    BedrockConverse.metadata = property(_patched_bedrock_metadata)
    print("[startup] BedrockConverse.metadata patched: context_window=200000")
except Exception as exc:
    # Patch failed (metadata is not a plain property in this LlamaIndex build).
    # Fall back: cap max_tokens below the model's default context_window (~4091)
    # so chunk_size = context_window - max_tokens stays positive.
    print(f"[startup] BedrockConverse.metadata patch failed ({exc}). Capping max_tokens=2048.")
    _BEDROCK_MAX_TOKENS = 2048

bedrock_llm = BedrockConverse(
    model=BEDROCK_MODEL_ID,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
    max_tokens=_BEDROCK_MAX_TOKENS,
) if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY else None

Settings.embed_model = OllamaEmbedding(base_url=OLLAMA_BASE_URL, model_name="mxbai-embed-large")

# Set a large PromptHelper globally so all response synthesizers get the full
# 200k context window. BedrockConverse reports context_window=4091 for any
# model ID it doesn't recognise (e.g. global.anthropic.* inference profiles),
# which makes long RCA prompts produce a negative available_context_size in
# CompactAndRefine. Setting Settings.prompt_helper here overrides the LLM
# metadata lookup that get_response_synthesizer falls back to.
Settings.prompt_helper = PromptHelper(context_window=200000, num_output=2048)
print("[startup] Settings.prompt_helper: context_window=200000, num_output=2048")

neo4j_driver = None
try:
    import neo4j as _neo4j_lib
    neo4j_driver = _neo4j_lib.GraphDatabase.driver(
        NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD)
    )
    neo4j_driver.verify_connectivity()
    print(f"[startup] Neo4j connected: {NEO4J_URI}")
except Exception as _neo4j_exc:
    print(f"[startup] Neo4j not available ({_neo4j_exc}); graph expansion disabled")
    neo4j_driver = None

_reranker: FlagEmbeddingReranker | None = None
_reranker_lock = threading.Lock()


def get_reranker() -> FlagEmbeddingReranker:
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=12)
    return _reranker
