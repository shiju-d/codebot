import asyncio
import httpx
import os
import traceback
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import TokenTextSplitter, CodeSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.core.llms import LLMMetadata
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import load_services, ServiceConfig
from message import parse_message
from jira import (
    parse_rca_input, extract_adf_text, build_rca_message,
    md_to_jira, fetch_jira_issue, post_jira_comment,
)
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

MAX_SESSIONS = 100

_reranker: FlagEmbeddingReranker | None = None


def _get_reranker() -> FlagEmbeddingReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=12)
    return _reranker


# { service_name: { "index": VectorStoreIndex, "system_prompt": str,
#                   "sessions": { "local": OrderedDict, "bedrock": OrderedDict } } }
services: dict = {}


def _get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        svc = services[service_name]
        sessions[session_id] = {
            "memory": memory,
            "engine": svc["index"].as_chat_engine(
                chat_mode="context",
                llm=llm,
                memory=memory,
                similarity_top_k=12,
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]


# Extension → tree-sitter language name.
# Extensions absent from this map fall back to _fallback_splitter.
_EXT_LANGUAGE: dict[str, str] = {
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".rb": "ruby",
    ".rake": "ruby",   # Rake DSL is valid Ruby
}

# Token-based splitter used for file types tree-sitter doesn't cover
# (.yml, .json, .erb, etc.) and as a fallback when parsing fails.
_fallback_splitter = TokenTextSplitter(
    chunk_size=600, chunk_overlap=100, separator="\n",
    backup_separators=["class ", "function ", "const ", "export ", "  "],
)

# CodeSplitter instances are expensive to create (they load grammars), so cache by language.
_code_splitters: dict[str, CodeSplitter] = {}


def _get_splitter(file_ext: str):
    lang = _EXT_LANGUAGE.get(file_ext.lower())
    if not lang:
        return _fallback_splitter
    if lang not in _code_splitters:
        _code_splitters[lang] = CodeSplitter(
            language=lang,
            chunk_lines=50,
            chunk_lines_overlap=15,
            max_chars=2000,
        )
    return _code_splitters[lang]


def _build_service_index(svc: ServiceConfig):
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    collection_name = f"{svc.name}_codebase"
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    if collection.count() > 0:
        print(f"[{svc.name}] Loading existing index ({collection.count()} chunks)...")
        return VectorStoreIndex.from_vector_store(vector_store)

    print(f"[{svc.name}] Building index from {len(svc.repos)} repo(s)...")
    all_nodes = []
    for repo_path in svc.repos:
        reader = SimpleDirectoryReader(
            input_dir=repo_path, recursive=True,
            required_exts=svc.file_extensions, exclude_hidden=True,
            exclude=[
                "**/node_modules/**", "**/dist/**", "**/.git/**",
                "**/log/**", "**/tmp/**",
                "**/__tests__/**", "**/*.spec.ts", "**/*.test.ts",
                "**/cypress/**", "**/e2e/**",
                "**/vendor/**", "**/coverage/**",
            ],
        )
        docs = reader.load_data()
        # Strip the /repos/ mount prefix so file paths in responses are repo-relative
        # e.g. /repos/ibe-admin/src/foo.ts → ibe-admin/src/foo.ts
        repos_prefix = os.path.dirname(repo_path).rstrip("/") + "/"
        for doc in docs:
            if "file_path" in doc.metadata:
                doc.metadata["file_path"] = doc.metadata["file_path"].removeprefix(repos_prefix)
            ext = os.path.splitext(doc.metadata.get("file_path", ""))[1]
            splitter = _get_splitter(ext)
            try:
                all_nodes.extend(splitter.get_nodes_from_documents([doc]))
            except Exception as exc:
                fp = doc.metadata.get("file_path", "?")
                print(f"[{svc.name}] CodeSplitter failed for {fp} ({exc}); falling back to token splitter")
                all_nodes.extend(_fallback_splitter.get_nodes_from_documents([doc]))

    index = VectorStoreIndex(all_nodes, storage_context=storage_context)
    print(f"[{svc.name}] Index built ({len(all_nodes)} chunks).")
    return index


def _init_all_services():
    global services
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        index = _build_service_index(svc)
        services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "jira_project_key": svc.jira_project_key,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    print(f"codebot ready. Services: {list(services.keys())}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_init_all_services)
    yield


app = FastAPI(title="codebot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class RcaRequest(BaseModel):
    input: str
    session_id: str = ""


async def _chat(request: ChatRequest, llm, llm_key: str):
    if not services:
        raise HTTPException(status_code=503, detail="RAG engine is initializing")

    try:
        service_name, message = parse_message(request.message)
    except ValueError:
        valid = ", ".join(services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Missing service prefix. Format: <service>: <message>. Valid services: {valid}",
        )

    if service_name not in services:
        valid = ", ".join(services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}. Format: <service>: <message>",
        )

    try:
        engine = _get_engine(request.session_id, service_name, llm, llm_key)
        response = await asyncio.to_thread(engine.chat, message)
        sources = list({
            node.metadata.get("file_path", "unknown")
            for node in response.source_nodes
        })
        return {"response": response.response, "sources": sources}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services")
def list_services():
    return {"services": list(services.keys())}


@app.get("/project/{project_key}")
def resolve_project(project_key: str):
    upper = project_key.upper()
    for name, svc in services.items():
        if svc.get("jira_project_key") == upper:
            return {"service": name, "project_key": upper}
    raise HTTPException(status_code=404, detail=f"No service mapped to Jira project '{project_key}'")


@app.post("/chat")
async def chat_local(request: ChatRequest):
    return await _chat(request, local_llm, "local")


@app.post("/chat/bedrock")
async def chat_bedrock(request: ChatRequest):
    if not bedrock_llm:
        raise HTTPException(status_code=503, detail="AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY not configured")
    return await _chat(request, bedrock_llm, "bedrock")


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    for svc in services.values():
        for llm_sessions in svc["sessions"].values():
            llm_sessions.pop(session_id, None)
    return {"cleared": session_id}


@app.post("/rca")
async def rca(request: RcaRequest):
    if not bedrock_llm:
        raise HTTPException(status_code=503, detail="AWS credentials not configured")
    if not all([JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN]):
        raise HTTPException(
            status_code=503,
            detail="Jira credentials not configured — set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN",
        )
    if not services:
        raise HTTPException(status_code=503, detail="RAG engine is initializing")

    try:
        service_name, issue_key, additional_context = parse_rca_input(request.input)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if service_name not in services:
        valid = ", ".join(services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}",
        )

    try:
        issue = await fetch_jira_issue(JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, issue_key)
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        detail = f"Jira API error fetching {issue_key}: {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else f"Jira connection error: {e}"
        raise HTTPException(status_code=502, detail=detail)

    summary = issue["fields"].get("summary", "(no summary)")
    desc_adf = issue["fields"].get("description")
    description = extract_adf_text(desc_adf).strip() if desc_adf else "No description provided."

    session_id = request.session_id or f"jira-{issue_key}"
    message = build_rca_message(service_name, issue_key, summary, description, additional_context)

    try:
        engine = _get_engine(session_id, service_name, bedrock_llm, "bedrock")
        response = await asyncio.to_thread(engine.chat, message)
        rca_text = response.response
        sources = list({
            node.metadata.get("file_path", "unknown")
            for node in response.source_nodes
        })
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    jira_body = "\n".join([
        "h2. \U0001f916 AI-Generated Root Cause Analysis",
        "",
        md_to_jira(rca_text),
        "",
        "----",
        "_Generated automatically by codebot. Please review before acting on it._",
    ])

    try:
        await post_jira_comment(JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, issue_key, jira_body)
    except (httpx.HTTPStatusError, httpx.RequestError) as e:
        detail = f"Failed to post Jira comment on {issue_key}: {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else f"Jira connection error posting comment: {e}"
        raise HTTPException(status_code=502, detail=detail)

    return {
        "response": rca_text,
        "sources": sources,
        "issue_key": issue_key,
        "comment_posted": True,
        "output": f"✅ RCA posted to {issue_key}.\n\n{rca_text}",
    }


@app.post("/reindex")
async def reindex_all():
    global services
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        try:
            chroma_client.delete_collection(f"{svc.name}_codebase")
        except Exception as e:
            print(f"[reindex] Warning: could not delete collection {svc.name}_codebase: {e}")

    new_services: dict = {}
    for svc in configs:
        index = await asyncio.to_thread(_build_service_index, svc)
        new_services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    services = new_services
    print(f"codebot reindexed. Services: {list(services.keys())}")
    return {"status": "reindexed", "services": list(services.keys())}


@app.post("/reindex/{service_name}")
async def reindex_service(service_name: str):
    configs = load_services(SERVICES_CONFIG_PATH)
    svc_config = next((s for s in configs if s.name == service_name), None)
    if not svc_config:
        valid = ", ".join(s.name for s in configs)
        raise HTTPException(
            status_code=404,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}",
        )

    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    try:
        chroma_client.delete_collection(f"{service_name}_codebase")
    except Exception as e:
        print(f"[reindex] Warning: could not delete collection {service_name}_codebase: {e}")

    index = await asyncio.to_thread(_build_service_index, svc_config)

    is_new = service_name not in services
    if not is_new:
        for llm_sessions in services[service_name]["sessions"].values():
            llm_sessions.clear()

    services[service_name] = {
        "index": index,
        "system_prompt": svc_config.system_prompt,
        "jira_project_key": svc_config.jira_project_key,
        "sessions": services.get(service_name, {}).get("sessions", {
            "local": OrderedDict(),
            "bedrock": OrderedDict(),
        }),
    }
    return {"status": "reindexed", "service": service_name}
