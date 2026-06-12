# RAG Response Quality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve retrieval quality across all endpoints by replacing the single similarity search with multi-query fusion + cross-encoder reranking, giving the LLM a better-selected 12 chunks from ~30 candidates.

**Architecture:** `QueryFusionRetriever` wraps the existing ChromaDB retriever and generates 3 query variants in parallel; `FlagEmbeddingReranker` (BAAI/bge-reranker-base, CPU) scores all candidates and keeps the top 12. Both plug into `_get_engine` via `ContextChatEngine.from_defaults`. All three endpoints (`/chat`, `/chat/bedrock`, `/rca`) benefit automatically.

**Tech Stack:** LlamaIndex `QueryFusionRetriever` (built-in), `llama-index-postprocessor-flag-embedding-reranker`, `BAAI/bge-reranker-base` (~280 MB HuggingFace model).

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Add `llama-index-postprocessor-flag-embedding-reranker` |
| `docker-compose.yml` | Add `hf_cache` named volume; mount at `/root/.cache/huggingface` |
| `runner.py` | Add 3 imports; add `_reranker` singleton + `_get_reranker()`; refactor `_get_engine` |
| `tests/test_api.py` | Remove stale `test_chat_claude_not_configured`; add 2 new tests |

---

## Task 1: Add dependency and HuggingFace cache volume

**Files:**
- Modify: `requirements.txt`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add the reranker package to requirements.txt**

Open `requirements.txt`. It currently ends with:
```
httpx
```

Add one line after `httpx`:
```
llama-index-postprocessor-flag-embedding-reranker
```

Full file after change:
```
llama-index
tree-sitter-languages
llama-index-llms-ollama
llama-index-llms-anthropic
llama-index-llms-bedrock-converse
llama-index-embeddings-ollama
llama-index-vector-stores-chroma
chromadb
fastapi
uvicorn
pyyaml
pytest
httpx
llama-index-postprocessor-flag-embedding-reranker
```

- [ ] **Step 2: Add the HuggingFace cache volume to docker-compose.yml**

Open `docker-compose.yml`. It currently reads:
```yaml
services:
  codebot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./services.yaml:/app/services.yaml:ro
      - chroma_data:/app/chroma_db
      - ${REPO_ROOT}/ibe-api:/repos/ibe-api:ro
      - ${REPO_ROOT}/ibe-frontend:/repos/ibe-frontend:ro
      - ${REPO_ROOT}/ibe-admin:/repos/ibe-admin:ro
      - ${REPO_ROOT}/rover-ifc:/repos/rover-ifc:ro
      - ${REPO_ROOT}/pms:/repos/pms:ro
    env_file: .env
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  chroma_data:
```

Replace with:
```yaml
services:
  codebot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./services.yaml:/app/services.yaml:ro
      - chroma_data:/app/chroma_db
      - hf_cache:/root/.cache/huggingface
      - ${REPO_ROOT}/ibe-api:/repos/ibe-api:ro
      - ${REPO_ROOT}/ibe-frontend:/repos/ibe-frontend:ro
      - ${REPO_ROOT}/ibe-admin:/repos/ibe-admin:ro
      - ${REPO_ROOT}/rover-ifc:/repos/rover-ifc:ro
      - ${REPO_ROOT}/pms:/repos/pms:ro
    env_file: .env
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  chroma_data:
  hf_cache:
```

- [ ] **Step 3: Commit**

```bash
git add requirements.txt docker-compose.yml
git commit -m "feat: add flag-embedding-reranker dep and hf_cache volume"
```

---

## Task 2: Add `_get_reranker()` singleton to runner.py

**Files:**
- Modify: `runner.py` — add import + singleton below the `MAX_SESSIONS` constant
- Modify: `tests/test_api.py` — remove stale test; add singleton test

### Step 2a: Remove stale test

`tests/test_api.py` contains `test_chat_claude_not_configured` (line 52) which references `runner.claude_llm` — an attribute that no longer exists in `runner.py`. This test will fail with `AttributeError` and must be removed before adding new tests.

- [ ] **Step 1: Remove the stale test**

Open `tests/test_api.py`. Delete the entire `test_chat_claude_not_configured` function (lines 52–60):

```python
def test_chat_claude_not_configured(client):
    original = runner.claude_llm
    runner.claude_llm = None
    try:
        response = client.post("/chat/claude", json={"message": "ibe: hello", "session_id": "t"})
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    finally:
        runner.claude_llm = original
```

- [ ] **Step 2: Verify existing tests pass without the stale test**

```bash
cd /Users/shijudevarajan/Codebase/poc/codebot
python -m pytest tests/test_api.py -v
```

Expected: all remaining tests pass. If any test fails for a reason unrelated to this task, stop and investigate before continuing.

### Step 2b: Write the failing singleton test

- [ ] **Step 3: Write the failing test**

Add to the bottom of `tests/test_api.py`:

```python
def test_get_reranker_returns_singleton():
    runner._reranker = None  # ensure clean state
    with patch("runner.FlagEmbeddingReranker") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        first = runner._get_reranker()
        second = runner._get_reranker()
    assert first is second
    mock_cls.assert_called_once_with(model="BAAI/bge-reranker-base", top_n=12)
    runner._reranker = None  # restore clean state
```

- [ ] **Step 4: Run test to verify it fails**

```bash
python -m pytest tests/test_api.py::test_get_reranker_returns_singleton -v
```

Expected: `FAILED` — `AttributeError: module 'runner' has no attribute '_reranker'`

### Step 2c: Implement the singleton

- [ ] **Step 5: Add the import for `FlagEmbeddingReranker` to runner.py**

Open `runner.py`. The imports currently end at line 24:
```python
from jira import (
    parse_rca_input, extract_adf_text, build_rca_message,
    md_to_jira, fetch_jira_issue, post_jira_comment,
)
```

Add one import line after the jira imports:
```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
```

- [ ] **Step 6: Add the `_reranker` singleton below `MAX_SESSIONS`**

In `runner.py`, find the block after `Settings.prompt_helper`:
```python
MAX_SESSIONS = 100

# { service_name: { "index": VectorStoreIndex, ...
services: dict = {}
```

Insert after `MAX_SESSIONS = 100`:
```python
_reranker: FlagEmbeddingReranker | None = None


def _get_reranker() -> FlagEmbeddingReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=12)
    return _reranker
```

- [ ] **Step 7: Run the test to verify it passes**

```bash
python -m pytest tests/test_api.py::test_get_reranker_returns_singleton -v
```

Expected: `PASSED`

- [ ] **Step 8: Run the full test suite to verify nothing broke**

```bash
python -m pytest tests/test_api.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add runner.py tests/test_api.py
git commit -m "feat: add _get_reranker() singleton for FlagEmbeddingReranker"
```

---

## Task 3: Refactor `_get_engine` to use `QueryFusionRetriever` + `ContextChatEngine`

**Files:**
- Modify: `runner.py` — add 3 imports; replace body of `_get_engine`
- Modify: `tests/test_api.py` — add engine wiring test

### Step 3a: Write the failing test first

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/test_api.py`:

```python
def test_get_engine_wires_fusion_retriever_and_reranker(client):
    mock_reranker = MagicMock()
    mock_base_retriever = MagicMock()
    mock_fusion_retriever = MagicMock()
    mock_engine = MagicMock()

    runner.services["ibe"]["index"].as_retriever.return_value = mock_base_retriever

    with patch("runner._get_reranker", return_value=mock_reranker) as mock_gr, \
         patch("runner.QueryFusionRetriever", return_value=mock_fusion_retriever) as mock_qfr, \
         patch("runner.ContextChatEngine") as mock_cce:
        mock_cce.from_defaults.return_value = mock_engine

        engine = runner._get_engine("test-fusion-session", "ibe", runner.local_llm, "local")

    runner.services["ibe"]["index"].as_retriever.assert_called_once_with(similarity_top_k=10)
    mock_qfr.assert_called_once()
    qfr_kwargs = mock_qfr.call_args.kwargs
    assert qfr_kwargs["num_queries"] == 3
    assert qfr_kwargs["use_async"] is True
    assert qfr_kwargs["retrievers"] == [mock_base_retriever]
    mock_cce.from_defaults.assert_called_once()
    cce_kwargs = mock_cce.from_defaults.call_args.kwargs
    assert cce_kwargs["retriever"] is mock_fusion_retriever
    assert mock_reranker in cce_kwargs["node_postprocessors"]
    assert engine is mock_engine

    # cleanup so this session doesn't pollute other tests
    runner.services["ibe"]["sessions"]["local"].pop("test-fusion-session", None)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_api.py::test_get_engine_wires_fusion_retriever_and_reranker -v
```

Expected: `FAILED` — `AttributeError: module 'runner' has no attribute 'QueryFusionRetriever'`

### Step 3b: Add imports

- [ ] **Step 3: Add the three new imports to runner.py**

In `runner.py`, after the `FlagEmbeddingReranker` import added in Task 2:
```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
```

Add three more lines immediately after it:
```python
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.chat_engine import ContextChatEngine
```

### Step 3c: Replace `_get_engine`

- [ ] **Step 4: Replace the body of `_get_engine`**

In `runner.py`, find the current `_get_engine` function (lines 92–109):

```python
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
```

Replace it entirely with:

```python
def _get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        svc = services[service_name]
        base_retriever = svc["index"].as_retriever(similarity_top_k=10)
        fusion_retriever = QueryFusionRetriever(
            retrievers=[base_retriever],
            llm=llm,
            num_queries=3,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
        )
        sessions[session_id] = {
            "memory": memory,
            "engine": ContextChatEngine.from_defaults(
                retriever=fusion_retriever,
                llm=llm,
                memory=memory,
                node_postprocessors=[_get_reranker()],
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]
```

- [ ] **Step 5: Run the new test to verify it passes**

```bash
python -m pytest tests/test_api.py::test_get_engine_wires_fusion_retriever_and_reranker -v
```

Expected: `PASSED`

- [ ] **Step 6: Run the full test suite**

```bash
python -m pytest tests/test_api.py -v
```

Expected: all tests pass. The `test_chat_valid_service_returns_response` test patches `runner._get_engine` entirely, so it is unaffected by the internal changes.

- [ ] **Step 7: Commit**

```bash
git add runner.py tests/test_api.py
git commit -m "feat: multi-query fusion + cross-encoder reranking in _get_engine"
```

---

## Verification

After all tasks are committed, rebuild the container and test end-to-end:

```bash
docker compose up --build -d
```

Watch startup logs for the model download on first request (one-time, ~10 s on a good connection, cached in `hf_cache` volume thereafter):
```
Downloading shards: 100%|████████████████| 1/1 [00:08<00:00]
```

Smoke test — the sources list should contain more precisely relevant files than before:
```bash
curl -s -X POST http://localhost:8000/chat/bedrock \
  -H "Content-Type: application/json" \
  -d '{"message": "ibe: why is checkout failing?", "session_id": "quality-test"}' \
  | python3 -m json.tool
```

Expected: `sources` contains files directly related to checkout flow, not unrelated utility files.
