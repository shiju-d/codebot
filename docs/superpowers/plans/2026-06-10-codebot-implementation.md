# codebot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the ibex RAG API into a configurable multi-service tool where each service groups one or more repos into its own ChromaDB index, selected via a `<service>:` message prefix.

**Architecture:** A `services.yaml` config defines named services, each with repo paths and a system prompt. On startup, codebot loads all services and builds/loads their ChromaDB collections. Chat requests carry a `<service>: message` prefix; codebot routes to the right index and system prompt. The n8n workflow additionally handles an optional `<llm>|` prefix to route to the right endpoint.

**Tech Stack:** Python 3.11, FastAPI, LlamaIndex, ChromaDB, PyYAML, pytest, httpx

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | Create | Load and parse `services.yaml` into `ServiceConfig` dataclasses |
| `message.py` | Create | Pure `parse_message()` — splits `service: message` prefix |
| `runner.py` | Modify | Service registry, indexing, all FastAPI endpoints |
| `services.yaml` | Create | Sample config with IBE service |
| `requirements.txt` | Modify | Add `pyyaml`, `pytest`, `httpx` |
| `Dockerfile` | Modify | Copy `config.py` and `message.py` into image |
| `docker-compose.yml` | Modify | Rename service to `codebot`, add `services.yaml` volume, remove hardcoded IBE path |
| `codebot.json` | Create | n8n workflow with `<llm>\|<service>: <message>` prefix parsing |
| `README.md` | Modify | Rewrite for codebot |
| `tests/__init__.py` | Create | Makes `tests/` a package |
| `tests/test_config.py` | Create | Tests for `load_services()` |
| `tests/test_message.py` | Create | Tests for `parse_message()` |
| `tests/test_api.py` | Create | Tests for API error responses |

---

### Task 1: Add dependencies and test infrastructure

**Files:**
- Modify: `requirements.txt`
- Create: `tests/__init__.py`

- [ ] **Step 1: Add pyyaml, pytest, httpx to requirements.txt**

Replace `requirements.txt` entirely:

```
llama-index
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
```

- [ ] **Step 2: Install new deps locally**

Run: `pip install pyyaml pytest httpx`

Expected: `Successfully installed ...` (no errors)

- [ ] **Step 3: Create tests directory**

Run: `mkdir -p tests && touch tests/__init__.py`

Expected: `tests/__init__.py` exists

- [ ] **Step 4: Commit**

```bash
git add requirements.txt tests/__init__.py
git commit -m "chore: add pyyaml, pytest, httpx and tests dir"
```

---

### Task 2: Config loading module (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `config.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import os
import tempfile
import pytest
import yaml
from config import load_services, ServiceConfig


def _write_yaml(data):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.close()
    return f.name


def test_load_single_service():
    path = _write_yaml({"services": [
        {"name": "ibe", "system_prompt": "IBE prompt.", "repos": ["/repos/ibe-api", "/repos/ibe-frontend"]}
    ]})
    try:
        result = load_services(path)
        assert len(result) == 1
        assert isinstance(result[0], ServiceConfig)
        assert result[0].name == "ibe"
        assert result[0].system_prompt == "IBE prompt."
        assert result[0].repos == ["/repos/ibe-api", "/repos/ibe-frontend"]
    finally:
        os.unlink(path)


def test_load_multiple_services():
    path = _write_yaml({"services": [
        {"name": "ibe", "system_prompt": "IBE prompt.", "repos": ["/repos/ibe-api"]},
        {"name": "pms", "system_prompt": "PMS prompt.", "repos": ["/repos/pms-api", "/repos/pms-frontend"]},
    ]})
    try:
        result = load_services(path)
        assert len(result) == 2
        assert result[0].name == "ibe"
        assert result[1].name == "pms"
        assert len(result[1].repos) == 2
    finally:
        os.unlink(path)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_services("/nonexistent/path/services.yaml")


def test_system_prompt_preserved_verbatim():
    path = _write_yaml({"services": [
        {"name": "test", "system_prompt": "  prompt with spaces  ", "repos": ["/r/a"]}
    ]})
    try:
        result = load_services(path)
        assert result[0].system_prompt == "  prompt with spaces  "
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config.py -v`

Expected: `ModuleNotFoundError: No module named 'config'`

- [ ] **Step 3: Implement config.py**

Create `config.py`:

```python
import yaml
from dataclasses import dataclass


@dataclass
class ServiceConfig:
    name: str
    system_prompt: str
    repos: list


def load_services(path: str) -> list:
    with open(path) as f:
        data = yaml.safe_load(f)
    return [
        ServiceConfig(
            name=svc["name"],
            system_prompt=svc["system_prompt"],
            repos=svc["repos"],
        )
        for svc in data["services"]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`

Expected: 4 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat: add config.py with ServiceConfig and load_services"
```

---

### Task 3: Message parsing module (TDD)

**Files:**
- Create: `tests/test_message.py`
- Create: `message.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_message.py`:

```python
import pytest
from message import parse_message


def test_parse_basic_prefix():
    service, msg = parse_message("ibe: why is checkout failing?")
    assert service == "ibe"
    assert msg == "why is checkout failing?"


def test_parse_strips_whitespace():
    service, msg = parse_message("  ibe  :  why is checkout failing?  ")
    assert service == "ibe"
    assert msg == "why is checkout failing?"


def test_parse_message_containing_colon():
    service, msg = parse_message("ibe: error: something went wrong")
    assert service == "ibe"
    assert msg == "error: something went wrong"


def test_parse_missing_colon_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("why is checkout failing?")


def test_parse_empty_service_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message(": why is checkout failing?")


def test_parse_empty_message_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("ibe:")


def test_parse_empty_string_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_message.py -v`

Expected: `ModuleNotFoundError: No module named 'message'`

- [ ] **Step 3: Implement message.py**

Create `message.py`:

```python
def parse_message(raw: str) -> tuple:
    """Parse 'service: message' into (service_name, message). Raises ValueError if prefix missing."""
    if ":" not in raw:
        raise ValueError("missing_prefix")
    service_name, _, message = raw.partition(":")
    service_name = service_name.strip()
    message = message.strip()
    if not service_name or not message:
        raise ValueError("missing_prefix")
    return service_name, message
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_message.py -v`

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add message.py tests/test_message.py
git commit -m "feat: add message.py with parse_message"
```

---

### Task 4: Write failing API tests

**Files:**
- Create: `tests/test_api.py`

- [ ] **Step 1: Write the API tests**

Create `tests/test_api.py`:

```python
from collections import OrderedDict
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
import runner


def _fresh_mock_services():
    return {
        "ibe": {
            "index": MagicMock(),
            "system_prompt": "You are IBE expert.",
            "sessions": {
                "local": OrderedDict(),
                "claude": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    }


@pytest.fixture
def client():
    runner.services = _fresh_mock_services()
    with patch("runner._init_all_services"):
        with TestClient(runner.app) as c:
            yield c


def test_list_services(client):
    response = client.get("/services")
    assert response.status_code == 200
    assert response.json() == {"services": ["ibe"]}


def test_chat_missing_prefix(client):
    response = client.post("/chat", json={"message": "why is checkout failing?", "session_id": "t"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Missing service prefix" in detail
    assert "ibe" in detail


def test_chat_unknown_service(client):
    response = client.post("/chat", json={"message": "xyz: hello", "session_id": "t"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unknown service 'xyz'" in detail
    assert "ibe" in detail


def test_chat_claude_not_configured(client):
    original = runner.claude_llm
    runner.claude_llm = None
    try:
        response = client.post("/chat/claude", json={"message": "ibe: hello", "session_id": "t"})
        assert response.status_code == 503
        assert "ANTHROPIC_API_KEY" in response.json()["detail"]
    finally:
        runner.claude_llm = original


def test_chat_bedrock_not_configured(client):
    original = runner.bedrock_llm
    runner.bedrock_llm = None
    try:
        response = client.post("/chat/bedrock", json={"message": "ibe: hello", "session_id": "t"})
        assert response.status_code == 503
        assert "AWS_ACCESS_KEY_ID" in response.json()["detail"]
    finally:
        runner.bedrock_llm = original


def test_reindex_unknown_service(client):
    response = client.post("/reindex/xyz")
    assert response.status_code == 400
    assert "Unknown service 'xyz'" in response.json()["detail"]


def test_clear_session(client):
    runner.services["ibe"]["sessions"]["local"]["my-session"] = {"memory": None, "engine": None}
    response = client.delete("/session/my-session")
    assert response.status_code == 200
    assert response.json() == {"cleared": "my-session"}
    assert "my-session" not in runner.services["ibe"]["sessions"]["local"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_api.py -v`

Expected: failures like `AttributeError: module 'runner' has no attribute 'services'` — old runner.py does not have the new structure yet.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_api.py
git commit -m "test: add failing API tests for multi-service runner"
```

---

### Task 5: Refactor runner.py

**Files:**
- Modify: `runner.py`

- [ ] **Step 1: Replace runner.py with the multi-service implementation**

Overwrite `runner.py` entirely:

```python
import asyncio
import os
from collections import OrderedDict
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import load_services, ServiceConfig
from message import parse_message

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
SERVICES_CONFIG_PATH = os.getenv("SERVICES_CONFIG_PATH", "/app/services.yaml")

local_llm = Ollama(base_url=OLLAMA_BASE_URL, model="qwen2.5-coder:7b", request_timeout=120.0)
claude_llm = Anthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
bedrock_llm = BedrockConverse(
    model=BEDROCK_MODEL_ID,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION,
) if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY else None

Settings.embed_model = OllamaEmbedding(base_url=OLLAMA_BASE_URL, model_name="mxbai-embed-large")

MAX_SESSIONS = 100

# { service_name: { "index": VectorStoreIndex, "system_prompt": str,
#                   "sessions": { "local": OrderedDict, "claude": OrderedDict, "bedrock": OrderedDict } } }
services: dict = {}


def _get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=2048)
        svc = services[service_name]
        sessions[session_id] = {
            "memory": memory,
            "engine": svc["index"].as_chat_engine(
                chat_mode="condense_plus_context",
                llm=llm,
                memory=memory,
                similarity_top_k=8,
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]


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
    splitter = TokenTextSplitter(
        chunk_size=600, chunk_overlap=100, separator="\n",
        backup_separators=["class ", "function ", "const ", "export ", "  "],
    )
    all_nodes = []
    for repo_path in svc.repos:
        reader = SimpleDirectoryReader(
            input_dir=repo_path, recursive=True,
            required_exts=[".js", ".jsx", ".ts", ".tsx"], exclude_hidden=True,
            exclude=[
                "**/node_modules/**", "**/dist/**", "**/.git/**",
                "**/log/**", "**/tmp/**",
                "**/__tests__/**", "**/*.spec.ts", "**/*.test.ts",
                "**/cypress/**", "**/e2e/**",
            ],
        )
        all_nodes.extend(splitter.get_nodes_from_documents(reader.load_data()))

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
            "sessions": {
                "local": OrderedDict(),
                "claude": OrderedDict(),
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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/services")
def list_services():
    return {"services": list(services.keys())}


@app.post("/chat")
async def chat_local(request: ChatRequest):
    return await _chat(request, local_llm, "local")


@app.post("/chat/claude")
async def chat_claude(request: ChatRequest):
    if not claude_llm:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")
    return await _chat(request, claude_llm, "claude")


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


@app.post("/reindex")
async def reindex_all():
    global services
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        try:
            chroma_client.delete_collection(f"{svc.name}_codebase")
        except Exception:
            pass
    services = {}
    await asyncio.to_thread(_init_all_services)
    return {"status": "reindexed", "services": list(services.keys())}


@app.post("/reindex/{service_name}")
async def reindex_service(service_name: str):
    if service_name not in services:
        valid = ", ".join(services.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unknown service '{service_name}'. Valid services: {valid}",
        )
    configs = load_services(SERVICES_CONFIG_PATH)
    svc_config = next((s for s in configs if s.name == service_name), None)
    if not svc_config:
        raise HTTPException(status_code=404, detail=f"Service '{service_name}' not found in config")

    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    try:
        chroma_client.delete_collection(f"{service_name}_codebase")
    except Exception:
        pass

    for llm_sessions in services[service_name]["sessions"].values():
        llm_sessions.clear()

    index = await asyncio.to_thread(_build_service_index, svc_config)
    services[service_name]["index"] = index
    return {"status": "reindexed", "service": service_name}
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`

Expected: all 15 tests PASSED

- [ ] **Step 3: Commit**

```bash
git add runner.py
git commit -m "feat: refactor runner.py into multi-service RAG engine"
```

---

### Task 6: Create services.yaml

**Files:**
- Create: `services.yaml`

- [ ] **Step 1: Create services.yaml with IBE service**

Create `services.yaml`:

```yaml
services:
  - name: ibe
    system_prompt: |
      You are an expert software engineer specialising in the Stayntouch IBE application.
      The codebase has three layers:
      - ibe-api: LoopBack 4 REST API (TypeScript) — controllers, services, repositories, models
      - ibe-frontend: Express + Jade server-rendered app (JavaScript) — controllers, services, Vue components
      - ibe-admin: Angular 19 admin dashboard (TypeScript) — feature modules, services, components

      When analysing bugs:
      1. Identify the affected layer (controller / service / repository / model)
      2. Trace the call chain across files
      3. Point to the exact file and function where the bug likely originates
      4. Suggest a fix with a code snippet
    repos:
      - /repos/ibe-api
      - /repos/ibe-frontend
      - /repos/ibe-admin
```

- [ ] **Step 2: Commit**

```bash
git add services.yaml
git commit -m "feat: add services.yaml with IBE service config"
```

---

### Task 7: Update Dockerfile and docker-compose.yml

**Files:**
- Modify: `Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Update Dockerfile to copy all Python modules**

Replace `Dockerfile` entirely:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /lib/apt/lists/*
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY config.py message.py runner.py ./

EXPOSE 8000

CMD ["uvicorn", "runner:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Update docker-compose.yml**

Replace `docker-compose.yml` entirely:

```yaml
services:
  codebot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./services.yaml:/app/services.yaml:ro
      - chroma_data:/app/chroma_db
      # Add one line per repo listed in services.yaml:
      # - /absolute/path/to/ibe-api:/repos/ibe-api:ro
      # - /absolute/path/to/ibe-frontend:/repos/ibe-frontend:ro
      # - /absolute/path/to/ibe-admin:/repos/ibe-admin:ro
    env_file: .env
    extra_hosts:
      - "host.docker.internal:host-gateway"

volumes:
  chroma_data:
```

- [ ] **Step 3: Commit**

```bash
git add Dockerfile docker-compose.yml
git commit -m "feat: update Dockerfile and docker-compose.yml for codebot"
```

---

### Task 8: Create codebot.json (n8n workflow)

**Files:**
- Create: `codebot.json`

- [ ] **Step 1: Create codebot.json**

Create `codebot.json`:

```json
{
  "name": "codebot",
  "nodes": [
    {
      "parameters": {
        "options": {
          "allowFileUploads": false
        },
        "placeholder": "e.g. claude|ibe: why is checkout failing?"
      },
      "type": "@n8n/n8n-nodes-langchain.chatTrigger",
      "typeVersion": 1.4,
      "position": [0, 0],
      "id": "a1b2c3d4-e5f6-7890-abcd-100000000001",
      "name": "When chat message received",
      "webhookId": "f1e2d3c4-b5a6-7890-abcd-200000000001"
    },
    {
      "parameters": {
        "content": "## codebot\n\n**Format:** `<llm>|<service>: <message>`\n\n**LLMs:** `local` | `claude` | `bedrock`\n\n**Services:** see `GET /services`\n\n**Examples:**\n```\nclaude|ibe: why is checkout failing?\nbedrock|pms: trace the check-in flow\nibe: where is the cart service?\n```\n\nOmitting `<llm>|` defaults to `local`.",
        "height": 280,
        "width": 340
      },
      "type": "n8n-nodes-base.stickyNote",
      "typeVersion": 1,
      "position": [-380, -60],
      "id": "a1b2c3d4-e5f6-7890-abcd-100000000005",
      "name": "Usage Guide"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "asg-001",
              "name": "llm",
              "value": "={{ /^(local|claude|bedrock)\\|/i.test($json.chatInput) ? $json.chatInput.match(/^(local|claude|bedrock)\\|/i)[1].toLowerCase() : 'local' }}",
              "type": "string"
            },
            {
              "id": "asg-002",
              "name": "message",
              "value": "={{ /^(local|claude|bedrock)\\|/i.test($json.chatInput) ? $json.chatInput.replace(/^(?:local|claude|bedrock)\\|/i, '') : $json.chatInput }}",
              "type": "string"
            },
            {
              "id": "asg-003",
              "name": "session_id",
              "value": "={{ $json.sessionId || 'default' }}",
              "type": "string"
            },
            {
              "id": "asg-004",
              "name": "url",
              "value": "={{ /^claude\\|/i.test($json.chatInput) ? 'http://host.docker.internal:8000/chat/claude' : /^bedrock\\|/i.test($json.chatInput) ? 'http://host.docker.internal:8000/chat/bedrock' : 'http://host.docker.internal:8000/chat' }}",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [240, 0],
      "id": "a1b2c3d4-e5f6-7890-abcd-100000000002",
      "name": "Parse Input"
    },
    {
      "parameters": {
        "method": "POST",
        "url": "={{ $json.url }}",
        "sendBody": true,
        "specifyBody": "keypair",
        "bodyParameters": {
          "parameters": [
            {
              "name": "message",
              "value": "={{ $json.message }}"
            },
            {
              "name": "session_id",
              "value": "={{ $json.session_id }}"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.4,
      "position": [480, 0],
      "id": "a1b2c3d4-e5f6-7890-abcd-100000000003",
      "name": "RAG Request"
    },
    {
      "parameters": {
        "assignments": {
          "assignments": [
            {
              "id": "asg-005",
              "name": "output",
              "value": "={{ $json.response }}",
              "type": "string"
            },
            {
              "id": "asg-006",
              "name": "sources",
              "value": "={{ $json.sources.join('\\n') }}",
              "type": "string"
            }
          ]
        },
        "options": {}
      },
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.4,
      "position": [720, 0],
      "id": "a1b2c3d4-e5f6-7890-abcd-100000000004",
      "name": "Format Response"
    }
  ],
  "pinData": {},
  "connections": {
    "When chat message received": {
      "main": [[{"node": "Parse Input", "type": "main", "index": 0}]]
    },
    "Parse Input": {
      "main": [[{"node": "RAG Request", "type": "main", "index": 0}]]
    },
    "RAG Request": {
      "main": [[{"node": "Format Response", "type": "main", "index": 0}]]
    }
  },
  "active": false,
  "settings": {"executionOrder": "v1"},
  "meta": {
    "instanceId": "300d955eef7411856c235fa3cfdcb547fe1d57154df1136a89ade29ebe71d75d"
  },
  "nodeGroups": [],
  "tags": []
}
```

- [ ] **Step 2: Commit**

```bash
git add codebot.json
git commit -m "feat: add codebot n8n workflow with llm|service prefix parsing"
```

---

### Task 9: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace README.md**

Overwrite `README.md` entirely:

```markdown
# codebot — Stayntouch Code Intelligence

> RAG-powered code assistant for Stayntouch repositories. Configure any set of repos as a named service, then ask questions across them using a simple chat prefix.

## How It Works

```
services.yaml  →  defines named services (groups of repos + system prompt)
repos (volumes) → mounted read-only into the container
ChromaDB        → one collection per service, persisted on disk

POST /chat  "ibe: why is checkout failing?"  →  searches ibe index  →  answer + sources
```

On first run, codebot indexes all configured repos (1–3 min per service). Subsequent starts load from disk in ~2 seconds.

---

## Services Config (`services.yaml`)

```yaml
services:
  - name: ibe
    system_prompt: |
      You are an expert in the Stayntouch IBE application.
      IBE has three layers: ibe-api (LoopBack 4), ibe-frontend (Express/Jade), ibe-admin (Angular 19).
      When analysing bugs: identify the layer, trace the call chain, point to the exact file, suggest a fix.
    repos:
      - /repos/ibe-api
      - /repos/ibe-frontend
      - /repos/ibe-admin

  - name: pms
    system_prompt: |
      You are an expert in the Stayntouch PMS application.
      ...
    repos:
      - /repos/pms-api
```

- **`name`** — used as the message prefix (`ibe:`, `pms:`)
- **`system_prompt`** — context given to the LLM for this service
- **`repos`** — container paths of repos to index (must match volume mounts below)

---

## Setup

### 1. Mount repos in `docker-compose.yml`

Add one volume mount per repo listed in `services.yaml`:

```yaml
volumes:
  - ./services.yaml:/app/services.yaml:ro
  - chroma_data:/app/chroma_db
  - /absolute/path/to/ibe-api:/repos/ibe-api:ro
  - /absolute/path/to/ibe-frontend:/repos/ibe-frontend:ro
  - /absolute/path/to/ibe-admin:/repos/ibe-admin:ro
```

### 2. Configure `.env`

```
# Required for Claude endpoint
ANTHROPIC_API_KEY=sk-ant-...

# Required for Bedrock endpoint
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0
```

### 3. Pull Ollama models (one-time, on host machine)

```bash
ollama pull qwen2.5-coder:7b
ollama pull mxbai-embed-large
```

### 4. Start codebot

```bash
docker-compose up --build
```

---

## Chat Message Format

```
<llm>|<service>: <message>
```

| Part | Values | Notes |
|------|--------|-------|
| `llm` | `local`, `claude`, `bedrock` | Optional — defaults to `local` |
| `service` | any name in `services.yaml` | Required |
| `message` | free text | Your question |

**Examples:**
```
claude|ibe: why is checkout failing when a promo code is applied?
bedrock|pms: trace the check-in call chain
ibe: where is the cart service?
```

---

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /services` | List configured service names |
| `POST /chat` | Chat using local Ollama (`qwen2.5-coder:7b`) |
| `POST /chat/claude` | Chat using Anthropic Claude |
| `POST /chat/bedrock` | Chat using AWS Bedrock |
| `DELETE /session/{id}` | Clear conversation history for a session (all services) |
| `POST /reindex` | Rebuild index for all services |
| `POST /reindex/{service}` | Rebuild index for one service only |

### Request / Response

```json
POST /chat
{ "message": "ibe: why is checkout failing?", "session_id": "debug-1" }

→ { "response": "...", "sources": ["/repos/ibe-api/src/services/cart.service.ts"] }
```

Unknown or missing service prefix returns `400` with the list of valid services and the expected format.

### After pulling new code

```bash
# Reindex one service
curl -X POST http://localhost:8000/reindex/ibe

# Reindex all
curl -X POST http://localhost:8000/reindex
```

---

## n8n Workflow

Import `codebot.json` into n8n. It parses the `<llm>|<service>:` prefix, routes to the right endpoint, and returns the response.

### Start n8n

```bash
docker run -it --rm --name n8n -p 5678:5678 \
  -v n8n_data:/home/node/.n8n \
  --add-host=host.docker.internal:host-gateway \
  docker.n8n.io/n8nio/n8n
```

Open `http://localhost:5678`, go to **Workflows → Import from file**, select `codebot.json`, activate it, and open the **Chat** panel.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README for codebot"
```

---

### Task 10: Final cleanup and verification

**Files:**
- Remove: `IBE-RAG-MultiEndpoint.json`, `IBE-RAG-Jira.json`, `IBE-RAG.json`

- [ ] **Step 1: Remove old IBE-specific workflow files**

```bash
git rm IBE-RAG-MultiEndpoint.json IBE-RAG-Jira.json IBE-RAG.json
```

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -v`

Expected: all 15 tests PASSED (4 config + 7 message + 7 API - note: test count assumes 7 API tests per test_api.py)

- [ ] **Step 3: Final commit**

```bash
git commit -m "chore: remove old IBE-specific n8n workflow files"
```
