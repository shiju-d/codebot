# How codebot works

codebot is a RAG-powered code assistant that answers questions about Stayntouch codebases and performs root cause analysis on Jira tickets. It runs as a FastAPI service backed by ChromaDB (vector search), Neo4j (code graph), a local LLM via Ollama, and Claude on AWS Bedrock.

---

## Infrastructure

```
┌─────────────────────────────────────────────────┐
│  Docker Compose                                 │
│                                                 │
│  ┌────────────┐     bolt://neo4j:7687           │
│  │  codebot   │ ──────────────────► ┌──────────┐│
│  │  :8000     │                     │  Neo4j 5 ││
│  │            │                     └──────────┘│
│  └────────────┘                                 │
│        │                                        │
│        ├── /app/chroma_db  (ChromaDB volume)    │
│        ├── /root/.cache/huggingface (HF models) │
│        └── /repos/*  (source repos, read-only)  │
│                                                 │
│  host.docker.internal:11434  (Ollama, on host)  │
│  AWS Bedrock  (Claude, via HTTPS)               │
└─────────────────────────────────────────────────┘
```

**Persistent volumes:**
- `chroma_data` — ChromaDB vector index (survives restarts)
- `neo4j_data` — code graph (survives restarts)
- `hf_cache` — Hugging Face reranker model weights (avoids re-downloading)

---

## Services

A **service** is a named codebase configuration defined in `services.yaml`. Each service maps to:

| Field | Purpose |
|---|---|
| `name` | Identifier used in all API calls (e.g. `rover-ifc`) |
| `repos` | One or more repo paths mounted into the container |
| `file_extensions` | Which file types to index |
| `system_prompt` | Expert persona + grounding rules given to the LLM |

Current services: **ibe** (TypeScript, 3 repos), **rover-ifc** (Ruby/Rails), **pms** (Ruby/Rails).

---

## Startup sequence

On container start, `engine.init_all_services()` runs in a background thread:

```
For each service:
  1. Build or load ChromaDB vector index  (indexer.build_service_index)
  2. Parse all source files with tree-sitter, write REQUIRES graph to Neo4j  (graph.build_service_graph)
  3. Create a GraphExpansionPostprocessor tied to that service's collection + Neo4j driver
  4. Register everything in the in-memory services dict
```

Step 1 is skipped if the ChromaDB collection already has chunks (the index persists across restarts).

---

## Module map

| File | Responsibility |
|---|---|
| `app.py` | FastAPI app, HTTP routes, request/response models |
| `engine.py` | Service registry (`services` dict), session management, `get_engine()`, init/reindex logic |
| `llms.py` | LLM clients (Ollama, Bedrock), embed model, prompt helper, Neo4j driver, reranker singleton |
| `indexer.py` | ChromaDB index building, code splitters (tree-sitter + token fallback) |
| `graph.py` | tree-sitter parser for Ruby/TS/JS → Neo4j nodes and REQUIRES edges |
| `graph_retriever.py` | Cypher query that expands a set of seed file paths by 1 hop in the graph |
| `graph_postprocessor.py` | LlamaIndex postprocessor: runs graph expansion between retrieval and reranking |
| `config.py` | `ServiceConfig` dataclass + YAML loader |
| `jira.py` | Jira API calls, ADF text extraction, RCA message builder, Markdown→Jira wiki markup |
| `message.py` | Parses `service: message` prefix format |

---

## Retrieval pipeline

Every chat or RCA query goes through this pipeline:

```
Query
  │
  ▼
QueryFusionRetriever
  ├── generates 3 query variants with the LLM
  ├── runs each against ChromaDB (similarity_top_k=10 each)
  └── fuses results via Reciprocal Rank Fusion → top 30 nodes
  │
  ▼
GraphExpansionPostprocessor  (if Neo4j is available)
  ├── extracts unique file paths from the top-30 nodes
  ├── runs a 1-hop Neo4j traversal (REQUIRES deps + callers)
  ├── fetches chunks for the expanded file paths from ChromaDB
  └── appends them to the node list (score=0, so they rank last before reranking)
  │
  ▼
FlagEmbeddingReranker  (BAAI/bge-reranker-base, CPU)
  ├── scores every node against the original query
  └── keeps top 12
  │
  ▼
ContextChatEngine → LLM (Bedrock/Ollama)
```

**Why graph expansion?** Pure vector search matches by vocabulary. If a Jira ticket describes a symptom with different words than the source file uses, the relevant file may not appear in the top-30. The Neo4j graph finds it through structural relationships: a test file that explicitly `require`s it, or a service that calls it.

**Why reranking after expansion?** The reranker uses a cross-encoder model that scores query-document pairs directly. It is more accurate than vector cosine similarity but too slow to run on thousands of candidates — running it on the fused+expanded set of ~35 nodes is the right trade-off.

---

## Code graph (Neo4j)

`graph.py` parses each source file with `tree-sitter-language-pack` and writes:

**Node types:**
- `File {path, service, language}`
- `Class {name, service, file_path}`
- `Method {name, class_name, file_path, service, start_line, end_line}`

**Edge types:**
- `REQUIRES` — `require`/`require_relative` (Ruby) or `import` (TS/JS)
- `DEFINES` — File → Class
- `INHERITS` — Class → superclass
- `INCLUDES` — Class → mixed-in module (Ruby `include`/`extend`)
- `HAS_METHOD` — Class → Method

The graph expansion query (`graph_retriever.py`) returns up to 5 file paths ranked:
1. Files the seed files directly `REQUIRE`
2. Files that `REQUIRE` the seed files (callers)

> **Rails caveat:** Rails apps use Zeitwerk autoloading — classes are loaded by name, not by explicit `require`. REQUIRES edges only capture explicit `require`/`require_relative` calls. Autoloaded classes have no graph edge.

---

## Sessions

Each (service, LLM backend, session_id) triple has its own `ContextChatEngine` with a `ChatMemoryBuffer` (4096 token limit). Sessions are stored in an `OrderedDict` per service per LLM key. When the cap of 100 sessions is reached, the oldest is evicted (LRU).

Calling `DELETE /session/{id}` clears a session across all services and both LLM backends.

---

## LLM backends

| Backend | Model | Used for |
|---|---|---|
| **Bedrock** | `global.anthropic.claude-opus-4-8` (configurable) | `/chat/bedrock`, `/rca` |
| **Ollama** | `qwen2.5-coder:7b` | `/chat` (local) |
| **Embed** | `mxbai-embed-large` (Ollama) | Indexing + retrieval |
| **Reranker** | `BAAI/bge-reranker-base` (local CPU) | Post-retrieval scoring |

Bedrock requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env`. The `/rca` endpoint is Bedrock-only.

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/services` | List available services |
| `POST` | `/chat` | Chat with a service using the local Ollama LLM |
| `POST` | `/chat/bedrock` | Chat using Claude on Bedrock |
| `DELETE` | `/session/{id}` | Clear a chat session |
| `POST` | `/rca` | Fetch a Jira ticket, run RAG, post the RCA as a Jira comment |
| `POST` | `/reindex` | Rebuild vector index + graph for all services |
| `POST` | `/reindex/{service}` | Rebuild for one service only |

### Chat request format

All chat endpoints expect `service: message` in the message body:

```json
{ "message": "rover-ifc: why is the rate for 4 occupants wrong?", "session_id": "my-session" }
```

### RCA request format

```json
{ "input": "rover-ifc: CICO-134027" }
```

Optionally add extra context after the ticket key:
```json
{ "input": "rover-ifc: CICO-134027 focus on the channex exporter" }
```

The RCA flow:
1. Fetch the Jira issue summary + description
2. Build a structured prompt with the grounding rule
3. Run the retrieval pipeline
4. Post the LLM response as a Jira comment (Markdown → Jira wiki markup)
5. Return the response and source file list

---

## Configuration

All configuration lives in two files:

**`.env`** — secrets and infrastructure URLs (not committed):
```
REPO_ROOT=/path/to/repos
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
BEDROCK_MODEL_ID=global.anthropic.claude-opus-4-8
JIRA_BASE_URL=https://yourorg.atlassian.net
JIRA_EMAIL=you@yourorg.com
JIRA_API_TOKEN=...
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=codebot-secret
```

**`services.yaml`** — service definitions (committed, safe):
```yaml
services:
  - name: my-service
    system_prompt: |
      You are an expert in ...
    file_extensions: [.rb, .erb]
    repos:
      - /repos/my-service
```

---

## Adding a new service

1. Mount the repo in `docker-compose.yml` under `codebot.volumes`
2. Add a service entry to `services.yaml`
3. Call `POST /reindex/{service-name}` to build its index and graph
