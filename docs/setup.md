# Setup guide

## Prerequisites

| Tool | Why |
|---|---|
| Docker + Docker Compose | Runs codebot and Neo4j |
| [Ollama](https://ollama.com) | Local LLM and embedding model — runs on the host, not in Docker |
| AWS credentials | Required for Bedrock (Claude) and the `/rca` endpoint |
| Jira API token | Required for the `/rca` endpoint |

---

## 1. Pull Ollama models

codebot needs two models running locally before the container starts.

```bash
ollama pull qwen2.5-coder:7b      # chat LLM
ollama pull mxbai-embed-large     # embedding model (used for indexing)
```

Verify they're available:

```bash
ollama list
```

---

## 2. Clone the source repos

codebot indexes source repos from the host filesystem. They must be checked out before the container starts. By default all repos are expected under a common parent directory — set that path as `REPO_ROOT` in your `.env` (next step).

Expected layout (matching `docker-compose.yml`):

```
$REPO_ROOT/
  ibe-api/
  ibe-frontend/
  ibe-admin/
  rover-ifc/
  pms/
```

---

## 3. Create `.env`

Copy the template below to `.env` in the project root (same directory as `docker-compose.yml`). This file is **not committed** — keep it out of version control.

```bash
# Path on your host machine containing all source repos
REPO_ROOT=/path/to/your/repos

# AWS — required for Bedrock (/chat/bedrock, /rca)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=global.anthropic.claude-sonnet-4-5-20250929-v1:0

# Jira — required for /rca
JIRA_BASE_URL=https://yourorg.atlassian.net
JIRA_EMAIL=you@yourorg.com
JIRA_API_TOKEN=...

# Neo4j (defaults match docker-compose.yml — only change if you've customised them)
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=codebot-secret

# Ollama (default works when Ollama is running on the host)
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

**Getting a Jira API token:** go to https://id.atlassian.com/manage-profile/security/api-tokens → Create API token.

---

## 4. Start the stack

```bash
docker compose up --build
```

On first run this will:
1. Build the codebot image (downloads Python deps including `FlagEmbedding` and CPU torch — takes a few minutes)
2. Start Neo4j
3. Start codebot, which in the background:
   - Builds a ChromaDB vector index for each service (chunking and embedding all source files)
   - Parses all source files with tree-sitter and writes the code graph to Neo4j
   - Downloads the reranker model weights to the `hf_cache` volume

Startup is complete when you see:

```
codebot ready. Services: ['ibe', 'rover-ifc', 'pms']
```

Subsequent starts are faster: the ChromaDB index and Neo4j graph persist in Docker volumes and are not rebuilt unless you call `/reindex`.

---

## 5. Verify

```bash
curl http://localhost:8000/services
# → {"services":["ibe","rover-ifc","pms"]}
```

---

## Common operations

### Chat (local LLM)

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message": "rover-ifc: how does the OTA rate push work?", "session_id": "dev"}'
```

### Chat (Bedrock / Claude)

```bash
curl -s -X POST http://localhost:8000/chat/bedrock \
  -H 'Content-Type: application/json' \
  -d '{"message": "pms: what does the end-of-day job do?", "session_id": "dev"}'
```

### Root cause analysis

Runs against Bedrock and posts the result as a Jira comment.

```bash
curl -s -X POST http://localhost:8000/rca \
  -H 'Content-Type: application/json' \
  -d '{"input": "rover-ifc: CICO-134027"}'
```

Add optional context after the ticket key:

```bash
-d '{"input": "rover-ifc: CICO-134027 focus on the channex rate exporter"}'
```

### Rebuild the index for one service

Use this after pulling new commits for a service repo:

```bash
curl -s -X POST http://localhost:8000/reindex/rover-ifc
```

### Rebuild all services

```bash
curl -s -X POST http://localhost:8000/reindex
```

---

## Adding a new service

1. Check out the repo under `$REPO_ROOT`
2. Mount it in `docker-compose.yml` under `codebot.volumes`:
   ```yaml
   - ${REPO_ROOT}/my-service:/repos/my-service:ro
   ```
3. Add an entry to `services.yaml`:
   ```yaml
   - name: my-service
     system_prompt: |
       You are an expert in ...
     file_extensions: [.rb, .erb]
     repos:
       - /repos/my-service
   ```
4. Restart the stack (`docker compose up --build`) or call `POST /reindex/my-service` if the container is already running with the new volume mount

---

## Troubleshooting

**`codebot ready` never appears**

Check container logs: `docker compose logs -f codebot`. Common causes:
- Ollama is not running or the models are not pulled — run `ollama list` to verify
- A repo path under `REPO_ROOT` does not exist — Docker will mount it as an empty directory and log a warning

**`/rca` returns 503**

Either `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are missing from `.env`, or the Jira credentials are not set. All three Jira variables (`JIRA_BASE_URL`, `JIRA_EMAIL`, `JIRA_API_TOKEN`) must be present.

**Disk space error during build**

The CPU torch and FlagEmbedding packages are large. Run `docker system prune -a` to free space before building.

**Graph expansion not working**

If codebot logs `Neo4j not available`, the graph postprocessor is silently disabled and only vector search runs. Check that `NEO4J_URI` is `bolt://neo4j:7687` (not `localhost`) inside the container.

**Responses based on stale code**

Pull the latest commits for the relevant repo, then call `POST /reindex/{service}` to rebuild.
