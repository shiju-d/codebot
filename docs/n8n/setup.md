# n8n setup guide

n8n is used to expose codebot as conversational chat interfaces. Each workflow in this directory maps to one codebot endpoint and opens as a chat window in the n8n UI.

---

## 1. Start n8n

n8n runs as a standalone Docker container. Start it with:

```bash
docker run -d \
  --name n8n \
  --restart unless-stopped \
  -p 5678:5678 \
  -e N8N_SECURE_COOKIE=false \
  --add-host host.docker.internal:host-gateway \
  -v n8n_data:/home/node/.n8n \
  docker.n8n.io/n8nio/n8n
```

Key flags:
- `-e N8N_SECURE_COOKIE=false` — allows the UI to work over plain HTTP (required for local development)
- `--add-host host.docker.internal:host-gateway` — lets n8n reach codebot at `http://host.docker.internal:8000` (same mechanism codebot itself uses to reach Ollama)
- `-v n8n_data:/home/node/.n8n` — persists workflows and credentials across restarts

Open **http://localhost:5678** in a browser. On first launch you will be prompted to create an owner account — use any email and password.

---

## 2. Import a workflow

1. In the n8n sidebar click **Workflows → Add workflow**
2. Click the **⋯** menu (top-right) → **Import from file**
3. Select a JSON file from `docs/n8n/`:

| File | What it does |
|---|---|
| `rca.json` | Runs RCA on a Jira ticket and posts the result as a comment |
| `chat-local.json` | Chat with codebot using the local Ollama model |
| `chat-bedrock.json` | Chat with codebot using Claude on AWS Bedrock |
| `reindex.json` | Triggers a re-index for one service or all services |

4. Click **Save** (top-right).
5. Toggle the workflow **Active** (top-right switch). The Chat Trigger only works when the workflow is active.

Repeat for each workflow you want to use.

---

## 3. Open the chat interface

Once a workflow is active:

1. Click the **Chat** button in the top-right of the workflow canvas, or
2. Go to **Workflows**, find the workflow, and click the chat bubble icon next to it.

A chat window opens in your browser. Type your input and press Enter.

---

## 4. Workflow reference

### Chat (Local LLM) — `chat-local.json`

Requires codebot running and Ollama running on the host with both models pulled.

```
rover-ifc: how does the OTA rate push work?
ibe: what does the cart service do?
pms: explain the end-of-day job
```

Responses include a **Sources** section listing the files the answer was drawn from.

---

### Chat (Bedrock / Claude) — `chat-bedrock.json`

Same as the local workflow but uses Claude on AWS Bedrock. Requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` set in codebot's `.env`.

```
rover-ifc: why is the rate for 4 occupants wrong?
ibe: trace the full checkout flow from controller to DB
pms: how does the nightly audit reconcile payments?
```

---

### RCA — `rca.json`

Fetches the Jira ticket, runs analysis, and posts the result as a Jira comment. Requires both AWS Bedrock credentials and Jira credentials in codebot's `.env`.

Input is `service: TICKET-KEY`. Optional free-text context can follow the ticket key.

```
rover-ifc: CICO-134027
ibe: IBE-1152
rover-ifc: CICO-134027 focus on the channex rate exporter
```

---

### Reindex — `reindex.json`

Rebuilds the vector index and code graph. Type a service name to reindex one service, or `all` to reindex everything.

```
rover-ifc
all
```

Reindexing a single service takes 1–3 minutes. Reindexing all services takes longer depending on codebase sizes.

---

## 5. Networking

All workflows call codebot at `http://host.docker.internal:8000`. This hostname resolves to the Docker host machine, where codebot is listening on port 8000.

If n8n was started **without** `--add-host host.docker.internal:host-gateway`, the HTTP calls will fail with a connection error. Stop the container and restart it with that flag.

If codebot is not yet fully initialised (still building indexes on startup), requests will return `503 RAG engine is initializing`. Wait for the `codebot ready` log line before using the workflows.

---

## 6. Updating a workflow

To pull in changes from a JSON file after a git pull:

1. Open the workflow in n8n
2. **⋯** menu → **Import from file** → select the updated JSON
3. Save and confirm the overwrite
