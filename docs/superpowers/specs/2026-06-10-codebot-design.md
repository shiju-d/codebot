# codebot — Design Spec

**Date:** 2026-06-10
**Status:** Approved

## Overview

codebot is a RAG-powered code intelligence API for Stayntouch repositories. It extends ibex (IBE-specific) into a configurable, multi-service tool where each service bundles one or more repos into a single queryable index.

---

## Config (`services.yaml`)

A single YAML file defines all services. Each service has a name, a system prompt, and a list of local repo paths.

```yaml
services:
  - name: ibe
    system_prompt: |
      You are an expert in the Stayntouch IBE application.
      IBE has three layers: ibe-api (LoopBack 4), ibe-frontend (Express/Jade), ibe-admin (Angular 19).
      When analysing bugs: identify the layer, trace the call chain, point to the exact file and function, suggest a fix.
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
      - /repos/pms-frontend
```

- Repos are local paths mounted as read-only volumes in Docker Compose
- The service `name` doubles as the message prefix
- File types indexed: `.js`, `.jsx`, `.ts`, `.tsx` (global, not per-service)

---

## Runtime Architecture

On startup, codebot reads `services.yaml` and for each service:

1. Creates or loads a ChromaDB collection named `{service_name}_codebase`
2. If the collection is empty, indexes all repos in that service
3. Caches `{ index, system_prompt, sessions }` keyed by service name

**Request flow:**

```
"claude|ibe: why is checkout failing?"
    │
    ├── n8n strips LLM prefix → routes to POST /chat/claude
    │   passes message: "ibe: why is checkout failing?"
    │
    └── codebot strips service prefix → service = "ibe", message = "why is checkout failing?"
          ├── look up ibe index + ibe system_prompt
          ├── get/create session engine for session_id
          └── run chat → return response + sources
```

Missing or unknown service prefix → `400` response with the expected format and a list of valid service names. Example:
```json
{
  "error": "Unknown service 'xyz'. Valid services: ibe, pms. Format: <llm>|<service>: <message>"
}
```

**Session storage** is nested by service name and LLM type. Each service × LLM combination maintains up to 100 sessions (LRU eviction).

---

## API

### Chat endpoints (unchanged URLs)

| Endpoint | LLM |
|----------|-----|
| `POST /chat` | local Ollama (`qwen2.5-coder:7b`) |
| `POST /chat/claude` | Anthropic Claude |
| `POST /chat/bedrock` | AWS Bedrock |

**Request body (unchanged):**
```json
{
  "message": "ibe: why is checkout failing?",
  "session_id": "debug-session-1"
}
```

**Response (unchanged):**
```json
{
  "response": "...",
  "sources": ["/repos/ibe-api/src/services/cart.service.ts"]
}
```

### New endpoints

| Endpoint | What it does |
|----------|-------------|
| `GET /services` | List all configured service names (names only — no repo paths or system prompts) |
| `POST /reindex/{service_name}` | Wipe and rebuild index for one service |
| `POST /reindex` | Wipe and rebuild all services |
| `DELETE /session/{session_id}` | Clear session across all services and LLMs (unchanged) |

---

## Message Format

```
<llm>|<service>: <message>
```

| Part | Values | Notes |
|------|--------|-------|
| `llm` | `local`, `claude`, `bedrock` | Optional — defaults to `local` if omitted |
| `service` | any name in `services.yaml` | Required |
| `message` | free text | The question |

**Examples:**
```
claude|ibe: why is checkout failing when a promo code is applied?
bedrock|pms: trace the check-in call chain
ibe: where is the cart service?
```

---

## Docker Compose

```yaml
services:
  codebot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./services.yaml:/app/services.yaml:ro
      - ./chroma_db:/app/chroma_db
      - /path/to/ibe-api:/repos/ibe-api:ro
      - /path/to/ibe-frontend:/repos/ibe-frontend:ro
      - /path/to/ibe-admin:/repos/ibe-admin:ro
      - /path/to/pms-api:/repos/pms-api:ro
    env_file: .env
```

No container rebuild needed when adding a new service — update `services.yaml`, add the volume mount, and restart.

**`.env` file is unchanged:** `ANTHROPIC_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `BEDROCK_MODEL_ID`.

---

## n8n Workflow

The workflow file is renamed from `IBE-RAG-MultiEndpoint.json` to `codebot.json`.

**Parsing logic:**
1. Extract LLM prefix (`local|`, `claude|`, `bedrock|`) — default to `local` if absent
2. Route to the corresponding endpoint (`/chat`, `/chat/claude`, `/chat/bedrock`)
3. Pass the remainder (e.g., `ibe: why is checkout failing?`) as the message body

**Chat Trigger placeholder text:**
```
e.g. claude|ibe: why is checkout failing?
```

**Canvas sticky note:**
```
Format:   <llm>|<service>: <message>

LLMs:     local | claude | bedrock
Services: ibe | pms | ...    (see GET /services)

Examples:
  claude|ibe: why is checkout failing?
  bedrock|pms: trace the check-in flow
  ibe: where is the cart service?
```

---

## Key Changes from ibex

| ibex | codebot |
|------|---------|
| Single hardcoded index for IBE | One ChromaDB collection per service |
| `SYSTEM_PROMPT` constant | Per-service system prompt from `services.yaml` |
| `/app/ibe` hardcoded path | Configurable repo paths via `services.yaml` |
| Three separate session dicts | Sessions nested by service name |
| `POST /reindex` only | `POST /reindex` + `POST /reindex/{service_name}` |
| No service listing | `GET /services` |
| `IBE-RAG-MultiEndpoint.json` | `codebot.json` with updated prefix parsing |
