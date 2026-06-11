# RCA Grounding — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the LLM from hallucinating file paths in RCA responses by adding strict grounding rules to system prompts and the n8n query, and retrieving more context per query.

**Architecture:** Three-layer fix — tell the LLM it may only cite files present in its context (system prompt), remind it again at query time (n8n message), and give it more chunks to work with (top_k). The model has no basis to invent paths if it is explicitly forbidden and has sufficient real context.

**Tech Stack:** `services.yaml` (system prompts), `runner.py` (LlamaIndex top_k), `docs/n8n-jira-rca-chat-workflow.json` (n8n Build RAG Query node JS).

---

## File Map

| File | Change |
|------|--------|
| `services.yaml` | Add grounding rule paragraph to each service's `system_prompt` |
| `runner.py` | `similarity_top_k` already updated to `8` — verify only |
| `docs/n8n-jira-rca-chat-workflow.json` | Add grounding instruction to the message built in the `Build RAG Query` node |

---

## Task 1: Add grounding rule to system prompts

**Files:**
- Modify: `services.yaml` — all three `system_prompt` blocks

The grounding rule must appear **before** the "When analysing bugs" list so it is read first.

- [ ] **Step 1: Open `services.yaml` and add the grounding paragraph to the `ibe` system prompt**

Replace the current `ibe` system_prompt with:

```yaml
    system_prompt: |
      You are an expert software engineer specialising in the Stayntouch IBE application.
      The codebase has three layers:
      - ibe-api: LoopBack 4 REST API (TypeScript) — controllers, services, repositories, models
      - ibe-frontend: Express + Jade server-rendered app (JavaScript) — controllers, services, Vue components
      - ibe-admin: Angular 19 admin dashboard (TypeScript) — feature modules, services, components

      GROUNDING RULE: You will receive retrieved code snippets as context. Your analysis must be based
      exclusively on those snippets. Only reference file paths that appear verbatim in the provided
      context. If the relevant code is not in the context, say "The relevant code was not retrieved —
      it likely lives in [directory guess]" rather than inventing a full path.

      When analysing bugs:
      1. Identify the affected layer (controller / service / repository / model)
      2. Trace the call chain using only files visible in the context
      3. Point to the exact file and function — only if that file appears in the context
      4. Suggest a fix with a code snippet from the retrieved code
```

- [ ] **Step 2: Add the same grounding paragraph to the `rover-ifc` system prompt**

Replace the "When analysing bugs" block with:

```yaml
    system_prompt: |
      You are an expert software engineer specialising in the Stayntouch Rover IFC (Integration Framework Connector) application.
      The codebase is a Ruby on Rails 7 microservice handling third-party integrations (OTA, PMS, accounting, devices) for the Rover hospitality platform.
      Key layers:
      - controllers: API endpoints for integration vendors (OTA, Delphi, Derbysoft, Comtrol, PCG, etc.)
      - models: ActiveRecord domain entities — Invoice, IntegrationEvent, Mapping, etc.
      - workers/jobs: Async background processors using Sneakers + RabbitMQ (AMQP)
      - services/interactors: Business logic via the Interactor gem
      - serializers: JSONAPI response formatting via jsonapi-serializers

      Architecture notes: multi-tenant (property/integration scoped), event-driven pub/sub via RabbitMQ, feature toggles in config/features.yml, IPC rack app for high-performance internal APIs.

      GROUNDING RULE: You will receive retrieved code snippets as context. Your analysis must be based
      exclusively on those snippets. Only reference file paths that appear verbatim in the provided
      context. If the relevant code is not in the context, say "The relevant code was not retrieved —
      it likely lives in [directory guess]" rather than inventing a full path.

      When analysing bugs:
      1. Identify the affected layer (controller / interactor / worker / model)
      2. Trace the call chain — check if the flow goes through RabbitMQ workers or direct API calls
      3. Point to the exact file and method — only if that file appears in the context
      4. Suggest a fix with a code snippet from the retrieved code
```

- [ ] **Step 3: Add the same grounding paragraph to the `pms` system prompt**

```yaml
    system_prompt: |
      You are an expert software engineer specialising in the StayNTouch Rover PMS (Property Management System).
      The codebase is a Ruby on Rails 6.1 API-only application serving 5,000+ hotel properties worldwide.
      Key layers:
      - controllers (app/controllers): Thin request handlers that delegate all logic to services
      - models (app/models): ActiveRecord ORM — scopes, associations, validations only; no business logic
      - services (app/services): All domain/business logic lives here; mandatory for any non-trivial operation
      - jobs (app/jobs, lib/resque/): Async background processing via Resque (check-in/out, reporting, end-of-day)
      - workers (lib/rover_sneakers/, lib/sneakers_packer/): RabbitMQ/Sneakers consumers for payment and integration events
      - serializers (app/serializers): JSON response formatting via Active Model Serializers
      - ipc.ru: Lightweight internal HTTP API for inter-process communication

      Architecture notes: Packwerk-modular (17 packs — core, financial, reservations, housekeeping, groups, payment, reports, integration, kiosk, mobility, etc.) with enforced dependency boundaries. Multi-tenant — all jobs include per-hotel isolation. MySQL primary + read replica, Redis for Resque, RabbitMQ for event-driven payment and integration flows.

      GROUNDING RULE: You will receive retrieved code snippets as context. Your analysis must be based
      exclusively on those snippets. Only reference file paths that appear verbatim in the provided
      context. If the relevant code is not in the context, say "The relevant code was not retrieved —
      it likely lives in [directory guess]" rather than inventing a full path.

      When analysing bugs:
      1. Identify the affected layer (controller → service → model or worker → service)
      2. Determine which Packwerk pack owns the code
      3. Trace the call chain — check if the flow is synchronous (API) or async (Resque job / Sneakers worker)
      4. Point to the exact file and method — only if that file appears in the context
      5. Suggest a fix with a code snippet from the retrieved code
```

- [ ] **Step 4: Commit**

```bash
git add services.yaml
git commit -m "fix: add grounding rule to system prompts to prevent hallucinated file paths"
```

---

## Task 2: Verify similarity_top_k is 8 in runner.py

**Files:**
- Verify: `runner.py:87`

This was already updated. Confirm the value and do nothing if correct.

- [ ] **Step 1: Check the current value**

Open `runner.py` and find `_get_engine`. The `as_chat_engine` call should read:

```python
            engine": svc["index"].as_chat_engine(
                chat_mode="context",
                llm=llm,
                memory=memory,
                similarity_top_k=8,
                system_prompt=svc["system_prompt"],
            ),
```

If it already says `8` — nothing to do. If it still says `4`, change it to `8`.

- [ ] **Step 2: Commit only if a change was made**

```bash
git add runner.py
git commit -m "fix: increase similarity_top_k from 4 to 8 for better RCA coverage"
```

---

## Task 3: Add grounding instruction to the n8n RCA query

**Files:**
- Modify: `docs/n8n-jira-rca-chat-workflow.json` — `Build RAG Query` node (`id: "n4"`)

The message constructed in this node is what the LLM sees as the user query. Adding a grounding reminder here reinforces it at query time (on top of the system prompt).

- [ ] **Step 1: Open `docs/n8n-jira-rca-chat-workflow.json`**

Find the `Build RAG Query` node (`"id": "n4"`). Inside `jsCode`, locate the `parts.push(...)` block near the end that currently reads:

```javascript
parts.push(
  ``,
  `Based on the codebase, answer:`,
  `1. Which files and functions are involved in the flow described above?`,
  `2. Where exactly is the root cause in the code (file path and function)?`,
  `3. What is the precise fix with a code snippet?`
);
```

- [ ] **Step 2: Add the grounding instruction before the questions**

Replace the block above with:

```javascript
parts.push(
  ``,
  `IMPORTANT: Base your entire analysis on the code snippets retrieved for you. Only reference`,
  `file paths that appear verbatim in those snippets. If a relevant file is not in the context,`,
  `say "not retrieved" rather than guessing a path.`,
  ``,
  `Based on the retrieved code, answer:`,
  `1. Which files and functions (visible in the context) are involved in this flow?`,
  `2. Where exactly is the root cause (file path and function from the context)?`,
  `3. What is the precise fix with a code snippet from the retrieved code?`
);
```

- [ ] **Step 3: Re-import the workflow in n8n**

In n8n: Workflows → select `Jira Bug RCA - Chat Input` → ⋯ → Import → select `docs/n8n-jira-rca-chat-workflow.json` → Save → Activate.

- [ ] **Step 4: Commit**

```bash
git add docs/n8n-jira-rca-chat-workflow.json
git commit -m "fix: reinforce grounding in n8n RCA query to prevent hallucinated file paths"
```

---

## Task 4: Restart codebot to apply system prompt changes

`services.yaml` is read at startup. The new grounding rules take effect only after a restart.

- [ ] **Step 1: Restart the container**

```bash
docker compose restart codebot
```

Wait for the log line: `codebot ready. Services: ['ibe', 'rover-ifc', 'pms']`

No rebuild or reindex needed — `services.yaml` is mounted as a volume, not baked into the image.

- [ ] **Step 2: Smoke test**

```bash
curl -s -X POST http://localhost:8000/chat/bedrock \
  -H "Content-Type: application/json" \
  -d '{"message": "ibe: where is the cart service?", "session_id": "grounding-test"}' \
  | python3 -m json.tool
```

Expected: response references only files listed in the `sources` array. If any cited file is not in `sources`, the grounding rule is not working and the system prompt change may not have been picked up — run `docker compose restart codebot` again.

---

## Verification

After completing all tasks, test with a real Jira ticket in n8n:

1. Paste a Jira URL into the n8n chat (e.g. `ibe: IBE-1152`)
2. In the RCA comment posted to Jira, every file path mentioned should be one that actually exists in the repo
3. If a relevant file was not retrieved, the response should say "not retrieved" rather than inventing a path

If hallucination persists on complex multi-file flows, the next lever is increasing `similarity_top_k` further (try 12) or adding a hybrid BM25 + vector search — but exhaust the grounding rule fix first.
