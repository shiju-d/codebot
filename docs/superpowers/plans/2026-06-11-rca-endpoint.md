# RCA Endpoint — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the entire Jira RCA workflow from n8n into a single `POST /rca` endpoint in codebot, so n8n is reduced to a 3-node passthrough (Chat Trigger → HTTP Request → Respond to Chat).

**Architecture:** A new `jira.py` module holds all Jira-specific helpers (input parsing, ADF extraction, Markdown→Jira conversion, RAG message construction, HTTP client). `runner.py` gets three new env vars and one new endpoint that calls through to the existing `_get_engine()` + `_chat()` infrastructure. The endpoint fetches the Jira issue, runs RAG, formats the comment, posts it, and returns a ready-to-display response string.

**Tech Stack:** Python `httpx` (already in requirements), FastAPI, existing LlamaIndex RAG pipeline, Jira REST API v3 (fetch) + v2 (comment).

---

## File Map

| File | Action |
|------|--------|
| `jira.py` | **Create** — pure helpers + async Jira HTTP client |
| `tests/test_jira.py` | **Create** — unit tests for all pure functions in `jira.py` |
| `runner.py` | **Modify** — add Jira env vars, `RcaRequest`, `POST /rca` endpoint |
| `docs/n8n-rca-simple-workflow.json` | **Create** — 3-node n8n workflow |
| `README.md` | **Modify** — add `/rca` endpoint, Jira env vars to setup section |

---

## Task 1: Create `jira.py` — pure helper functions

**Files:**
- Create: `jira.py`
- Test: `tests/test_jira.py`

The pure functions have no I/O and are straightforward to test.

- [ ] **Step 1: Write failing tests for `parse_rca_input`**

Create `tests/test_jira.py`:

```python
import pytest
from jira import parse_rca_input, extract_adf_text, build_rca_message, md_to_jira


# --- parse_rca_input ---

def test_parse_rca_bare_key():
    service, key, ctx = parse_rca_input("ibe: IBE-1152")
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""

def test_parse_rca_full_url():
    service, key, ctx = parse_rca_input(
        "ibe: https://stayntouch.atlassian.net/browse/IBE-1152"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""

def test_parse_rca_with_additional_context():
    service, key, ctx = parse_rca_input(
        "ibe: IBE-1152\nStack trace: TypeError at line 45"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == "Stack trace: TypeError at line 45"

def test_parse_rca_uppercase_service_normalised():
    service, key, _ = parse_rca_input("IBE: IBE-1152")
    assert service == "ibe"

def test_parse_rca_missing_colon_raises():
    with pytest.raises(ValueError, match="Format"):
        parse_rca_input("ibe IBE-1152")

def test_parse_rca_missing_key_raises():
    with pytest.raises(ValueError, match="Jira issue key"):
        parse_rca_input("ibe: some random text")

def test_parse_rca_empty_service_raises():
    with pytest.raises(ValueError):
        parse_rca_input(": IBE-1152")
```

- [ ] **Step 2: Run to confirm they all fail**

```bash
cd /path/to/codebot && pytest tests/test_jira.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'jira'`

- [ ] **Step 3: Write failing tests for `extract_adf_text`**

Append to `tests/test_jira.py`:

```python
# --- extract_adf_text ---

def test_extract_adf_plain_text():
    node = {"type": "text", "text": "hello"}
    assert extract_adf_text(node) == "hello"

def test_extract_adf_nested():
    node = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Error "},
                {"type": "text", "text": "occurs here"},
            ]},
        ],
    }
    assert "Error" in extract_adf_text(node)
    assert "occurs here" in extract_adf_text(node)

def test_extract_adf_none_returns_empty():
    assert extract_adf_text(None) == ""

def test_extract_adf_no_content_returns_empty():
    assert extract_adf_text({"type": "hardBreak"}) == ""
```

- [ ] **Step 4: Write failing tests for `build_rca_message`**

Append to `tests/test_jira.py`:

```python
# --- build_rca_message ---

def test_build_rca_message_contains_service_prefix():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "User gets 500", "")
    assert msg.startswith("ibe: You are performing Root Cause Analysis")

def test_build_rca_message_contains_ticket_info():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "User gets 500", "")
    assert "Ticket: IBE-1152" in msg
    assert "Summary: Checkout fails" in msg
    assert "User gets 500" in msg

def test_build_rca_message_contains_grounding_instruction():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "Desc", "")
    assert "IMPORTANT" in msg
    assert "not retrieved" in msg

def test_build_rca_message_includes_additional_context():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "Stack trace: ...")
    assert "Additional context provided by reporter" in msg
    assert "Stack trace: ..." in msg

def test_build_rca_message_no_additional_context_section_when_empty():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "")
    assert "Additional context" not in msg
```

- [ ] **Step 5: Write failing tests for `md_to_jira`**

Append to `tests/test_jira.py`:

```python
# --- md_to_jira ---

def test_md_to_jira_heading_levels():
    assert md_to_jira("# H1") == "h1. H1"
    assert md_to_jira("## H2") == "h2. H2"
    assert md_to_jira("### H3") == "h3. H3"

def test_md_to_jira_bold():
    assert md_to_jira("**bold**") == "*bold*"

def test_md_to_jira_inline_code():
    assert md_to_jira("`foo`") == "{{foo}}"

def test_md_to_jira_fenced_code_with_language():
    result = md_to_jira("```python\nprint('hi')\n```")
    assert "{code:python}" in result
    assert "print('hi')" in result
    assert result.endswith("{code}")

def test_md_to_jira_fenced_code_no_language():
    result = md_to_jira("```\nsome code\n```")
    assert "{code}" in result
    assert "some code" in result

def test_md_to_jira_unordered_list():
    assert "* item" in md_to_jira("- item")
    assert "* item" in md_to_jira("* item")

def test_md_to_jira_horizontal_rule():
    assert "----" in md_to_jira("---")
```

- [ ] **Step 6: Implement `jira.py`**

Create `jira.py`:

```python
import re
import httpx


def parse_rca_input(input_str: str) -> tuple[str, str, str]:
    """Parse '<service>: <jira_url_or_key> [optional context]' → (service, issue_key, additional_context)."""
    colon_idx = input_str.find(':')
    if colon_idx == -1:
        raise ValueError("Format: <service>: <jira_url_or_key>  e.g. ibe: IBE-1152")
    service = input_str[:colon_idx].strip().lower()
    if not service:
        raise ValueError("Format: <service>: <jira_url_or_key>  e.g. ibe: IBE-1152")
    rest = input_str[colon_idx + 1:].strip()
    match = re.search(r'([A-Z]+-\d+)', rest)
    if not match:
        raise ValueError(f"Could not find a Jira issue key (e.g. IBE-1152) in: {rest}")
    issue_key = match.group(1)
    after_key = rest[match.end():].strip()
    additional_context = re.sub(r'^/[^\s]*', '', after_key).strip()
    return service, issue_key, additional_context


def extract_adf_text(node: dict | None) -> str:
    """Recursively extract plain text from Jira's Atlassian Document Format (ADF)."""
    if not node:
        return ''
    if node.get('type') == 'text':
        return node.get('text', '')
    if 'content' in node:
        return ' '.join(extract_adf_text(child) for child in node['content'])
    return ''


def build_rca_message(
    service: str,
    issue_key: str,
    summary: str,
    description: str,
    additional_context: str = '',
) -> str:
    """Build the RAG query message sent to the LLM for RCA."""
    parts = [
        f"{service}: You are performing Root Cause Analysis (RCA) for a production bug. Analyse the codebase carefully.",
        "",
        f"Ticket: {issue_key}",
        f"Summary: {summary}",
        "",
        "Description:",
        description,
    ]
    if additional_context:
        parts.extend(["", "Additional context provided by reporter:", additional_context])
    parts.extend([
        "",
        "IMPORTANT: Base your entire analysis on the code snippets retrieved for you. Only reference",
        "file paths that appear verbatim in those snippets. If a relevant file is not in the context,",
        'say "not retrieved" rather than guessing a path.',
        "",
        "Based on the retrieved code, answer:",
        "1. Which files and functions (visible in the context) are involved in this flow?",
        "2. Where exactly is the root cause (file path and function from the context)?",
        "3. What is the precise fix with a code snippet from the retrieved code?",
    ])
    return "\n".join(parts)


def md_to_jira(text: str) -> str:
    """Convert Markdown to Jira wiki markup."""
    text = re.sub(r'```(\w+)\n([\s\S]*?)```', r'{code:\1}\n\2{code}', text)
    text = re.sub(r'```\n?([\s\S]*?)```', r'{code}\n\1{code}', text)
    text = re.sub(r'`([^`\n]+)`', r'{{\1}}', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'^### (.+)$', r'h3. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'h2. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'h1. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. (.+)$', r'# \1', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]*[-*] (.+)$', r'* \1', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', r'----', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


async def fetch_jira_issue(base_url: str, email: str, token: str, issue_key: str) -> dict:
    """Fetch a Jira issue via REST API v3. Raises httpx.HTTPStatusError on failure."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{base_url}/rest/api/3/issue/{issue_key}",
            auth=(email, token),
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()


async def post_jira_comment(base_url: str, email: str, token: str, issue_key: str, body: str) -> None:
    """Post a plain-text comment to a Jira issue via REST API v2. Raises httpx.HTTPStatusError on failure."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{base_url}/rest/api/2/issue/{issue_key}/comment",
            auth=(email, token),
            json={"body": body},
            timeout=30.0,
        )
        r.raise_for_status()
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
pytest tests/test_jira.py -v
```

Expected: all tests pass. If any fail, fix `jira.py` before proceeding.

- [ ] **Step 8: Commit**

```bash
git add jira.py tests/test_jira.py
git commit -m "feat: add jira helpers — parse_rca_input, extract_adf_text, build_rca_message, md_to_jira, Jira HTTP client"
```

---

## Task 2: Add `POST /rca` endpoint to `runner.py`

**Files:**
- Modify: `runner.py`

- [ ] **Step 1: Add Jira imports and env vars**

At the top of `runner.py`, after the existing imports, add:

```python
from jira import (
    parse_rca_input, extract_adf_text, build_rca_message,
    md_to_jira, fetch_jira_issue, post_jira_comment,
)
```

After the existing env var block (after `SERVICES_CONFIG_PATH`), add:

```python
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
```

- [ ] **Step 2: Add `RcaRequest` model**

After the existing `ChatRequest` class, add:

```python
class RcaRequest(BaseModel):
    input: str
    session_id: str = ""
```

- [ ] **Step 3: Add `POST /rca` endpoint**

After the `DELETE /session/{session_id}` endpoint, add:

```python
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
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Jira API error fetching {issue_key}: {e.response.status_code}")

    summary = issue["fields"]["summary"]
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
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to post Jira comment on {issue_key}: {e.response.status_code}",
        )

    return {
        "response": rca_text,
        "sources": sources,
        "issue_key": issue_key,
        "comment_posted": True,
        "output": f"✅ RCA posted to {issue_key}.\n\n{rca_text}",
    }
```

Note: `httpx` must be imported at the top of `runner.py`. Add it alongside the existing imports:

```python
import httpx
```

- [ ] **Step 4: Commit**

```bash
git add runner.py
git commit -m "feat: add POST /rca endpoint — full Jira RCA flow in codebot"
```

---

## Task 3: Create simplified n8n workflow

**Files:**
- Create: `docs/n8n-rca-simple-workflow.json`

This replaces the 8-node workflow with 3 nodes: Chat Trigger → HTTP Request → Respond to Chat.

- [ ] **Step 1: Create `docs/n8n-rca-simple-workflow.json`**

```json
{
  "name": "Jira RCA - Simple",
  "nodes": [
    {
      "parameters": {
        "options": {}
      },
      "id": "s1",
      "name": "Chat Trigger",
      "type": "@n8n/n8n-nodes-langchain.chatTrigger",
      "typeVersion": 1.1,
      "position": [240, 300],
      "webhookId": "jira-rca-simple"
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://host.docker.internal:8000/rca",
        "sendBody": true,
        "specifyBody": "keypair",
        "bodyParameters": {
          "parameters": [
            { "name": "input", "value": "={{ $json.chatInput }}" },
            { "name": "session_id", "value": "" }
          ]
        },
        "options": {}
      },
      "id": "s2",
      "name": "Ask codebot RCA",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.2,
      "position": [460, 300]
    },
    {
      "parameters": {
        "language": "javaScript",
        "jsCode": "return { output: $('Ask codebot RCA').item.json.output };"
      },
      "id": "s3",
      "name": "Respond to Chat",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [680, 300]
    }
  ],
  "connections": {
    "Chat Trigger": {
      "main": [
        [{ "node": "Ask codebot RCA", "type": "main", "index": 0 }]
      ]
    },
    "Ask codebot RCA": {
      "main": [
        [{ "node": "Respond to Chat", "type": "main", "index": 0 }]
      ]
    }
  },
  "settings": {
    "executionOrder": "v1"
  }
}
```

- [ ] **Step 2: Commit**

```bash
git add docs/n8n-rca-simple-workflow.json
git commit -m "feat: add simplified 3-node n8n workflow for POST /rca"
```

---

## Task 4: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add `/rca` to the API Endpoints table**

In the `## API Endpoints` section, add a row:

```markdown
| `POST /rca` | Full Jira RCA: fetch issue → RAG → format → post Jira comment |
```

- [ ] **Step 2: Add Jira env vars to the `.env` example in Setup**

In the `### 2. Configure .env` section, add:

```
# Jira (required for POST /rca)
JIRA_BASE_URL=https://stayntouch.atlassian.net
JIRA_EMAIL=you@stayntouch.com
JIRA_API_TOKEN=<your-jira-api-token>
```

- [ ] **Step 3: Add `/rca` request/response example**

In the `### Request / Response` section, add after the existing example:

```markdown
```json
POST /rca
{ "input": "ibe: IBE-1152" }

→ {
    "response": "...",
    "sources": ["ibe-api/src/services/cart.service.ts"],
    "issue_key": "IBE-1152",
    "comment_posted": true,
    "output": "✅ RCA posted to IBE-1152.\n\n..."
  }
```
```

- [ ] **Step 4: Update n8n section**

Replace the import instruction to reference both workflow files:

```markdown
- **Full workflow** (original, 8 nodes): `docs/n8n-jira-rca-chat-workflow.json`
- **Simple workflow** (recommended, 3 nodes): `docs/n8n-rca-simple-workflow.json` — requires `JIRA_*` env vars in codebot
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add /rca endpoint, Jira env vars, simplified n8n workflow to README"
```

---

## Manual Steps After Implementation

These are not code tasks — the engineer must do them by hand:

1. **Add Jira credentials to `.env`:**
   ```
   JIRA_BASE_URL=https://stayntouch.atlassian.net
   JIRA_EMAIL=shiju.devarajan@stayntouch.com
   JIRA_API_TOKEN=<token from https://id.atlassian.com/manage-profile/security/api-tokens>
   ```

2. **Rebuild and restart codebot:**
   ```bash
   docker compose up --build
   ```

3. **Smoke test the new endpoint:**
   ```bash
   curl -s -X POST http://localhost:8000/rca \
     -H "Content-Type: application/json" \
     -d '{"input": "ibe: IBE-1152"}' | python3 -m json.tool
   ```
   Expected: `comment_posted: true`, RCA text in `response`.

4. **Import simplified n8n workflow:**
   Open `http://localhost:5678` → Workflows → Import from file → `docs/n8n-rca-simple-workflow.json` → Activate.

5. **Test from n8n chat:**
   Type `ibe: IBE-1152` in the chat panel. Should get `✅ RCA posted to IBE-1152.` response and see the comment in Jira.
