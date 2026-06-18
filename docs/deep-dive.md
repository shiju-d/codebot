# codebot — Deep Dive

This document explains every part of codebot from first principles. It is written for someone who wants to understand the system well enough to explain it, extend it, or debug it — not just use it.

---

## 1. What problem does codebot solve?

Stayntouch has three large codebases: **ibe** (TypeScript, ~3 repos), **rover-ifc** (Ruby/Rails), and **pms** (Ruby/Rails). When a production bug is reported as a Jira ticket, an engineer has to:

1. Read the ticket and understand the symptom
2. Open the relevant codebase and navigate to the right area
3. Trace the call chain across files
4. Form a hypothesis about the root cause
5. Write a comment in Jira explaining it

This is time-consuming and requires deep codebase knowledge. codebot automates steps 2–5: it reads the Jira ticket, searches the codebase for relevant code, and uses an LLM to produce a structured root cause analysis, which it posts back to Jira.

It also works as a general-purpose code assistant: developers can ask questions about how a piece of functionality works, without having to navigate the code themselves.

---

## 2. How does it work at a high level?

codebot is a **Retrieval Augmented Generation (RAG)** system. The idea behind RAG is simple:

> Instead of asking an LLM to answer from its training data, first retrieve the relevant context from your own data, then give that context to the LLM along with the question.

This matters for code because an LLM trained on public data does not know your private codebase. Even if it did, codebases change constantly. RAG solves both problems: codebot retrieves the current, private source code and hands it to the LLM as context.

The high-level flow for every question is:

```
User question
      ↓
Search the codebase for relevant code chunks   ← "retrieval"
      ↓
Expand using the code structure graph
      ↓
Re-rank the chunks by relevance
      ↓
Send the top chunks + question to the LLM      ← "generation"
      ↓
LLM answer (grounded in actual code)
```

Each of these steps is a distinct component. The rest of this document explains each one.

---

## 3. Infrastructure overview

```
Your machine (host)
│
├── Ollama (port 11434)            ← Local LLM + embedding model
│
└── Docker
    ├── codebot (port 8000)        ← FastAPI app, Python
    │   ├── /app/chroma_db         ← ChromaDB vector index (volume: chroma_data)
    │   ├── /root/.cache/hf        ← Reranker model weights (volume: hf_cache)
    │   └── /repos/*               ← Source repos, read-only mounts
    │
    └── neo4j (port 7474/7687)     ← Code structure graph (volume: neo4j_data)
```

codebot communicates with:
- **Ollama** at `host.docker.internal:11434` — for the local LLM and embeddings
- **Neo4j** at `neo4j:7687` (internal Docker network) — for graph queries
- **AWS Bedrock** (HTTPS) — for Claude when using `/chat/bedrock` or `/rca`
- **Jira** (HTTPS) — to fetch ticket details and post comments

---

## 4. Services and configuration

Everything in codebot is organized around the concept of a **service** — a named codebase configuration. Services are defined in `services.yaml`:

```yaml
services:
  - name: rover-ifc
    system_prompt: |
      You are an expert in the Stayntouch Rover IFC application...
    file_extensions: [.rb, .erb, .rake, .yml, .json]
    repos:
      - /repos/rover-ifc
```

`config.py` loads this into a `ServiceConfig` dataclass. The `name` is the identifier used in all API calls. The `system_prompt` shapes the LLM's persona and grounding rules — it tells Claude which layer to look in, how to reason about the architecture, and importantly, to only reference file paths that actually appear in the retrieved context (to prevent hallucination).

At runtime, each service gets its own:
- **ChromaDB collection** (`{name}_codebase`) — the vector index of its source files
- **Neo4j subgraph** — nodes and edges scoped with `service: {name}` property
- **Session store** — an in-memory dict of active chat sessions

---

## 5. Startup: building the index and graph

When the Docker container starts, `app.py` runs `engine.init_all_services()` in a background thread (via `asyncio.to_thread`). This is run in a thread rather than blocking the async event loop because it's CPU and I/O intensive and takes minutes on first run.

For each service, two things happen:

### 5a. Building the vector index (indexer.py)

The goal is to split every source file into overlapping chunks and store them as **embeddings** in ChromaDB.

**What is an embedding?** An embedding is a list of numbers (a vector) that represents the semantic meaning of a piece of text. Similar texts have vectors that are mathematically close together. By converting both source code and user questions into embeddings, we can find code that is semantically similar to the question — even if it uses different words.

The embedding model is **mxbai-embed-large**, running in Ollama on the host. Every chunk of code gets embedded at index time. At query time, the question also gets embedded, and ChromaDB does a fast approximate nearest-neighbour search to find the closest chunks.

**Chunking strategy:** Different file types are chunked differently:

- `.ts`, `.tsx`, `.js`, `.jsx`, `.rb`, `.rake` — `CodeSplitter` from LlamaIndex, backed by tree-sitter grammar. It understands the syntax and tries to split at function/class boundaries. Parameters: 50 lines per chunk, 15 lines of overlap, max 2000 characters.
- Everything else (`.yml`, `.json`, `.erb`) — `TokenTextSplitter`, a simpler approach that splits on token count (600 tokens, 100 overlap) with separator hints.

If `CodeSplitter` fails (malformed syntax), it falls back to `TokenTextSplitter`.

Each chunk keeps a `file_path` metadata field (stripped of the `/repos/` mount prefix so it reads as `rover-ifc/app/models/rate.rb` rather than `/repos/rover-ifc/app/models/rate.rb`). This is important — it's how the response tells you which file to look at.

**Caching:** If the ChromaDB collection already has chunks (from a previous run), the index is loaded from disk rather than rebuilt. This is why subsequent container starts are fast. The index only rebuilds when you call `/reindex`.

### 5b. Building the code graph (graph.py)

The graph captures the **structural relationships** between files, classes, and methods. It lives in Neo4j. The motivation is explained in section 8; this section covers how it's built.

`graph.py` walks every supported source file, parses it with **tree-sitter**, and writes the results to Neo4j.

**What is tree-sitter?** It is a parser library that produces an Abstract Syntax Tree (AST) — a structured representation of code that understands syntax, not just text. For example, it can tell you "this is a method definition named `push_rate`, it starts on line 42, and it's inside a class called `RateExporter`." The library used here is `tree-sitter-language-pack`, which bundles grammars for Ruby, TypeScript, JavaScript, and dozens of other languages.

The parser walks the AST recursively using `_walk_ruby()` or `_walk_ts_js()`, collecting:

- **Classes** with their names and superclass (for INHERITS edges)
- **Methods** with their names, owning class, and line numbers
- **`require` / `require_relative` / `import` calls** — the file dependencies

One important limitation: **Rails uses Zeitwerk autoloading**. Classes are loaded automatically by name convention, not by explicit `require` calls. rover-ifc and pms both use Rails, so most class-to-class dependencies are invisible to the parser. The graph captures only the explicit `require`/`require_relative`/`import` statements it can see in the source.

**Thread safety note:** tree-sitter's Python bindings (pyo3) do not allow Parser objects to be shared across threads. Since `init_all_services` runs in a thread pool, a naive module-level parser would panic. The fix is `threading.local()` — each thread gets its own parser instance, created on first use in that thread.

**Writing to Neo4j:** After parsing, everything is written in batches of 500 using Cypher's `UNWIND ... MERGE` pattern. `MERGE` means "create if not exists, update if exists" — so re-running the graph build on the same service is safe (though you should use `/reindex` which clears first). Constraints on `(path, service)` and `(file_path, name, class_name)` ensure uniqueness.

---

## 6. The retrieval pipeline

Every chat message and every RCA request goes through this exact pipeline. Understanding it is the key to understanding codebot.

```
User question
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  QueryFusionRetriever                               │
│  ─────────────────────────────────────────────────  │
│  1. LLM generates 2 extra query variants            │
│  2. All 3 queries run against ChromaDB (top 10 each)│
│  3. 30 results merged via Reciprocal Rank Fusion    │
└─────────────────────────────────────────────────────┘
      │ 30 nodes
      ▼
┌─────────────────────────────────────────────────────┐
│  GraphExpansionPostprocessor                        │
│  ─────────────────────────────────────────────────  │
│  1. Extract unique file paths from the 30 nodes     │
│  2. Query Neo4j for 1-hop neighbours                │
│     (files these files require, and their callers)  │
│  3. Fetch chunks for the new paths from ChromaDB    │
│  4. Append as new nodes with score=0                │
└─────────────────────────────────────────────────────┘
      │ ~35–45 nodes
      ▼
┌─────────────────────────────────────────────────────┐
│  FlagEmbeddingReranker                              │
│  ─────────────────────────────────────────────────  │
│  Cross-encoder scores every node against the query  │
│  Keeps top 12                                       │
└─────────────────────────────────────────────────────┘
      │ 12 nodes
      ▼
┌─────────────────────────────────────────────────────┐
│  ContextChatEngine → LLM                           │
│  ─────────────────────────────────────────────────  │
│  Formats context + system prompt + chat history     │
│  Calls Ollama or Bedrock                            │
│  Returns answer                                     │
└─────────────────────────────────────────────────────┘
```

### 6a. QueryFusionRetriever — why three queries?

A single vector search has a vocabulary problem. If a Jira ticket says "nightly rate synchronisation fails for OTA channels" but the actual code uses the word `push_rate`, a single embedding search may not bridge that gap.

`QueryFusionRetriever` (from LlamaIndex) asks the LLM to generate two alternative phrasings of the question, then runs all three against ChromaDB in parallel (because `use_async=True`). The LLM used for variant generation is the **same LLM that handles the final response** — Ollama for `/chat`, Claude on Bedrock for `/chat/bedrock` and `/rca`. This is because `get_engine()` passes its `llm` argument directly into `QueryFusionRetriever(llm=llm, ...)`. The three result sets are merged using **Reciprocal Rank Fusion (RRF)**.

**What is RRF?** It is a rank-merging algorithm. Each document gets a score of `1 / (rank + 60)` from each query, and the scores are summed. Documents that appear consistently high across multiple queries score higher than documents that only match one. It does not depend on the raw similarity scores, only on ranks — which makes it robust to score scale differences. The merged list is capped at `similarity_top_k=30`.

### 6b. GraphExpansionPostprocessor — why the graph?

Vector search finds code that is **semantically similar** to the question. But some important files will not match the question semantically at all — they are important because they are *called by* a matching file, or they *call into* a matching file.

Example: a ticket about rate calculation might vector-match `rate_manager.rb`. But the actual bug might be in `ota_rate_sync.rb`, which `rate_manager.rb` requires. The symptom description never mentions `ota_rate_sync`, so it will not appear in the top-30.

The `GraphExpansionPostprocessor` runs a Cypher query (`graph_retriever.py`) against Neo4j:

```cypher
MATCH (f:File)
WHERE f.path IN $seed_paths AND f.service = $service
OPTIONAL MATCH (f)-[:REQUIRES]->(dep:File)        -- files this file depends on
OPTIONAL MATCH (caller:File)-[:REQUIRES]->(f)     -- files that depend on this file
...
RETURN COLLECT(DISTINCT dep.path) AS requires,
       COLLECT(DISTINCT caller.path) AS required_by, ...
```

It returns up to 5 new file paths (dependencies ranked before callers). It then fetches all ChromaDB chunks for those paths and appends them to the node list with a score of 0.0 (they rank last — the reranker will promote the ones that are actually relevant).

This is a **1-hop** expansion. It does not recurse further.

### 6c. FlagEmbeddingReranker — why rerank?

Vector search uses a **bi-encoder**: both the query and the document are encoded independently, and similarity is measured by cosine distance between the two vectors. It is fast but approximate — the two encodings have no direct interaction, so it can miss subtle relevance signals.

A **cross-encoder** (what `FlagEmbeddingReranker` uses) sees the query and the document together in a single forward pass. It can detect relevance patterns that require understanding both sides simultaneously. It is far more accurate, but much slower — too slow to run on thousands of candidates.

The pipeline uses the best of both: fast bi-encoder retrieval to get to ~35 candidates, then accurate cross-encoder reranking to select the top 12. The model is `BAAI/bge-reranker-base`, downloaded once and cached in the `hf_cache` Docker volume.

The reranker is a singleton (`get_reranker()` with a double-checked lock) because loading the model is expensive and should happen exactly once.

### 6d. ContextChatEngine — memory and the LLM call

`ContextChatEngine` (from LlamaIndex) wraps the retrieval pipeline and the LLM. It does three things:

1. **Context formatting:** Takes the top-12 nodes and formats them into a context block that gets prepended to the LLM prompt.
2. **Memory:** Maintains a `ChatMemoryBuffer` (4096 token limit) per session. Each turn, the recent chat history is included in the prompt so the LLM can answer follow-up questions.
3. **LLM call:** Sends the formatted prompt to either Ollama (`local_llm`) or Bedrock (`bedrock_llm`).

---

## 7. Sessions

A session represents a continuing conversation. Sessions are stored in an in-memory dict:

```
services[service_name]["sessions"][llm_key][session_id] = {
    "memory": ChatMemoryBuffer,
    "engine": ContextChatEngine,
}
```

The `session_id` comes from the API request. In the n8n workflows, it is set to the `$json.sessionId` value that n8n's Chat Trigger generates per browser window.

Each (service, LLM backend) pair has its own `OrderedDict` capped at 100 sessions. When the cap is hit, the oldest session is evicted (the `OrderedDict` is used precisely because it maintains insertion order, enabling FIFO eviction with `popitem(last=False)`).

Sessions survive container restarts only as long as the container is running. They are not persisted to disk — if you restart the container, all chat history is lost. The ChromaDB index and Neo4j graph survive restarts because they are in volumes.

---

## 8. The RCA flow end to end

The `/rca` endpoint combines several concerns. Here is exactly what happens when you call it with `rover-ifc: CICO-134027`:

1. **Parse input** (`jira.py::parse_rca_input`): Split on the first colon to get `service=rover-ifc`. Use a regex to find the Jira key `CICO-134027`. Any text after the key becomes `additional_context`.

2. **Fetch the Jira issue** (`jira.py::fetch_jira_issue`): HTTP GET to `JIRA_BASE_URL/rest/api/3/issue/CICO-134027` with Basic auth (email + API token). Returns the full issue JSON.

3. **Extract description** (`jira.py::extract_adf_text`): Jira descriptions are in **Atlassian Document Format (ADF)** — a nested JSON tree, not plain text. `extract_adf_text` walks the tree recursively, collecting all `text` leaf nodes.

4. **Build the RCA message** (`jira.py::build_rca_message`): Assembles a structured prompt:
   ```
   rover-ifc: You are performing Root Cause Analysis...
   Ticket: CICO-134027
   Summary: <summary from Jira>
   Description: <extracted plain text>
   IMPORTANT: Base your analysis only on retrieved code snippets...
   Answer: 1. Which files? 2. Where is the root cause? 3. What is the fix?
   ```
   Note the message starts with `rover-ifc:` — it goes through the same `parse_message` routing as a normal chat message.

5. **Run the full retrieval pipeline**: The assembled message is passed to `engine.get_engine()` which creates a `ContextChatEngine` with the full pipeline (fusion retriever → graph expansion → reranker). `achat(message)` runs it.

6. **Convert Markdown to Jira wiki markup** (`jira.py::md_to_jira`): The LLM response is Markdown. Jira comments use a different format (Jira wiki markup). The converter handles headers (`## Title` → `h2. Title`), bold, code blocks (` ```ruby ``` ` → `{code:ruby}...{code}`), inline code, bullet lists, and horizontal rules.

7. **Post the comment** (`jira.py::post_jira_comment`): HTTP POST to `JIRA_BASE_URL/rest/api/2/issue/CICO-134027/comment`. Uses API v2 here (v3 uses ADF; v2 accepts plain wiki markup as a string).

8. **Return the response**: The API returns the response text, source file list, and `comment_posted: true`.

---

## 9. The LLM setup and the Bedrock workaround

### Two LLMs

`llms.py` sets up two LLM clients at module load time:

- **`local_llm`**: Ollama client pointing to `host.docker.internal:11434`, using model `qwen2.5-coder:7b`. Available immediately, no cloud credentials. Good for quick questions. Used by `/chat`.

- **`bedrock_llm`**: AWS Bedrock Converse client using a Claude model (configurable via `BEDROCK_MODEL_ID`). Only created if `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are present; otherwise it is `None`. Used by `/chat/bedrock` and `/rca`.

### The Bedrock metadata patch

Stayntouch uses a cross-region inference profile whose model ID starts with `global.` — a prefix LlamaIndex does not recognise. This causes LlamaIndex to assume a ~4091 token context window, then crash trying to reserve 4096 tokens for the response (`4091 - 4096 = -5`). The fix is a class-level property replacement applied at module load time so every `BedrockConverse` instance is affected, including ones LlamaIndex creates internally.

See **[docs/bedrock-context-window-patch.md](bedrock-context-window-patch.md)** for the full explanation: why the constructor argument alone is not enough, how the patch works, why `num_output=2048` is used, and what happens if the patch itself fails.

---

## 10. Module structure and data flow

```
.env + services.yaml
        │
        ▼
   config.py              Loads ServiceConfig dataclasses
        │
        ▼
   llms.py                Creates LLM clients, embed model, Neo4j driver, reranker
        │
        ├──► indexer.py   Builds ChromaDB vector index per service
        │
        ├──► graph.py     Parses source files with tree-sitter → Neo4j
        │
        └──► engine.py    Assembles and caches ContextChatEngine per session
                │
                └──► graph_postprocessor.py + graph_retriever.py
                         Neo4j expansion between retrieval and reranking

   app.py                 FastAPI routes → delegates to engine.py, jira.py
   jira.py                Jira API calls, text extraction, markup conversion
   message.py             Parses "service: message" prefix
```

There are no circular imports. The dependency direction is always:
`app → engine → (llms, indexer, graph, graph_postprocessor) → config`

---

## 11. What each file does — one paragraph each

**`app.py`** — The FastAPI application. Defines the HTTP API and the startup sequence. Handles input validation (checking service prefix, checking LLM availability), calls `engine.get_engine()` to get a chat engine, calls `achat()`, and formats the response. For RCA, it additionally calls into `jira.py` to fetch the ticket and post the comment. No business logic lives here.

**`engine.py`** — The service registry and session manager. Holds the `services` dict that maps service names to their index, postprocessors, and session stores. `get_engine()` is the key function: given a session ID, service name, and LLM, it returns the `ContextChatEngine` for that session (creating it if it doesn't exist). `init_all_services()`, `reindex_all_services()`, and `reindex_one_service()` manage the lifecycle of services.

**`llms.py`** — All singleton objects that are expensive to create: the Ollama LLM, the Bedrock LLM, the embedding model (registered globally on `Settings.embed_model`), the Neo4j driver, and the reranker. Module-level execution sets them up at import time. The Bedrock metadata patch lives here.

**`indexer.py`** — Builds and loads the ChromaDB vector index for a service. Reads source files, splits them into chunks using language-appropriate splitters, embeds them, and stores them in ChromaDB. If the collection already has data, returns immediately without re-indexing.

**`graph.py`** — Parses source files with tree-sitter and writes the code structure graph to Neo4j. Contains two AST walkers: `_walk_ruby` and `_walk_ts_js`. Each walker traverses the syntax tree and records classes, methods, require/import statements, and (for Ruby) include/extend calls. The collected data is batch-written to Neo4j.

**`graph_postprocessor.py`** — A LlamaIndex `BaseNodePostprocessor` that sits in the pipeline between `QueryFusionRetriever` and `FlagEmbeddingReranker`. It takes the vector search results, expands them via the Neo4j graph, fetches additional chunks from ChromaDB, and passes the combined set to the reranker.

**`graph_retriever.py`** — A single Cypher query that takes a list of seed file paths and returns file paths reachable in one hop: direct dependencies (REQUIRES), files that depend on the seeds (required_by), and method call targets (CALLS — currently empty, as CALLS edges are not written). Returns up to 5 paths.

**`config.py`** — `ServiceConfig` dataclass and `load_services()`. No logic.

**`jira.py`** — All Jira-related code: parsing the `service: TICKET-123` input, extracting plain text from ADF, building the RCA prompt, converting Markdown to Jira wiki markup, fetching issues, posting comments.

**`message.py`** — Parses `"service: message"` into `(service_name, message)`. Three lines of logic.

---

## 12. Known limitations

**Rails Zeitwerk autoloading:** rover-ifc and pms are Rails apps. Rails loads classes by name convention (`require` is not needed). The graph only captures explicit `require`/`require_relative` calls, so most class-to-class dependencies in these services are invisible to the graph expansion step. This means graph expansion is more effective for ibe (TypeScript uses explicit `import`) than for the Ruby services.

**Stale index:** The ChromaDB index and Neo4j graph are not automatically updated when source code changes. After pulling new commits, you must call `POST /reindex/{service}` to rebuild. If you don't, codebot answers based on stale code.

**Session memory is ephemeral:** All chat sessions live in memory and are lost on container restart. This is intentional (keeps the design simple) but means you cannot resume a conversation after a restart.

**CALLS edges are not built:** The graph builder collects call relationships but does not write them to Neo4j. Writing them was found to hang on large codebases (pms has ~30,000 methods; matching by method name without a dedicated index is a full table scan). REQUIRES edges are sufficient for the expansion use case.

**Local LLM quality:** `qwen2.5-coder:7b` is a 7-billion-parameter model. It is capable for straightforward questions but will struggle with complex multi-file reasoning. For RCA and detailed analysis, use Bedrock (`/chat/bedrock` or `/rca`).

---

## 13. How to extend codebot

### Adding a new service

1. Check out the repo to `$REPO_ROOT/{name}`
2. Add a volume mount in `docker-compose.yml`
3. Add a service entry in `services.yaml` with a descriptive `system_prompt`
4. Restart the container or call `POST /reindex/{name}`

### Adding a new language to the code graph

In `graph.py`, add the extension to `_EXT_TO_LANGUAGE` and write a `_parse_{language}` function using tree-sitter-language-pack. The walk function follows the same pattern as `_walk_ruby` or `_walk_ts_js`.

### Changing the LLM

Set `BEDROCK_MODEL_ID` in `.env` to any model supported by the Bedrock Converse API. The metadata patch in `llms.py` ensures LlamaIndex treats any Bedrock model as having a 200k context window.

### Changing retrieval parameters

- `similarity_top_k=10` on `as_retriever()` — how many chunks per query variant
- `similarity_top_k=30` on `QueryFusionRetriever` — total after fusion
- `num_queries=3` — how many query variants (including the original)
- `max_expanded_files=5` on `GraphExpansionPostprocessor` — Neo4j hop limit
- `top_n=12` on `FlagEmbeddingReranker` — final context window size

These live in `engine.py::get_engine()` and `engine.py::_make_service_entry()`.
