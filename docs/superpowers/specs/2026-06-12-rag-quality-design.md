# RAG Response Quality — Design

**Goal:** Improve retrieval quality across all endpoints (`/chat`, `/chat/bedrock`, `/rca`) by combining multi-query fusion with cross-encoder reranking, giving the LLM a better-selected set of code chunks without changing the number of chunks it synthesises over.

**Architecture:** Two new components wire into `_get_engine` only. `QueryFusionRetriever` expands a single query into three variants and merges their results; `FlagEmbeddingReranker` scores the merged candidates and keeps the best twelve. All endpoints benefit automatically because they all call `_get_engine`.

**Tech Stack:** LlamaIndex `QueryFusionRetriever` (built-in), `llama-index-postprocessor-flag-embedding-reranker`, `BAAI/bge-reranker-base` (HuggingFace, CPU, ~280 MB).

---

## Problem

The current pipeline does a single cosine-similarity search with `similarity_top_k=12`. This fails in two ways:

- **Recall gap** — if the right chunk doesn't match the exact phrasing of the query, it isn't retrieved at all. Reranking can't recover a chunk that was never fetched.
- **Precision gap** — cosine similarity ranks by embedding proximity, not by how useful a chunk is for answering a specific question. Off-target chunks crowd out relevant ones.

The result is RCA analyses that miss key files and chat answers that synthesise over loosely related code.

---

## Design

### Data Flow

```
user query
  → LLM generates 2 additional query variants (same LLM as the endpoint: Ollama or Bedrock)
  → 3 parallel ChromaDB similarity searches  top_k=10 each
  → union + RECIPROCAL_RANK deduplication  (~20–30 unique chunks)
  → bge-reranker-base scores every (query, chunk) pair on CPU
  → top_n=12 chunks forwarded to CompactAndRefine + LLM
  → response
```

The final LLM call receives the same 12 chunks as today, but they are the best 12 out of ~30 candidates spanning three query phrasings.

### Components

#### 1. `QueryFusionRetriever`

LlamaIndex built-in (`llama_index.retrievers.QueryFusionRetriever`). No new package required.

- Wraps `svc["index"].as_retriever(similarity_top_k=10)`.
- `num_queries=3`: generates the original query plus two paraphrased variants using the passed `llm`.
- `mode=FUSION_MODES.RECIPROCAL_RANK`: rank-weighted deduplication — a chunk that appears in multiple variant result sets is ranked higher.
- `use_async=True`: the three ChromaDB searches run concurrently.

The `llm` parameter is whatever is passed to `_get_engine` — `local_llm` (Ollama) for `/chat`, `bedrock_llm` for `/chat/bedrock` and `/rca` — so query expansion uses the same model as synthesis.

#### 2. `FlagEmbeddingReranker`

Package: `llama-index-postprocessor-flag-embedding-reranker`.
Model: `BAAI/bge-reranker-base` (downloaded from HuggingFace on first use, ~280 MB, CPU inference).

A cross-encoder reads the query and each candidate chunk together in a single forward pass and produces a relevance score — far more accurate than cosine similarity, which embeds query and chunk independently and then compares vectors.

Created once as a module-level singleton (`_get_reranker()`) so the model is loaded from disk only on the first request.

```python
_reranker: FlagEmbeddingReranker | None = None

def _get_reranker() -> FlagEmbeddingReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-base", top_n=12)
    return _reranker
```

Plugs in as a `node_postprocessors` entry on `ContextChatEngine`.

#### 3. `_get_engine` refactor

`index.as_chat_engine()` creates its own internal retriever and does not accept a custom one. Switch to `ContextChatEngine.from_defaults(retriever=..., ...)` directly.

**Before:**
```python
"engine": svc["index"].as_chat_engine(
    chat_mode="context",
    llm=llm,
    memory=memory,
    similarity_top_k=12,
    system_prompt=svc["system_prompt"],
)
```

**After:**
```python
base_retriever = svc["index"].as_retriever(similarity_top_k=10)
fusion_retriever = QueryFusionRetriever(
    retrievers=[base_retriever],
    llm=llm,
    num_queries=3,
    mode=FUSION_MODES.RECIPROCAL_RANK,
    use_async=True,
)
"engine": ContextChatEngine.from_defaults(
    retriever=fusion_retriever,
    llm=llm,
    memory=memory,
    node_postprocessors=[_get_reranker()],
    system_prompt=svc["system_prompt"],
)
```

Everything else — session management, reindex endpoints, Jira RCA flow — is untouched.

---

## Dependencies

| Package | Purpose | New? |
|---------|---------|------|
| `llama-index-postprocessor-flag-embedding-reranker` | `FlagEmbeddingReranker` class | Yes |
| `BAAI/bge-reranker-base` | Cross-encoder model weights | Downloaded at runtime |

Add to `requirements.txt`:
```
llama-index-postprocessor-flag-embedding-reranker
```

---

## Model Persistence

`BAAI/bge-reranker-base` is cached at `~/.cache/huggingface` inside the container. Without a volume mount this is lost on every `docker compose up --build`, forcing a ~10 s re-download on first request.

Add a named volume in `docker-compose.yml`:

```yaml
volumes:
  hf_cache:

services:
  codebot:
    volumes:
      - hf_cache:/root/.cache/huggingface
```

---

## File Map

| File | Change |
|------|--------|
| `requirements.txt` | Add `llama-index-postprocessor-flag-embedding-reranker` |
| `runner.py` | Add imports; add `_reranker` singleton + `_get_reranker()`; refactor `_get_engine` |
| `docker-compose.yml` | Add `hf_cache` named volume and mount it at `/root/.cache/huggingface` |

---

## Latency Budget

| Step | Added latency |
|------|--------------|
| Query variant generation (LLM, parallel with retrieval) | ~1–2 s |
| 3× ChromaDB searches (async) | ~0.1 s extra vs. 1× |
| `bge-reranker-base` scoring 30 chunks on CPU | ~1–2 s |
| **Total additional latency** | **~2–4 s** |

Fits within the stated moderate tolerance (3–5 s acceptable).

---

## Out of Scope

- Knowledge graph / call-chain traversal (separate design)
- Auto-indexing on code push (separate design)
- Agentic RAG (separate design)
- Switching embedding model (not needed; `mxbai-embed-large` is sufficient)
