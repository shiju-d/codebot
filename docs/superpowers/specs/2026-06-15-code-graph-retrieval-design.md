# Code Graph Retrieval — Design

**Goal:** Improve retrieval accuracy across all endpoints (`/chat`, `/chat/bedrock`, `/rca`) by augmenting vector search with a static code knowledge graph. When vector search retrieves a file that is structurally related to the answer file, graph traversal surfaces the answer file regardless of vocabulary mismatch between the query and the code.

**Motivating failure:** Jira ticket CICO-134027 described "4-person rate missing extra adult price." Vector search retrieved `channex_test.rb` and reservation fixtures — but not `rate_exporter.rb`, where the bug actually lived. Adding a reporter hint "Related to channex rate exporter" fixed retrieval. The graph makes that hint unnecessary: `channex_test.rb` has a `REQUIRES` edge to `rate_exporter.rb`, so a 1-hop traversal surfaces the correct file automatically.

**Architecture:** At index time, tree-sitter parses every Ruby and TypeScript/JavaScript file to extract File, Class, and Method nodes plus structural edges (REQUIRES, DEFINES, HAS_METHOD, CALLS, INHERITS, INCLUDES). These are stored in Neo4j. At query time, a new `GraphExpansionPostprocessor` takes the file paths from vector search hits, queries Neo4j for 1-hop neighbours, fetches their chunks from ChromaDB, and merges them with the original hits before the existing reranker scores and trims to the final 12. All other pipeline components are unchanged.

---

## Problem

Pure vector similarity fails when query vocabulary and code location use different terms. The current pipeline — `QueryFusionRetriever` (3 query variants) + `FlagEmbeddingReranker` — improves recall and precision within the semantic similarity space, but cannot bridge a vocabulary gap where the right file simply never scores highly enough to be retrieved.

The structural relationships in code (require, inheritance, method calls) are vocabulary-independent: `channex_test.rb` requires `rate_exporter.rb` regardless of what words appear in a query. The graph captures those relationships.

---

## Data Flow

### Index Time

```
repo files
  → tree-sitter parse (Ruby + TS/JS)
  → extract nodes:  File, Class, Method
  → extract edges:  REQUIRES, DEFINES, HAS_METHOD, CALLS, INHERITS, INCLUDES
  → write to Neo4j  (MERGE — idempotent)
  → (existing) chunk + embed + write to ChromaDB
```

Graph build runs after ChromaDB indexing. Failure is non-fatal: the service starts with vector-only retrieval.

### Query Time

```
user query
  → QueryFusionRetriever → ~30 chunks from ChromaDB      (existing)
  → GraphExpansionPostprocessor (NEW):
      extract file_path from each chunk's metadata
      → Neo4j Cypher: 1-hop expand (REQUIRES + CALLS edges)
      → fetch chunks for expanded file paths from ChromaDB
      → merge + deduplicate (cap: +5 files, +15 chunks)
  → FlagEmbeddingReranker → top 12                        (existing)
  → LLM synthesis                                         (existing)
```

---

## Graph Schema

### Nodes

```
(:File   { path: str, service: str, language: str })
(:Class  { name: str, file_path: str, service: str })
(:Method { name: str, class_name: str, file_path: str, service: str,
           start_line: int, end_line: int })
```

### Edges

| Edge | From → To | Extracted From |
|------|-----------|----------------|
| `REQUIRES` | File → File | `require`/`require_relative`/`import` |
| `DEFINES` | File → Class | class/module declaration |
| `HAS_METHOD` | Class → Method | `def` / function definition |
| `CALLS` | Method → Method | call expressions (best-effort) |
| `INHERITS` | Class → Class | `< ParentClass` / `extends` |
| `INCLUDES` | Class → Class | Ruby `include` / `extend` |

`REQUIRES` and `DEFINES`/`HAS_METHOD` edges are reliable (static, unambiguous syntax). `CALLS` edges are best-effort: same-file and `self.method` calls are captured; cross-object dynamic dispatch in Ruby has gaps. The file-level `REQUIRES` graph is the reliable backbone; method-level edges add precision on top.

### Neo4j Constraints

```cypher
CREATE CONSTRAINT file_path_unique IF NOT EXISTS
  FOR (f:File) REQUIRE (f.path, f.service) IS UNIQUE;

CREATE CONSTRAINT method_unique IF NOT EXISTS
  FOR (m:Method) REQUIRE (m.file_path, m.name, m.class_name) IS UNIQUE;
```

---

## Graph Expansion Cypher (1 hop)

```cypher
UNWIND $seed_paths AS seed_path
MATCH (f:File {path: seed_path, service: $service})
OPTIONAL MATCH (f)-[:REQUIRES]->(dep:File)
OPTIONAL MATCH (caller:File)-[:REQUIRES]->(f)
OPTIONAL MATCH (f)-[:DEFINES]->(:Class)-[:HAS_METHOD]->(m:Method)
               -[:CALLS]->(target:Method)<-[:HAS_METHOD]-(:Class)
               <-[:DEFINES]-(target_file:File)
RETURN
  COLLECT(DISTINCT dep.path)      AS requires,
  COLLECT(DISTINCT caller.path)   AS required_by,
  COLLECT(DISTINCT target_file.path) AS call_targets
```

Results are merged and ranked: direct `REQUIRES` dependencies first, then `CALLS` targets, then reverse `REQUIRES` (callers). Capped at 5 additional files to prevent widely-required files (e.g., `application_controller.rb`) from flooding the context.

---

## Parsing

**tree-sitter is already installed** (`tree-sitter-languages` in `requirements.txt`).

### Ruby (`.rb`, `.rake`)

| Syntax | Graph output |
|--------|-------------|
| `require 'foo'` / `require_relative 'bar'` | `REQUIRES` edge (path resolved relative to file) |
| `include Foo` / `extend Bar` | `INCLUDES` edge |
| `class Foo < Bar` | `Class` node + `INHERITS` edge |
| `module Foo` | `Class` node (modules treated as classes) |
| `def method_name` | `Method` node |
| `foo.bar(...)` / `bar(...)` | `CALLS` edge (best-effort) |

### TypeScript / JavaScript (`.ts`, `.tsx`, `.js`, `.jsx`)

| Syntax | Graph output |
|--------|-------------|
| `import ... from '...'` | `REQUIRES` edge |
| `class Foo extends Bar` | `Class` node + `INHERITS` edge |
| `function foo` / `const foo = () =>` / class methods | `Method` node |
| `foo(...)` / `this.foo(...)` / `service.bar()` | `CALLS` edge (best-effort) |

**Files not covered** (`.yml`, `.json`, `.erb`) — no graph nodes built. Existing vector chunks for these files are still indexed and retrieved as today. No regression.

---

## New Files

### `graph.py`

Pure parsing and Neo4j write logic. No LlamaIndex dependency.

```python
def build_service_graph(svc: ServiceConfig, driver: neo4j.Driver) -> None
def clear_service_graph(service_name: str, driver: neo4j.Driver) -> None
def _parse_file(file_path: str, language: str) -> ParsedFile
def _parse_ruby(source: str, file_path: str) -> ParsedFile
def _parse_ts_js(source: str, file_path: str, language: str) -> ParsedFile
def _resolve_require(require_str: str, current_file: str, repo_root: str) -> str | None
```

`ParsedFile` is a plain dataclass holding lists of nodes and edges — no I/O, making parsers straightforward to unit test with fixture strings.

Graph writes use batched `MERGE` transactions (500 nodes/edges per transaction) to avoid memory spikes on large repos.

### `graph_retriever.py`

Pure Neo4j query logic. No LlamaIndex dependency.

```python
def expand_file_paths(
    seed_paths: list[str],
    service: str,
    driver: neo4j.Driver,
    max_files: int = 5,
) -> list[str]
```

Returns deduplicated expanded file paths. Scores by relationship type (REQUIRES > CALLS > REQUIRED_BY) before applying the cap.

### `graph_postprocessor.py`

LlamaIndex integration layer.

```python
class GraphExpansionPostprocessor(BaseNodePostprocessor):
    def __init__(
        self,
        service_name: str,
        vector_index: VectorStoreIndex,
        driver: neo4j.Driver,
        max_expanded_files: int = 5,
        max_expanded_chunks: int = 15,
    ): ...

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle,
    ) -> list[NodeWithScore]: ...
```

Failure mode: if `expand_file_paths` raises (Neo4j down, timeout), logs a warning and returns the original `nodes` unchanged. The pipeline continues with vector-only results.

---

## Changed Files

### `runner.py`

1. Add `neo4j.GraphDatabase.driver(NEO4J_URI, auth=...)` singleton at module level (like `bedrock_llm`).
2. In `_build_service_index`: after ChromaDB build, call `graph.build_service_graph(svc, neo4j_driver)` in a try/except (graph failure does not abort service startup).
3. In `_init_all_services` (after `_build_service_index` returns): create a `GraphExpansionPostprocessor` per service and store it in `services[service_name]["graph_postprocessor"]`. This is a per-service singleton — not re-created per session — analogous to how `_get_reranker()` returns the same reranker instance on every call.
4. In `_get_engine`: read `services[service_name]["graph_postprocessor"]` and pass it as the first entry in `node_postprocessors`, before `_get_reranker()`.
5. In `/reindex` and `/reindex/{service}`: call `graph.clear_service_graph` + `graph.build_service_graph` alongside the ChromaDB rebuild, then rebuild the `GraphExpansionPostprocessor` for the affected service.

### `docker-compose.yml`

Add Neo4j service and volume:

```yaml
services:
  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/codebot-secret
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

`codebot` container gets `NEO4J_URI=bolt://neo4j:7687` and `NEO4J_PASSWORD=codebot-secret` via `.env`.

### `requirements.txt`

Add:
```
neo4j
```

---

## Latency Budget

| Step | Added latency |
|------|--------------|
| Neo4j Cypher expansion query | ~5–20 ms |
| ChromaDB fetches for expanded files | ~10–50 ms |
| **Total additional query latency** | **~15–70 ms** |

Graph build time at index: ~5–15 s per service (tree-sitter is fast; batched MERGE).

---

## Testing

### Unit Tests (no Docker, no Neo4j)

| File | Covers |
|------|--------|
| `tests/test_graph_parser.py` | `_parse_ruby` / `_parse_ts_js` — fixture source strings → assert correct nodes and edges |
| `tests/test_graph_retriever.py` | `expand_file_paths` — mock Neo4j driver → assert correct Cypher paths returned and capped at `max_files` |
| `tests/test_graph_postprocessor.py` | `GraphExpansionPostprocessor` — mock `expand_file_paths` + mock vector index → assert expanded nodes merged and original nodes preserved |

### Integration Test (requires Neo4j — skipped unless `NEO4J_URI` set)

| File | Covers |
|------|--------|
| `tests/test_graph_integration.py` | Build graph for a small fixture repo mimicking rover-ifc structure → seed from `channex_test.rb` → assert `rate_exporter.rb` appears in expansion results |

### Existing Tests

All current tests (`test_api.py`, `test_jira.py`, `test_message.py`, `test_config.py`) are unaffected. The postprocessor is a new `node_postprocessors` entry that is mocked/absent in existing engine tests.

---

## Out of Scope

- **Multi-hop traversal** (depth > 1) — adds noise risk; revisit if 1-hop proves insufficient
- **Automatic graph updates on code push** — would require a file-watcher or git hook; separate design
- **Graph-based question answering** (Cypher generation from natural language) — separate design
- **Switching graph DB** — Neo4j 5 Community Edition is sufficient; no clustering needed for a single-node PoC
- **erb/yml/json structural parsing** — these file types have no call-graph semantics; vector chunks are sufficient
