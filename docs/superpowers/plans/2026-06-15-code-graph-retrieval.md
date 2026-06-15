# Code Graph Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Augment vector search with a static code knowledge graph so that when retrieval finds a file that is structurally related to the answer (via require/import/call edges), 1-hop Neo4j traversal surfaces the answer file automatically — without requiring extra context from the reporter.

**Architecture:** At index time, tree-sitter parses every Ruby and TS/JS file to extract File/Class/Method nodes and structural edges (REQUIRES, DEFINES, HAS_METHOD, CALLS, INHERITS, INCLUDES), written to Neo4j. At query time, `GraphExpansionPostprocessor` takes file paths from vector search hits, expands 1 hop via Neo4j, fetches extra chunks from ChromaDB, and passes the combined set to the existing `FlagEmbeddingReranker`. Three new files (`graph.py`, `graph_retriever.py`, `graph_postprocessor.py`) plus changes to `runner.py`, `docker-compose.yml`, and `requirements.txt`.

**Tech Stack:** Neo4j 5 (new container), `neo4j` Python driver (new dep), `tree-sitter-languages` (already installed), LlamaIndex `BaseNodePostprocessor`, ChromaDB metadata filters.

---

## File Map

| File | Action |
|------|--------|
| `docker-compose.yml` | Add `neo4j` service + `neo4j_data` volume; add `NEO4J_URI`/`NEO4J_PASSWORD` env vars to codebot |
| `requirements.txt` | Add `neo4j` |
| `.env.example` | Add `NEO4J_URI`, `NEO4J_PASSWORD` |
| `graph.py` | **Create** — ParsedFile dataclass, Ruby parser, TS/JS parser, path resolution, Neo4j write |
| `graph_retriever.py` | **Create** — 1-hop Cypher expansion, returns sorted+capped file paths |
| `graph_postprocessor.py` | **Create** — `GraphExpansionPostprocessor(BaseNodePostprocessor)` |
| `runner.py` | **Modify** — Neo4j driver singleton, `_init_all_services`, `_get_engine`, reindex endpoints |
| `tests/test_graph_parser.py` | **Create** — unit tests for `_parse_ruby` and `_parse_ts_js` |
| `tests/test_graph_retriever.py` | **Create** — unit tests for `expand_file_paths` (mocked driver) |
| `tests/test_graph_postprocessor.py` | **Create** — unit tests for `GraphExpansionPostprocessor` (mocked) |
| `tests/test_graph_integration.py` | **Create** — integration test (skipped unless `NEO4J_URI` set) |

---

## Task 1: Infrastructure — Neo4j container, driver dep, env vars

**Files:**
- Modify: `docker-compose.yml`
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: Add Neo4j service to docker-compose.yml**

Replace the entire contents of `docker-compose.yml` with:

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

  codebot:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./services.yaml:/app/services.yaml:ro
      - chroma_data:/app/chroma_db
      - hf_cache:/root/.cache/huggingface
      - ${REPO_ROOT}/ibe-api:/repos/ibe-api:ro
      - ${REPO_ROOT}/ibe-frontend:/repos/ibe-frontend:ro
      - ${REPO_ROOT}/ibe-admin:/repos/ibe-admin:ro
      - ${REPO_ROOT}/rover-ifc:/repos/rover-ifc:ro
      - ${REPO_ROOT}/pms:/repos/pms:ro
    env_file: .env
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - neo4j

volumes:
  chroma_data:
  hf_cache:
  neo4j_data:
```

- [ ] **Step 2: Add neo4j to requirements.txt**

Append one line to `requirements.txt`:

```
neo4j
```

Final file:
```
llama-index
tree-sitter-languages
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
llama-index-postprocessor-flag-embedding-reranker
neo4j
```

- [ ] **Step 3: Update .env.example**

Replace the entire contents of `.env.example` with:

```
REPO_ROOT=<path_to_your_repos>
AWS_ACCESS_KEY_ID=<your_aws_access_key_id>
AWS_SECRET_ACCESS_KEY=<your_aws_secret_access_key>
BEDROCK_MODEL_ID=<your_bedrock_model_id>

# Jira (required for POST /rca)
JIRA_BASE_URL=https://stayntouch.atlassian.net
JIRA_EMAIL=you@stayntouch.com
JIRA_API_TOKEN=<your-jira-api-token>

# Neo4j (graph retrieval)
NEO4J_URI=bolt://neo4j:7687
NEO4J_PASSWORD=codebot-secret
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml requirements.txt .env.example
git commit -m "feat: add Neo4j container and neo4j driver dep for code graph retrieval"
```

---

## Task 2: graph.py — data classes and Ruby parser

**Files:**
- Create: `graph.py`

- [ ] **Step 1: Create graph.py with data classes and Ruby parser**

Create `/Users/shijudevarajan/Codebase/poc/codebot/graph.py` with the full contents below.

```python
import os
from dataclasses import dataclass, field
from typing import Optional

import neo4j
from tree_sitter_languages import get_parser

from config import ServiceConfig


# ---------------------------------------------------------------------------
# Internal data classes — represent parsed graph elements before writing to Neo4j
# ---------------------------------------------------------------------------

@dataclass
class _RequireEdge:
    from_path: str      # rel_path of the file doing the require
    require_str: str    # raw string from require/import call
    is_relative: bool   # True for require_relative / relative TS import


@dataclass
class _ClassNode:
    name: str
    parent: Optional[str]   # superclass name, or None


@dataclass
class _IncludeEdge:
    class_name: str
    module_name: str


@dataclass
class _MethodNode:
    name: str
    class_name: str     # '__module__' for top-level functions
    start_line: int
    end_line: int


@dataclass
class _CallEdge:
    from_class: str
    from_method: str
    to_method: str      # best-effort: just the method name, no class


@dataclass
class ParsedFile:
    file_path: str      # rel_path (stripped of /repos/ prefix)
    service: str
    language: str
    requires: list[_RequireEdge] = field(default_factory=list)
    classes: list[_ClassNode] = field(default_factory=list)
    includes: list[_IncludeEdge] = field(default_factory=list)
    methods: list[_MethodNode] = field(default_factory=list)
    calls: list[_CallEdge] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Ruby tree-sitter helpers
# ---------------------------------------------------------------------------

def _ruby_node_name(node) -> Optional[str]:
    name_child = node.child_by_field_name('name')
    if not name_child:
        return None
    if name_child.type == 'constant':
        return name_child.text.decode('utf-8')
    if name_child.type == 'scope_resolution':
        right = name_child.child_by_field_name('name') or name_child.children[-1]
        return right.text.decode('utf-8')
    return None


def _ruby_superclass_name(class_node) -> Optional[str]:
    sc = class_node.child_by_field_name('superclass')
    if not sc:
        return None
    for child in sc.children:
        if child.type == 'constant':
            return child.text.decode('utf-8')
        if child.type == 'scope_resolution':
            right = child.child_by_field_name('name') or child.children[-1]
            return right.text.decode('utf-8')
    return None


def _ruby_method_name_from_def(node) -> Optional[str]:
    name_child = node.child_by_field_name('name')
    if name_child and name_child.type in ('identifier', 'operator', 'constant'):
        return name_child.text.decode('utf-8')
    return None


def _ruby_call_method_name(call_node) -> Optional[str]:
    method_child = call_node.child_by_field_name('method')
    if method_child and method_child.type == 'identifier':
        return method_child.text.decode('utf-8')
    return None


def _ruby_string_arg(call_node) -> Optional[str]:
    args = call_node.child_by_field_name('arguments')
    if not args:
        return None
    for child in args.children:
        if child.type == 'string':
            content = child.child_by_field_name('string_content')
            if not content:
                content = next((c for c in child.children if c.type == 'string_content'), None)
            if content:
                return content.text.decode('utf-8')
    return None


def _ruby_const_arg(call_node) -> Optional[str]:
    args = call_node.child_by_field_name('arguments')
    if not args:
        return None
    for child in args.children:
        if child.type == 'constant':
            return child.text.decode('utf-8')
        if child.type == 'scope_resolution':
            return child.text.decode('utf-8')
    return None


def _walk_ruby(node, parsed: ParsedFile, class_stack: list, method_stack: list) -> None:
    t = node.type

    if t in ('class', 'module'):
        name = _ruby_node_name(node)
        if name:
            parent = _ruby_superclass_name(node) if t == 'class' else None
            parsed.classes.append(_ClassNode(name=name, parent=parent))
            class_stack.append(name)
            for child in node.children:
                _walk_ruby(child, parsed, class_stack, method_stack)
            class_stack.pop()
        return

    if t in ('method', 'singleton_method') and class_stack:
        mname = _ruby_method_name_from_def(node)
        if mname:
            parsed.methods.append(_MethodNode(
                name=mname,
                class_name=class_stack[-1],
                start_line=node.start_point[0],
                end_line=node.end_point[0],
            ))
            method_stack.append(mname)
            for child in node.children:
                _walk_ruby(child, parsed, class_stack, method_stack)
            method_stack.pop()
        return

    if t == 'call':
        mname = _ruby_call_method_name(node)
        if mname in ('require', 'require_relative'):
            req_str = _ruby_string_arg(node)
            if req_str:
                parsed.requires.append(_RequireEdge(
                    from_path=parsed.file_path,
                    require_str=req_str,
                    is_relative=(mname == 'require_relative'),
                ))
        elif mname in ('include', 'extend') and class_stack:
            module_name = _ruby_const_arg(node)
            if module_name:
                parsed.includes.append(_IncludeEdge(
                    class_name=class_stack[-1],
                    module_name=module_name,
                ))
        elif mname and class_stack and method_stack:
            parsed.calls.append(_CallEdge(
                from_class=class_stack[-1],
                from_method=method_stack[-1],
                to_method=mname,
            ))

    for child in node.children:
        _walk_ruby(child, parsed, class_stack, method_stack)


def _parse_ruby(source: str, file_path: str, service: str) -> ParsedFile:
    parser = get_parser('ruby')
    tree = parser.parse(source.encode('utf-8'))
    parsed = ParsedFile(file_path=file_path, service=service, language='ruby')
    _walk_ruby(tree.root_node, parsed, [], [])
    return parsed
```

- [ ] **Step 2: Commit**

```bash
git add graph.py
git commit -m "feat: add graph.py — ParsedFile dataclasses and Ruby parser"
```

---

## Task 3: graph.py — TS/JS parser and path resolution helpers

**Files:**
- Modify: `graph.py`

- [ ] **Step 1: Append TS/JS parser and path resolution to graph.py**

Open `graph.py`. After the `_parse_ruby` function (end of file), append the following:

```python

# ---------------------------------------------------------------------------
# TypeScript / JavaScript tree-sitter helpers
# ---------------------------------------------------------------------------

def _ts_class_name(class_node) -> Optional[str]:
    name_child = class_node.child_by_field_name('name')
    if name_child and name_child.type in ('type_identifier', 'identifier'):
        return name_child.text.decode('utf-8')
    return None


def _ts_superclass_name(class_node) -> Optional[str]:
    for child in class_node.children:
        if child.type == 'class_heritage':
            for heritage_child in child.children:
                if heritage_child.type == 'extends_clause':
                    for ext_child in heritage_child.children:
                        if ext_child.type in ('identifier', 'type_identifier'):
                            return ext_child.text.decode('utf-8')
    return None


def _walk_ts_js(node, parsed: ParsedFile, class_stack: list, method_stack: list) -> None:
    t = node.type

    if t == 'import_statement':
        source_node = node.child_by_field_name('source')
        if source_node:
            frag = next(
                (c for c in source_node.children if c.type == 'string_fragment'),
                None,
            )
            if frag:
                import_str = frag.text.decode('utf-8')
                parsed.requires.append(_RequireEdge(
                    from_path=parsed.file_path,
                    require_str=import_str,
                    is_relative=import_str.startswith('.'),
                ))
        return

    if t in ('class_declaration', 'abstract_class_declaration'):
        class_name = _ts_class_name(node)
        if class_name:
            parent = _ts_superclass_name(node)
            parsed.classes.append(_ClassNode(name=class_name, parent=parent))
            class_stack.append(class_name)
            for child in node.children:
                _walk_ts_js(child, parsed, class_stack, method_stack)
            class_stack.pop()
        return

    if t == 'method_definition' and class_stack:
        name_child = node.child_by_field_name('name')
        if name_child:
            mname = name_child.text.decode('utf-8')
            parsed.methods.append(_MethodNode(
                name=mname,
                class_name=class_stack[-1],
                start_line=node.start_point[0],
                end_line=node.end_point[0],
            ))
            method_stack.append(mname)
            for child in node.children:
                _walk_ts_js(child, parsed, class_stack, method_stack)
            method_stack.pop()
        return

    if t == 'function_declaration':
        name_child = node.child_by_field_name('name')
        if name_child:
            fname = name_child.text.decode('utf-8')
            cls = class_stack[-1] if class_stack else '__module__'
            parsed.methods.append(_MethodNode(
                name=fname,
                class_name=cls,
                start_line=node.start_point[0],
                end_line=node.end_point[0],
            ))
            method_stack.append(fname)
            for child in node.children:
                _walk_ts_js(child, parsed, class_stack, method_stack)
            method_stack.pop()
        return

    if t == 'call_expression' and class_stack and method_stack:
        func_child = node.child_by_field_name('function')
        if func_child:
            if func_child.type == 'identifier':
                parsed.calls.append(_CallEdge(
                    from_class=class_stack[-1],
                    from_method=method_stack[-1],
                    to_method=func_child.text.decode('utf-8'),
                ))
            elif func_child.type == 'member_expression':
                prop = func_child.child_by_field_name('property')
                if prop:
                    parsed.calls.append(_CallEdge(
                        from_class=class_stack[-1],
                        from_method=method_stack[-1],
                        to_method=prop.text.decode('utf-8'),
                    ))

    for child in node.children:
        _walk_ts_js(child, parsed, class_stack, method_stack)


def _parse_ts_js(source: str, file_path: str, service: str, language: str) -> ParsedFile:
    parser = get_parser(language)
    tree = parser.parse(source.encode('utf-8'))
    parsed = ParsedFile(file_path=file_path, service=service, language=language)
    _walk_ts_js(tree.root_node, parsed, [], [])
    return parsed


# ---------------------------------------------------------------------------
# Path resolution — converts require strings to absolute file paths
# ---------------------------------------------------------------------------

def _resolve_require(
    require_str: str,
    current_abs_path: str,
    is_relative: bool,
    repo_roots: list[str],
) -> Optional[str]:
    """Resolve a Ruby require string to an absolute path, or None if not found."""
    if is_relative:
        base = os.path.join(os.path.dirname(current_abs_path), require_str)
        for ext in ('', '.rb'):
            candidate = os.path.normpath(base + ext)
            if os.path.exists(candidate):
                return candidate
        return None
    for repo_root in repo_roots:
        for search_dir in (
            os.path.join(repo_root, 'lib'),
            repo_root,
        ):
            for ext in ('', '.rb'):
                candidate = os.path.normpath(os.path.join(search_dir, require_str + ext))
                if os.path.exists(candidate):
                    return candidate
    return None


def _resolve_ts_import(import_str: str, current_abs_path: str) -> Optional[str]:
    """Resolve a relative TS/JS import to an absolute path, or None."""
    if not import_str.startswith('.'):
        return None
    base = os.path.normpath(
        os.path.join(os.path.dirname(current_abs_path), import_str)
    )
    for suffix in ('', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.tsx', '/index.js'):
        candidate = base + suffix
        if os.path.exists(candidate):
            return candidate
    return None
```

- [ ] **Step 2: Commit**

```bash
git add graph.py
git commit -m "feat: add TS/JS parser and path resolution helpers to graph.py"
```

---

## Task 4: graph.py — Neo4j write logic

**Files:**
- Modify: `graph.py`

- [ ] **Step 1: Add constants, _ensure_constraints, _batch_write, build_service_graph, clear_service_graph to graph.py**

Open `graph.py`. After the `_resolve_ts_import` function (end of file), append:

```python

# ---------------------------------------------------------------------------
# Neo4j write logic
# ---------------------------------------------------------------------------

_EXCLUDED_DIRS = frozenset({
    'node_modules', 'dist', '.git', 'log', 'tmp', 'vendor',
    'coverage', '__tests__', 'cypress', 'e2e',
})

_EXT_TO_LANGUAGE: dict[str, str] = {
    '.rb': 'ruby',
    '.rake': 'ruby',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'javascript',
}


def _ensure_constraints(session: neo4j.Session) -> None:
    session.run("""
        CREATE CONSTRAINT file_path_unique IF NOT EXISTS
        FOR (f:File) REQUIRE (f.path, f.service) IS UNIQUE
    """)
    session.run("""
        CREATE CONSTRAINT method_unique IF NOT EXISTS
        FOR (m:Method) REQUIRE (m.file_path, m.name, m.class_name) IS UNIQUE
    """)


def _batch_write(
    session: neo4j.Session, query: str, params: list[dict], batch_size: int = 500
) -> None:
    for i in range(0, len(params), batch_size):
        batch = params[i:i + batch_size]
        if batch:
            session.run(query, nodes=batch)


def build_service_graph(svc: ServiceConfig, driver: neo4j.Driver) -> None:
    """Parse all files in svc.repos and write nodes/edges to Neo4j."""

    # --- Step 1: walk repos, parse every supported file ---
    all_parsed: list[tuple[str, str, ParsedFile]] = []  # (abs_path, rel_path, parsed)
    abs_to_rel: dict[str, str] = {}

    for repo_path in svc.repos:
        repos_prefix = os.path.dirname(repo_path).rstrip('/') + '/'
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in _EXCLUDED_DIRS]
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                lang = _EXT_TO_LANGUAGE.get(ext)
                if not lang:
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = abs_path.removeprefix(repos_prefix)
                abs_to_rel[abs_path] = rel_path
                try:
                    source = open(abs_path, encoding='utf-8', errors='replace').read()
                    if lang == 'ruby':
                        parsed = _parse_ruby(source, rel_path, svc.name)
                    else:
                        parsed = _parse_ts_js(source, rel_path, svc.name, lang)
                    all_parsed.append((abs_path, rel_path, parsed))
                except Exception as exc:
                    print(f"[graph:{svc.name}] Parse error {rel_path}: {exc}")

    print(f"[graph:{svc.name}] Parsed {len(all_parsed)} files.")

    # --- Step 2: resolve REQUIRES edges to rel_paths ---
    requires_edges: list[dict] = []
    for abs_path, rel_path, parsed in all_parsed:
        for req in parsed.requires:
            if parsed.language == 'ruby':
                resolved = _resolve_require(
                    req.require_str, abs_path, req.is_relative, svc.repos
                )
            else:
                resolved = _resolve_ts_import(req.require_str, abs_path)
            if resolved and resolved in abs_to_rel:
                requires_edges.append({
                    'from_path': rel_path,
                    'to_path': abs_to_rel[resolved],
                    'service': svc.name,
                })

    # --- Step 3: write nodes and edges to Neo4j ---
    with driver.session() as session:
        _ensure_constraints(session)

        # File nodes
        _batch_write(session, """
            UNWIND $nodes AS n
            MERGE (f:File {path: n.path, service: n.service})
            SET f.language = n.language
        """, [
            {'path': rel_path, 'service': svc.name, 'language': p.language}
            for _, rel_path, p in all_parsed
        ])

        # Class nodes + DEFINES edges
        class_params = [
            {'file_path': rel_path, 'class_name': cls.name, 'service': svc.name}
            for _, rel_path, p in all_parsed
            for cls in p.classes
        ]
        _batch_write(session, """
            UNWIND $nodes AS n
            MERGE (c:Class {name: n.class_name, service: n.service})
            SET c.file_path = n.file_path
            WITH c, n
            MATCH (f:File {path: n.file_path, service: n.service})
            MERGE (f)-[:DEFINES]->(c)
        """, class_params)

        # INHERITS edges
        inherits_params = [
            {'child': cls.name, 'parent': cls.parent, 'service': svc.name}
            for _, _, p in all_parsed
            for cls in p.classes
            if cls.parent
        ]
        _batch_write(session, """
            UNWIND $nodes AS n
            MATCH (child:Class {name: n.child, service: n.service})
            MERGE (parent:Class {name: n.parent, service: n.service})
            MERGE (child)-[:INHERITS]->(parent)
        """, inherits_params)

        # INCLUDES edges
        includes_params = [
            {'class_name': inc.class_name, 'module_name': inc.module_name, 'service': svc.name}
            for _, _, p in all_parsed
            for inc in p.includes
        ]
        _batch_write(session, """
            UNWIND $nodes AS n
            MATCH (cls:Class {name: n.class_name, service: n.service})
            MERGE (mod:Class {name: n.module_name, service: n.service})
            MERGE (cls)-[:INCLUDES]->(mod)
        """, includes_params)

        # Method nodes + HAS_METHOD edges
        method_params = [
            {
                'name': m.name,
                'class_name': m.class_name,
                'file_path': rel_path,
                'service': svc.name,
                'start_line': m.start_line,
                'end_line': m.end_line,
            }
            for _, rel_path, p in all_parsed
            for m in p.methods
        ]
        _batch_write(session, """
            UNWIND $nodes AS n
            MERGE (m:Method {name: n.name, class_name: n.class_name, file_path: n.file_path})
            SET m.service = n.service, m.start_line = n.start_line, m.end_line = n.end_line
            WITH m, n
            MATCH (c:Class {name: n.class_name, service: n.service})
            MERGE (c)-[:HAS_METHOD]->(m)
        """, method_params)

        # REQUIRES edges
        _batch_write(session, """
            UNWIND $nodes AS n
            MATCH (from:File {path: n.from_path, service: n.service})
            MATCH (to:File {path: n.to_path, service: n.service})
            MERGE (from)-[:REQUIRES]->(to)
        """, requires_edges)

        # CALLS edges (best-effort: match by method name + service, cross-file only)
        calls_params = [
            {
                'from_method': c.from_method,
                'from_class': c.from_class,
                'from_file': rel_path,
                'to_method': c.to_method,
                'service': svc.name,
            }
            for _, rel_path, p in all_parsed
            for c in p.calls
        ]
        _batch_write(session, """
            UNWIND $nodes AS n
            MATCH (from_m:Method {name: n.from_method, class_name: n.from_class, file_path: n.from_file})
            MATCH (to_m:Method {name: n.to_method, service: n.service})
            WHERE to_m.file_path <> n.from_file
            MERGE (from_m)-[:CALLS]->(to_m)
        """, calls_params)

    print(f"[graph:{svc.name}] Graph written: {len(all_parsed)} files, "
          f"{len(requires_edges)} REQUIRES edges.")


def clear_service_graph(service_name: str, driver: neo4j.Driver) -> None:
    """Delete all nodes and relationships for a service from Neo4j."""
    with driver.session() as session:
        session.run(
            "MATCH (n {service: $service}) DETACH DELETE n",
            service=service_name,
        )
    print(f"[graph:{service_name}] Graph cleared.")
```

- [ ] **Step 2: Commit**

```bash
git add graph.py
git commit -m "feat: add Neo4j write logic to graph.py (build_service_graph, clear_service_graph)"
```

---

## Task 5: graph_retriever.py — 1-hop Cypher expansion

**Files:**
- Create: `graph_retriever.py`

- [ ] **Step 1: Create graph_retriever.py**

Create `/Users/shijudevarajan/Codebase/poc/codebot/graph_retriever.py`:

```python
import neo4j

_EXPANSION_QUERY = """
MATCH (f:File)
WHERE f.path IN $seed_paths AND f.service = $service
OPTIONAL MATCH (f)-[:REQUIRES]->(dep:File)
OPTIONAL MATCH (caller:File)-[:REQUIRES]->(f)
OPTIONAL MATCH (f)-[:DEFINES]->(:Class)-[:HAS_METHOD]->(m:Method)
               -[:CALLS]->(target:Method)<-[:HAS_METHOD]-(:Class)
               <-[:DEFINES]-(target_file:File)
RETURN
  COLLECT(DISTINCT dep.path)         AS requires,
  COLLECT(DISTINCT caller.path)      AS required_by,
  COLLECT(DISTINCT target_file.path) AS call_targets
"""


def expand_file_paths(
    seed_paths: list[str],
    service: str,
    driver: neo4j.Driver,
    max_files: int = 5,
) -> list[str]:
    """Return up to max_files unique file paths reachable in 1 hop from seed_paths.

    Ranked: direct REQUIRES dependencies first, then CALLS targets, then callers.
    Seed paths are excluded from the result.
    """
    if not seed_paths:
        return []

    seed_set = set(seed_paths)

    with driver.session() as session:
        result = session.run(
            _EXPANSION_QUERY,
            seed_paths=seed_paths,
            service=service,
        )
        record = result.single()

    if not record:
        return []

    requires = record.get("requires") or []
    call_targets = record.get("call_targets") or []
    required_by = record.get("required_by") or []

    # Merge in priority order, deduplicate, strip None and seeds
    seen: set[str] = set()
    ordered: list[str] = []
    for path in requires + call_targets + required_by:
        if path and path not in seed_set and path not in seen:
            seen.add(path)
            ordered.append(path)

    return ordered[:max_files]
```

- [ ] **Step 2: Commit**

```bash
git add graph_retriever.py
git commit -m "feat: add graph_retriever.py — 1-hop Cypher expansion of file paths"
```

---

## Task 6: graph_postprocessor.py — GraphExpansionPostprocessor

**Files:**
- Create: `graph_postprocessor.py`

- [ ] **Step 1: Create graph_postprocessor.py**

Create `/Users/shijudevarajan/Codebase/poc/codebot/graph_postprocessor.py`:

```python
from typing import Any, Optional

import neo4j
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from graph_retriever import expand_file_paths


class GraphExpansionPostprocessor(BaseNodePostprocessor):
    """Expand retrieval results by 1 hop in the code knowledge graph.

    For each file path in the vector-search hits, queries Neo4j for directly
    required/requiring files and method-call targets, fetches their chunks from
    ChromaDB, and merges them with the original nodes before reranking.

    If Neo4j is unreachable the original nodes are returned unchanged.
    """

    max_expanded_files: int = 5
    max_expanded_chunks: int = 15

    _service_name: str = PrivateAttr()
    _collection: Any = PrivateAttr()   # chromadb.Collection
    _driver: Any = PrivateAttr()       # neo4j.Driver | None

    def __init__(
        self,
        service_name: str,
        chroma_collection: Any,
        driver: Any,
        max_expanded_files: int = 5,
        max_expanded_chunks: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_expanded_files=max_expanded_files,
            max_expanded_chunks=max_expanded_chunks,
            **kwargs,
        )
        self._service_name = service_name
        self._collection = chroma_collection
        self._driver = driver

    @classmethod
    def class_name(cls) -> str:
        return "GraphExpansionPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not self._driver or not nodes:
            return nodes

        seed_paths = list({
            n.node.metadata.get("file_path")
            for n in nodes
            if n.node.metadata.get("file_path")
        })
        if not seed_paths:
            return nodes

        try:
            expanded_paths = expand_file_paths(
                seed_paths=seed_paths,
                service=self._service_name,
                driver=self._driver,
                max_files=self.max_expanded_files,
            )
        except Exception as exc:
            print(f"[graph_postprocessor] Neo4j expansion failed: {exc}")
            return nodes

        new_paths = [p for p in expanded_paths if p not in set(seed_paths)]
        if not new_paths:
            return nodes

        try:
            results = self._collection.get(
                where={"file_path": {"$in": new_paths}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"[graph_postprocessor] ChromaDB fetch failed: {exc}")
            return nodes

        documents = (results.get("documents") or [])[:self.max_expanded_chunks]
        metadatas = (results.get("metadatas") or [])[:self.max_expanded_chunks]

        expanded_nodes = [
            NodeWithScore(node=TextNode(text=doc, metadata=meta or {}), score=0.0)
            for doc, meta in zip(documents, metadatas)
            if doc
        ]

        return nodes + expanded_nodes
```

- [ ] **Step 2: Commit**

```bash
git add graph_postprocessor.py
git commit -m "feat: add GraphExpansionPostprocessor — merges graph-expanded chunks before reranking"
```

---

## Task 7: runner.py — wire Neo4j driver, graph build, and postprocessor

**Files:**
- Modify: `runner.py`

This task makes six targeted changes to `runner.py`. Apply them in order.

- [ ] **Step 1: Add Neo4j env vars after the existing env var block**

In `runner.py`, find this block (around line 37–39):

```python
JIRA_BASE_URL = os.getenv("JIRA_BASE_URL", "")
JIRA_EMAIL = os.getenv("JIRA_EMAIL", "")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")
```

Add two lines immediately after it:

```python
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "codebot-secret")
```

- [ ] **Step 2: Add graph imports after the existing jira imports**

In `runner.py`, find this block:

```python
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.chat_engine import ContextChatEngine
```

Add three lines immediately after:

```python
import graph as _graph
from graph_postprocessor import GraphExpansionPostprocessor
```

- [ ] **Step 3: Add Neo4j driver singleton after the reranker lock block**

In `runner.py`, find these lines:

```python
_reranker: FlagEmbeddingReranker | None = None
_reranker_lock = threading.Lock()
```

Add the Neo4j driver singleton immediately after:

```python
neo4j_driver = None
try:
    import neo4j as _neo4j_lib
    neo4j_driver = _neo4j_lib.GraphDatabase.driver(
        NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD)
    )
    neo4j_driver.verify_connectivity()
    print(f"[startup] Neo4j connected: {NEO4J_URI}")
except Exception as _neo4j_exc:
    print(f"[startup] Neo4j not available ({_neo4j_exc}); graph expansion disabled")
    neo4j_driver = None
```

- [ ] **Step 4: Add chromadb import at the top of the file**

In `runner.py`, find:

```python
import chromadb
```

It is already imported (line 10). No change needed — verify it is present.

- [ ] **Step 5: Replace _init_all_services to store graph_postprocessor per service**

In `runner.py`, find the entire `_init_all_services` function:

```python
def _init_all_services():
    global services
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        index = _build_service_index(svc)
        services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "jira_project_key": svc.jira_project_key,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    print(f"codebot ready. Services: {list(services.keys())}")
```

Replace it with:

```python
def _init_all_services():
    global services
    configs = load_services(SERVICES_CONFIG_PATH)
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    for svc in configs:
        index = _build_service_index(svc)

        if neo4j_driver:
            try:
                _graph.build_service_graph(svc, neo4j_driver)
            except Exception as exc:
                print(f"[graph:{svc.name}] Graph build failed ({exc}); continuing without graph")

        graph_pp = None
        if neo4j_driver:
            collection = chroma_client.get_or_create_collection(f"{svc.name}_codebase")
            graph_pp = GraphExpansionPostprocessor(
                service_name=svc.name,
                chroma_collection=collection,
                driver=neo4j_driver,
            )

        services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "jira_project_key": svc.jira_project_key,
            "graph_postprocessor": graph_pp,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    print(f"codebot ready. Services: {list(services.keys())}")
```

- [ ] **Step 6: Replace _get_engine to prepend GraphExpansionPostprocessor**

In `runner.py`, find the entire `_get_engine` function:

```python
def _get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        svc = services[service_name]
        base_retriever = svc["index"].as_retriever(similarity_top_k=10)
        fusion_retriever = QueryFusionRetriever(
            retrievers=[base_retriever],
            llm=llm,
            num_queries=3,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            similarity_top_k=30,
        )
        sessions[session_id] = {
            "memory": memory,
            "engine": ContextChatEngine.from_defaults(
                retriever=fusion_retriever,
                llm=llm,
                memory=memory,
                node_postprocessors=[_get_reranker()],
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]
```

Replace it with:

```python
def _get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        svc = services[service_name]
        base_retriever = svc["index"].as_retriever(similarity_top_k=10)
        fusion_retriever = QueryFusionRetriever(
            retrievers=[base_retriever],
            llm=llm,
            num_queries=3,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            similarity_top_k=30,
        )
        postprocessors = []
        if svc.get("graph_postprocessor"):
            postprocessors.append(svc["graph_postprocessor"])
        postprocessors.append(_get_reranker())
        sessions[session_id] = {
            "memory": memory,
            "engine": ContextChatEngine.from_defaults(
                retriever=fusion_retriever,
                llm=llm,
                memory=memory,
                node_postprocessors=postprocessors,
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]
```

- [ ] **Step 7: Update /reindex and /reindex/{service_name} to rebuild graph**

In `runner.py`, find the `reindex_all` endpoint. It currently starts with:

```python
@app.post("/reindex")
async def reindex_all():
    global services
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        try:
            chroma_client.delete_collection(f"{svc.name}_codebase")
        except Exception as e:
            print(f"[reindex] Warning: could not delete collection {svc.name}_codebase: {e}")

    new_services: dict = {}
    for svc in configs:
        index = await asyncio.to_thread(_build_service_index, svc)
        new_services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    services = new_services
    print(f"codebot reindexed. Services: {list(services.keys())}")
    return {"status": "reindexed", "services": list(services.keys())}
```

Replace it with:

```python
@app.post("/reindex")
async def reindex_all():
    global services
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        try:
            chroma_client.delete_collection(f"{svc.name}_codebase")
        except Exception as e:
            print(f"[reindex] Warning: could not delete collection {svc.name}_codebase: {e}")
        if neo4j_driver:
            try:
                _graph.clear_service_graph(svc.name, neo4j_driver)
            except Exception as e:
                print(f"[reindex] Warning: could not clear graph {svc.name}: {e}")

    new_services: dict = {}
    for svc in configs:
        index = await asyncio.to_thread(_build_service_index, svc)
        if neo4j_driver:
            try:
                await asyncio.to_thread(_graph.build_service_graph, svc, neo4j_driver)
            except Exception as exc:
                print(f"[graph:{svc.name}] Graph build failed ({exc}); continuing without graph")
        graph_pp = None
        if neo4j_driver:
            collection = chroma_client.get_or_create_collection(f"{svc.name}_codebase")
            graph_pp = GraphExpansionPostprocessor(
                service_name=svc.name,
                chroma_collection=collection,
                driver=neo4j_driver,
            )
        new_services[svc.name] = {
            "index": index,
            "system_prompt": svc.system_prompt,
            "graph_postprocessor": graph_pp,
            "sessions": {
                "local": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    services = new_services
    print(f"codebot reindexed. Services: {list(services.keys())}")
    return {"status": "reindexed", "services": list(services.keys())}
```

Now find the `reindex_service` endpoint. It currently ends with:

```python
    is_new = service_name not in services
    if not is_new:
        for llm_sessions in services[service_name]["sessions"].values():
            llm_sessions.clear()

    services[service_name] = {
        "index": index,
        "system_prompt": svc_config.system_prompt,
        "jira_project_key": svc_config.jira_project_key,
        "sessions": services.get(service_name, {}).get("sessions", {
            "local": OrderedDict(),
            "bedrock": OrderedDict(),
        }),
    }
    return {"status": "reindexed", "service": service_name}
```

Replace it with:

```python
    is_new = service_name not in services
    if not is_new:
        for llm_sessions in services[service_name]["sessions"].values():
            llm_sessions.clear()

    if neo4j_driver:
        try:
            _graph.clear_service_graph(service_name, neo4j_driver)
            await asyncio.to_thread(_graph.build_service_graph, svc_config, neo4j_driver)
        except Exception as exc:
            print(f"[graph:{service_name}] Graph rebuild failed ({exc}); continuing without graph")

    graph_pp = None
    if neo4j_driver:
        chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
        collection = chroma_client.get_or_create_collection(f"{service_name}_codebase")
        graph_pp = GraphExpansionPostprocessor(
            service_name=service_name,
            chroma_collection=collection,
            driver=neo4j_driver,
        )

    services[service_name] = {
        "index": index,
        "system_prompt": svc_config.system_prompt,
        "jira_project_key": svc_config.jira_project_key,
        "graph_postprocessor": graph_pp,
        "sessions": services.get(service_name, {}).get("sessions", {
            "local": OrderedDict(),
            "bedrock": OrderedDict(),
        }),
    }
    return {"status": "reindexed", "service": service_name}
```

- [ ] **Step 8: Commit**

```bash
git add runner.py
git commit -m "feat: wire Neo4j driver and GraphExpansionPostprocessor into runner.py"
```

---

## Task 8: Write all test files

**Files:**
- Create: `tests/test_graph_parser.py`
- Create: `tests/test_graph_retriever.py`
- Create: `tests/test_graph_postprocessor.py`
- Create: `tests/test_graph_integration.py`

- [ ] **Step 1: Create tests/test_graph_parser.py**

```python
import pytest
from graph import _parse_ruby, _parse_ts_js


# --- _parse_ruby ---

def test_parse_ruby_bare_require():
    source = "require 'snt/channex/exporters/rate_exporter'\n"
    result = _parse_ruby(source, "rover-ifc/test/integration/channex_test.rb", "rover-ifc")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "snt/channex/exporters/rate_exporter"
    assert result.requires[0].is_relative is False


def test_parse_ruby_require_relative():
    source = "require_relative '../exporters/rate_exporter'\n"
    result = _parse_ruby(source, "rover-ifc/lib/snt/channex/services/rate_sync.rb", "rover-ifc")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "../exporters/rate_exporter"
    assert result.requires[0].is_relative is True


def test_parse_ruby_class_and_method():
    source = """
class RateExporter
  def build_occupancy_rates(room_rate_data)
    single_rate = room_rate_data[:single_amount].to_f
  end
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb", "rover-ifc")
    class_names = [c.name for c in result.classes]
    assert "RateExporter" in class_names
    method_names = [m.name for m in result.methods]
    assert "build_occupancy_rates" in method_names


def test_parse_ruby_inheritance():
    source = """
class RateExporter < BaseExporter
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/foo.rb", "rover-ifc")
    assert any(c.name == "RateExporter" and c.parent == "BaseExporter" for c in result.classes)


def test_parse_ruby_include():
    source = """
class Foo
  include RateHelpers
end
"""
    result = _parse_ruby(source, "rover-ifc/lib/foo.rb", "rover-ifc")
    assert any(i.class_name == "Foo" and i.module_name == "RateHelpers" for i in result.includes)


def test_parse_ruby_multiple_requires():
    source = """
require 'foo'
require 'bar'
"""
    result = _parse_ruby(source, "rover-ifc/lib/baz.rb", "rover-ifc")
    require_strs = [r.require_str for r in result.requires]
    assert "foo" in require_strs
    assert "bar" in require_strs


def test_parse_ruby_empty_source():
    result = _parse_ruby("", "rover-ifc/lib/empty.rb", "rover-ifc")
    assert result.requires == []
    assert result.classes == []
    assert result.methods == []


# --- _parse_ts_js ---

def test_parse_ts_relative_import():
    source = "import { RateService } from './rate.service';\n"
    result = _parse_ts_js(source, "ibe-api/src/services/checkout.service.ts", "ibe", "typescript")
    assert len(result.requires) == 1
    assert result.requires[0].require_str == "./rate.service"
    assert result.requires[0].is_relative is True


def test_parse_ts_absolute_import_not_marked_relative():
    source = "import { Injectable } from '@angular/core';\n"
    result = _parse_ts_js(source, "ibe-admin/src/app/foo.ts", "ibe", "typescript")
    assert len(result.requires) == 1
    assert result.requires[0].is_relative is False


def test_parse_ts_class_and_method():
    source = """
class CheckoutService {
  async processCheckout(cart: any): Promise<void> {
    return this.validate(cart);
  }
}
"""
    result = _parse_ts_js(source, "ibe-api/src/services/checkout.service.ts", "ibe", "typescript")
    class_names = [c.name for c in result.classes]
    assert "CheckoutService" in class_names
    method_names = [m.name for m in result.methods]
    assert "processCheckout" in method_names


def test_parse_ts_class_inheritance():
    source = """
class CheckoutService extends BaseService {
}
"""
    result = _parse_ts_js(source, "ibe-api/src/foo.ts", "ibe", "typescript")
    assert any(c.name == "CheckoutService" and c.parent == "BaseService" for c in result.classes)


def test_parse_ts_empty_source():
    result = _parse_ts_js("", "ibe-api/src/empty.ts", "ibe", "typescript")
    assert result.requires == []
    assert result.classes == []
    assert result.methods == []
```

- [ ] **Step 2: Create tests/test_graph_retriever.py**

```python
from unittest.mock import MagicMock
from graph_retriever import expand_file_paths


def _make_mock_driver(requires=None, required_by=None, call_targets=None):
    record = MagicMock()
    record.get.side_effect = lambda k: {
        "requires": requires or [],
        "required_by": required_by or [],
        "call_targets": call_targets or [],
    }.get(k, [])

    mock_result = MagicMock()
    mock_result.single.return_value = record

    session = MagicMock()
    session.run.return_value = mock_result
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session
    return driver


def test_expand_returns_requires_paths():
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"]
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb" in result


def test_expand_excludes_seed_paths():
    driver = _make_mock_driver(
        required_by=["rover-ifc/test/integration/channex_test.rb"]
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert "rover-ifc/test/integration/channex_test.rb" not in result


def test_expand_caps_at_max_files():
    many_paths = [f"rover-ifc/lib/file{i}.rb" for i in range(20)]
    driver = _make_mock_driver(requires=many_paths)
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=3,
    )
    assert len(result) <= 3


def test_expand_returns_empty_on_no_neighbours():
    driver = _make_mock_driver()
    result = expand_file_paths(
        seed_paths=["rover-ifc/lib/isolated.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result == []


def test_expand_returns_empty_on_no_seed_paths():
    driver = _make_mock_driver(requires=["some/path.rb"])
    result = expand_file_paths(
        seed_paths=[],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result == []


def test_expand_deduplicates_paths():
    # same path appears in both requires and call_targets
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/foo.rb"],
        call_targets=["rover-ifc/lib/foo.rb"],
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result.count("rover-ifc/lib/foo.rb") == 1


def test_expand_prioritises_requires_over_callers():
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/dep.rb"],
        required_by=["rover-ifc/lib/caller.rb"],
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/lib/seed.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=1,  # only 1 slot
    )
    # requires comes first in priority order
    assert result == ["rover-ifc/lib/dep.rb"]
```

- [ ] **Step 3: Create tests/test_graph_postprocessor.py**

```python
from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

from graph_postprocessor import GraphExpansionPostprocessor


def _make_node(file_path: str, text: str = "code snippet") -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, metadata={"file_path": file_path}),
        score=0.8,
    )


def test_returns_original_nodes_when_driver_is_none():
    pp = GraphExpansionPostprocessor(
        service_name="rover-ifc",
        chroma_collection=MagicMock(),
        driver=None,
    )
    nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
    result = pp._postprocess_nodes(nodes)
    assert result == nodes


def test_returns_original_nodes_when_nodes_empty():
    pp = GraphExpansionPostprocessor(
        service_name="rover-ifc",
        chroma_collection=MagicMock(),
        driver=MagicMock(),
    )
    result = pp._postprocess_nodes([])
    assert result == []


def test_merges_expanded_nodes():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["def build_occupancy_rates..."],
        "metadatas": [{"file_path": "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"}],
    }

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    assert len(result) == 2
    paths = [n.node.metadata["file_path"] for n in result]
    assert "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb" in paths


def test_degrades_gracefully_on_neo4j_failure():
    with patch("graph_postprocessor.expand_file_paths", side_effect=Exception("Neo4j down")):
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=MagicMock(),
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/lib/foo.rb")]
        result = pp._postprocess_nodes(nodes)

    assert result == nodes


def test_does_not_fetch_when_expanded_paths_are_all_seeds():
    mock_collection = MagicMock()

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        # expansion returns only the seed path itself
        mock_expand.return_value = ["rover-ifc/test/integration/channex_test.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    mock_collection.get.assert_not_called()
    assert len(result) == 1


def test_degrades_gracefully_on_chromadb_failure():
    mock_collection = MagicMock()
    mock_collection.get.side_effect = Exception("ChromaDB error")

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    assert result == nodes


def test_expanded_nodes_get_score_zero():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["expanded code"],
        "metadatas": [{"file_path": "rover-ifc/lib/rate_exporter.rb"}],
    }

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    expanded = [n for n in result if n.node.metadata.get("file_path") == "rover-ifc/lib/rate_exporter.rb"]
    assert len(expanded) == 1
    assert expanded[0].score == 0.0
```

- [ ] **Step 4: Create tests/test_graph_integration.py**

```python
import os
import pytest

from config import ServiceConfig

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "codebot-secret")

pytestmark = pytest.mark.skipif(
    not NEO4J_URI,
    reason="NEO4J_URI not set — skipping graph integration tests",
)


@pytest.fixture
def driver():
    import neo4j as _neo4j
    d = _neo4j.GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    yield d
    d.close()


@pytest.fixture
def fixture_repo(tmp_path):
    """Create a minimal repo structure mimicking rover-ifc."""
    lib_dir = tmp_path / "lib" / "snt" / "channex" / "exporters"
    lib_dir.mkdir(parents=True)
    test_dir = tmp_path / "test" / "integration"
    test_dir.mkdir(parents=True)

    (lib_dir / "rate_exporter.rb").write_text("""
class RateExporter
  def build_occupancy_rates(room_rate_data)
    single_rate = room_rate_data[:single_amount]&.to_f
  end
end
""")

    (test_dir / "channex_test.rb").write_text("""
require 'snt/channex/exporters/rate_exporter'

class ChannexTest
  def test_rate_building
    exporter = RateExporter.new
  end
end
""")

    return tmp_path


@pytest.fixture
def svc(fixture_repo):
    return ServiceConfig(
        name="test-service",
        repos=[str(fixture_repo)],
        system_prompt="Test",
        file_extensions=[".rb"],
        jira_project_key="TEST",
    )


def test_rate_exporter_surfaces_from_channex_test(driver, svc):
    from graph import build_service_graph, clear_service_graph
    from graph_retriever import expand_file_paths

    clear_service_graph("test-service", driver)
    build_service_graph(svc, driver)

    repo_path = svc.repos[0]
    repos_prefix = os.path.dirname(repo_path).rstrip("/") + "/"

    test_abs = os.path.join(repo_path, "test/integration/channex_test.rb")
    seed_rel = test_abs.removeprefix(repos_prefix)

    expanded = expand_file_paths(
        seed_paths=[seed_rel],
        service="test-service",
        driver=driver,
        max_files=5,
    )

    exporter_abs = os.path.join(repo_path, "lib/snt/channex/exporters/rate_exporter.rb")
    exporter_rel = exporter_abs.removeprefix(repos_prefix)

    assert exporter_rel in expanded, (
        f"rate_exporter.rb ({exporter_rel!r}) not found in expansion: {expanded}"
    )

    clear_service_graph("test-service", driver)
```

- [ ] **Step 5: Commit test files**

```bash
git add tests/test_graph_parser.py tests/test_graph_retriever.py tests/test_graph_postprocessor.py tests/test_graph_integration.py
git commit -m "test: add unit and integration tests for graph parser, retriever, and postprocessor"
```

---

## Task 9: Run all tests

- [ ] **Step 1: Install dependencies (outside Docker, for local test run)**

```bash
cd /Users/shijudevarajan/Codebase/poc/codebot
pip install neo4j tree-sitter-languages llama-index llama-index-postprocessor-flag-embedding-reranker chromadb
```

- [ ] **Step 2: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected output — all these tests should pass:

```
tests/test_config.py::test_load_services PASSED
tests/test_message.py::... PASSED
tests/test_jira.py::... PASSED
tests/test_api.py::... PASSED
tests/test_graph_parser.py::test_parse_ruby_bare_require PASSED
tests/test_graph_parser.py::test_parse_ruby_require_relative PASSED
tests/test_graph_parser.py::test_parse_ruby_class_and_method PASSED
tests/test_graph_parser.py::test_parse_ruby_inheritance PASSED
tests/test_graph_parser.py::test_parse_ruby_include PASSED
tests/test_graph_parser.py::test_parse_ruby_multiple_requires PASSED
tests/test_graph_parser.py::test_parse_ruby_empty_source PASSED
tests/test_graph_parser.py::test_parse_ts_relative_import PASSED
tests/test_graph_parser.py::test_parse_ts_absolute_import_not_marked_relative PASSED
tests/test_graph_parser.py::test_parse_ts_class_and_method PASSED
tests/test_graph_parser.py::test_parse_ts_class_inheritance PASSED
tests/test_graph_parser.py::test_parse_ts_empty_source PASSED
tests/test_graph_retriever.py::... PASSED (7 tests)
tests/test_graph_postprocessor.py::... PASSED (7 tests)
tests/test_graph_integration.py::... SKIPPED (NEO4J_URI not set)
```

If any graph parser tests fail with `AttributeError` on `child_by_field_name` or unexpected node types, check the tree-sitter-languages version and adjust the node type strings (e.g., `'string_content'` may be `'string_fragment'` in some versions). The fix is to update the type check in the relevant helper function.

- [ ] **Step 3: Fix any failing tests, then commit**

If all tests pass without changes, skip this step. If fixes are needed, make the minimal change required, verify tests pass, then:

```bash
git add -p  # stage only the fix
git commit -m "fix: adjust tree-sitter node types for graph parser compatibility"
```
