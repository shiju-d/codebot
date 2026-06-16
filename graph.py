import os
import threading
import neo4j
from dataclasses import dataclass, field
from typing import Optional

from config import ServiceConfig

from tree_sitter_language_pack import get_parser

# Thread-local parser cache — tree-sitter Parser objects are not thread-safe.
_tl = threading.local()


def _ruby_parser():
    if not hasattr(_tl, 'ruby'):
        _tl.ruby = get_parser('ruby')
    return _tl.ruby


def _ts_parser(lang: str):
    if not hasattr(_tl, 'ts_parsers'):
        _tl.ts_parsers = {}
    if lang not in _tl.ts_parsers:
        _tl.ts_parsers[lang] = get_parser(lang)
    return _tl.ts_parsers[lang]

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
# tree-sitter-language-pack API adapters
# (All node attributes are methods: node.kind(), node.child_count(), etc.)
# ---------------------------------------------------------------------------

def _ntext(node, src: bytes) -> str:
    br = node.byte_range()
    return src[br.start:br.end].decode('utf-8')


def _children(node):
    return [node.child(i) for i in range(node.child_count())]


# ---------------------------------------------------------------------------
# Ruby tree-sitter helpers
# ---------------------------------------------------------------------------

def _ruby_node_name(node, src: bytes) -> Optional[str]:
    name_child = node.child_by_field_name('name')
    if not name_child:
        return None
    k = name_child.kind()
    if k in ('constant', 'scope_resolution'):
        return _ntext(name_child, src)
    return None


def _ruby_superclass_name(class_node, src: bytes) -> Optional[str]:
    sc = class_node.child_by_field_name('superclass')
    if not sc:
        return None
    for child in _children(sc):
        if child.kind() in ('constant', 'scope_resolution'):
            return _ntext(child, src)
    return None


def _ruby_method_name_from_def(node, src: bytes) -> Optional[str]:
    name_child = node.child_by_field_name('name')
    if name_child and name_child.kind() in ('identifier', 'operator', 'constant'):
        return _ntext(name_child, src)
    return None


def _ruby_call_method_name(call_node, src: bytes) -> Optional[str]:
    method_child = call_node.child_by_field_name('method')
    if method_child and method_child.kind() == 'identifier':
        return _ntext(method_child, src)
    return None


def _ruby_string_arg(call_node, src: bytes) -> Optional[str]:
    args = call_node.child_by_field_name('arguments')
    if not args:
        return None
    for child in _children(args):
        if child.kind() == 'string':
            content = child.child_by_field_name('string_content')
            if not content:
                content = next((c for c in _children(child) if c.kind() == 'string_content'), None)
            if content:
                return _ntext(content, src)
    return None


def _ruby_const_arg(call_node, src: bytes) -> Optional[str]:
    args = call_node.child_by_field_name('arguments')
    if not args:
        return None
    for child in _children(args):
        if child.kind() in ('constant', 'scope_resolution'):
            return _ntext(child, src)
    return None


def _walk_ruby(node, parsed: ParsedFile, class_stack: list, method_stack: list, src: bytes) -> None:
    t = node.kind()

    if t in ('class', 'module'):
        name = _ruby_node_name(node, src)
        if name:
            parent = _ruby_superclass_name(node, src) if t == 'class' else None
            parsed.classes.append(_ClassNode(name=name, parent=parent))
            class_stack.append(name)
            for child in _children(node):
                _walk_ruby(child, parsed, class_stack, method_stack, src)
            class_stack.pop()
        else:
            for child in _children(node):
                _walk_ruby(child, parsed, class_stack, method_stack, src)
        return

    if t in ('method', 'singleton_method'):
        mname = _ruby_method_name_from_def(node, src)
        if mname:
            cls = class_stack[-1] if class_stack else '__module__'
            parsed.methods.append(_MethodNode(
                name=mname,
                class_name=cls,
                start_line=node.start_position().row,
                end_line=node.end_position().row,
            ))
            method_stack.append(mname)
            for child in _children(node):
                _walk_ruby(child, parsed, class_stack, method_stack, src)
            method_stack.pop()
        return

    if t == 'call':
        mname = _ruby_call_method_name(node, src)
        if mname in ('require', 'require_relative'):
            req_str = _ruby_string_arg(node, src)
            if req_str:
                parsed.requires.append(_RequireEdge(
                    from_path=parsed.file_path,
                    require_str=req_str,
                    is_relative=(mname == 'require_relative'),
                ))
        elif mname in ('include', 'extend') and class_stack:
            module_name = _ruby_const_arg(node, src)
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

    for child in _children(node):
        _walk_ruby(child, parsed, class_stack, method_stack, src)


def _parse_ruby(source: str, file_path: str, service: str) -> ParsedFile:
    src = source.encode('utf-8')
    tree = _ruby_parser().parse(source)
    parsed = ParsedFile(file_path=file_path, service=service, language='ruby')
    _walk_ruby(tree.root_node(), parsed, [], [], src)
    return parsed


# ---------------------------------------------------------------------------
# TypeScript / JavaScript tree-sitter helpers
# ---------------------------------------------------------------------------

def _ts_class_name(class_node, src: bytes) -> Optional[str]:
    name_child = class_node.child_by_field_name('name')
    if name_child and name_child.kind() in ('type_identifier', 'identifier'):
        return _ntext(name_child, src)
    return None


def _ts_superclass_name(class_node, src: bytes) -> Optional[str]:
    for child in _children(class_node):
        if child.kind() == 'class_heritage':
            for heritage_child in _children(child):
                if heritage_child.kind() == 'extends_clause':
                    for ext_child in _children(heritage_child):
                        if ext_child.kind() in ('identifier', 'type_identifier'):
                            return _ntext(ext_child, src)
    return None


def _walk_ts_js(node, parsed: ParsedFile, class_stack: list, method_stack: list, src: bytes) -> None:
    t = node.kind()

    if t == 'import_statement':
        source_node = node.child_by_field_name('source')
        if source_node:
            frag = next(
                (c for c in _children(source_node) if c.kind() == 'string_fragment'),
                None,
            )
            if frag:
                import_str = _ntext(frag, src)
                parsed.requires.append(_RequireEdge(
                    from_path=parsed.file_path,
                    require_str=import_str,
                    is_relative=import_str.startswith('.'),
                ))
        return

    if t in ('class_declaration', 'abstract_class_declaration'):
        class_name = _ts_class_name(node, src)
        if class_name:
            parent = _ts_superclass_name(node, src)
            parsed.classes.append(_ClassNode(name=class_name, parent=parent))
            class_stack.append(class_name)
            for child in _children(node):
                _walk_ts_js(child, parsed, class_stack, method_stack, src)
            class_stack.pop()
        return

    if t == 'method_definition' and class_stack:
        name_child = node.child_by_field_name('name')
        if name_child:
            mname = _ntext(name_child, src)
            parsed.methods.append(_MethodNode(
                name=mname,
                class_name=class_stack[-1],
                start_line=node.start_position().row,
                end_line=node.end_position().row,
            ))
            method_stack.append(mname)
            for child in _children(node):
                _walk_ts_js(child, parsed, class_stack, method_stack, src)
            method_stack.pop()
        return

    if t == 'function_declaration':
        name_child = node.child_by_field_name('name')
        if name_child:
            fname = _ntext(name_child, src)
            cls = class_stack[-1] if class_stack else '__module__'
            parsed.methods.append(_MethodNode(
                name=fname,
                class_name=cls,
                start_line=node.start_position().row,
                end_line=node.end_position().row,
            ))
            method_stack.append(fname)
            for child in _children(node):
                _walk_ts_js(child, parsed, class_stack, method_stack, src)
            method_stack.pop()
        return

    if t == 'lexical_declaration':
        for decl in _children(node):
            if decl.kind() == 'variable_declarator':
                name_child = decl.child_by_field_name('name')
                value_child = decl.child_by_field_name('value')
                if (name_child and value_child
                        and name_child.kind() == 'identifier'
                        and value_child.kind() == 'arrow_function'):
                    fname = _ntext(name_child, src)
                    cls = class_stack[-1] if class_stack else '__module__'
                    parsed.methods.append(_MethodNode(
                        name=fname,
                        class_name=cls,
                        start_line=decl.start_position().row,
                        end_line=decl.end_position().row,
                    ))
                    method_stack.append(fname)
                    for child in _children(value_child):
                        _walk_ts_js(child, parsed, class_stack, method_stack, src)
                    method_stack.pop()
        return

    if t == 'call_expression' and class_stack and method_stack:
        func_child = node.child_by_field_name('function')
        if func_child:
            if func_child.kind() == 'identifier':
                parsed.calls.append(_CallEdge(
                    from_class=class_stack[-1],
                    from_method=method_stack[-1],
                    to_method=_ntext(func_child, src),
                ))
            elif func_child.kind() == 'member_expression':
                prop = func_child.child_by_field_name('property')
                if prop:
                    parsed.calls.append(_CallEdge(
                        from_class=class_stack[-1],
                        from_method=method_stack[-1],
                        to_method=_ntext(prop, src),
                    ))

    for child in _children(node):
        _walk_ts_js(child, parsed, class_stack, method_stack, src)


def _parse_ts_js(source: str, file_path: str, service: str, language: str) -> ParsedFile:
    src = source.encode('utf-8')
    parser = _ts_parser(language)
    tree = parser.parse(source)
    parsed = ParsedFile(file_path=file_path, service=service, language=language)
    _walk_ts_js(tree.root_node(), parsed, [], [], src)
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
                    with open(abs_path, encoding='utf-8', errors='replace') as fh:
                        source = fh.read()
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

        # CALLS edges skipped — requires full Method scan without dedicated index;
        # REQUIRES edges (above) are sufficient for the graph expansion use case.

    print(f"[graph:{svc.name}] Graph written: {len(all_parsed)} files, "
          f"{len(requires_edges)} REQUIRES edges.")


def clear_service_graph(service_name: str, driver: neo4j.Driver) -> None:
    """Delete all nodes and relationships for a service from Neo4j."""
    with driver.session() as session:
        session.run("MATCH (n:File {service: $service}) DETACH DELETE n", service=service_name)
        session.run("MATCH (n:Method {service: $service}) DETACH DELETE n", service=service_name)
        session.run("MATCH (n:Class {service: $service}) DETACH DELETE n", service=service_name)
    print(f"[graph:{service_name}] Graph cleared.")
