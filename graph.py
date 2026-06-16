import os
import neo4j
from dataclasses import dataclass, field
from typing import Optional

from config import ServiceConfig

from tree_sitter_language_pack import get_parser

_RUBY_PARSER = get_parser('ruby')
_TS_PARSERS: dict = {
    'typescript': get_parser('typescript'),
    'tsx': get_parser('tsx'),
    'javascript': get_parser('javascript'),
}

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
        return name_child.text.decode('utf-8')
    return None


def _ruby_superclass_name(class_node) -> Optional[str]:
    sc = class_node.child_by_field_name('superclass')
    if not sc:
        return None
    for child in sc.children:
        if child.type in ('constant', 'scope_resolution'):
            return child.text.decode('utf-8')
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
        else:
            for child in node.children:
                _walk_ruby(child, parsed, class_stack, method_stack)
        return

    if t in ('method', 'singleton_method'):
        mname = _ruby_method_name_from_def(node)
        if mname:
            cls = class_stack[-1] if class_stack else '__module__'
            parsed.methods.append(_MethodNode(
                name=mname,
                class_name=cls,
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
    tree = _RUBY_PARSER.parse(source.encode('utf-8'))
    parsed = ParsedFile(file_path=file_path, service=service, language='ruby')
    _walk_ruby(tree.root_node, parsed, [], [])
    return parsed


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

    if t == 'lexical_declaration':
        for decl in node.children:
            if decl.type == 'variable_declarator':
                name_child = decl.child_by_field_name('name')
                value_child = decl.child_by_field_name('value')
                if (name_child and value_child
                        and name_child.type == 'identifier'
                        and value_child.type == 'arrow_function'):
                    fname = name_child.text.decode('utf-8')
                    cls = class_stack[-1] if class_stack else '__module__'
                    parsed.methods.append(_MethodNode(
                        name=fname,
                        class_name=cls,
                        start_line=decl.start_point[0],
                        end_line=decl.end_point[0],
                    ))
                    method_stack.append(fname)
                    for child in value_child.children:
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
    parser = _TS_PARSERS.get(language) or get_parser(language)
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
        session.run("MATCH (n:File {service: $service}) DETACH DELETE n", service=service_name)
        session.run("MATCH (n:Method {service: $service}) DETACH DELETE n", service=service_name)
        session.run("MATCH (n:Class {service: $service}) DETACH DELETE n", service=service_name)
    print(f"[graph:{service_name}] Graph cleared.")
