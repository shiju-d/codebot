from dataclasses import dataclass, field
from typing import Optional

from tree_sitter_languages import get_parser

_RUBY_PARSER = get_parser('ruby')

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

import os as _os


def _resolve_require(
    require_str: str,
    current_abs_path: str,
    is_relative: bool,
    repo_roots: list[str],
) -> Optional[str]:
    """Resolve a Ruby require string to an absolute path, or None if not found."""
    if is_relative:
        base = _os.path.join(_os.path.dirname(current_abs_path), require_str)
        for ext in ('', '.rb'):
            candidate = _os.path.normpath(base + ext)
            if _os.path.exists(candidate):
                return candidate
        return None
    for repo_root in repo_roots:
        for search_dir in (
            _os.path.join(repo_root, 'lib'),
            repo_root,
        ):
            for ext in ('', '.rb'):
                candidate = _os.path.normpath(_os.path.join(search_dir, require_str + ext))
                if _os.path.exists(candidate):
                    return candidate
    return None


def _resolve_ts_import(import_str: str, current_abs_path: str) -> Optional[str]:
    """Resolve a relative TS/JS import to an absolute path, or None."""
    if not import_str.startswith('.'):
        return None
    base = _os.path.normpath(
        _os.path.join(_os.path.dirname(current_abs_path), import_str)
    )
    for suffix in ('', '.ts', '.tsx', '.js', '.jsx', '/index.ts', '/index.tsx', '/index.js'):
        candidate = base + suffix
        if _os.path.exists(candidate):
            return candidate
    return None
