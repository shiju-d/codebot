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
