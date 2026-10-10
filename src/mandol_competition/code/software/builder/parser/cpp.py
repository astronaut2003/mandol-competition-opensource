"""C++ repository parser backed by the bundled tree-sitter C++ grammar."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .c import CParser
from .tree_sitter import unique


class CppParser(CParser):
    """Extract namespaces, classes, methods, overload identities, and calls."""

    language = "cpp"
    grammar_name = "cpp"

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        tree = self._tree(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        imports: list[str] = []
        macros: list[str] = []
        typedefs: list[str] = []
        globals_: list[str] = []
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []
        visited: set[tuple[int, int]] = set()

        for node in root.named_children:
            if node.type == "preproc_include":
                value = (
                    self._text(source_bytes, node.child_by_field_name("path"))
                    .strip()
                    .strip('<>"')
                )
                if value:
                    imports.append(value)
            elif node.type in {"preproc_def", "preproc_function_def"}:
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                if name:
                    macros.append(name)
            elif node.type in {"type_definition", "alias_declaration"}:
                name = self._identifier(
                    source_bytes, node.child_by_field_name("declarator")
                )
                if not name:
                    name = self._identifier(
                        source_bytes, node.child_by_field_name("name")
                    )
                if name:
                    typedefs.append(name)
            elif (
                node.type == "declaration"
                and self._function_declarator(node.child_by_field_name("declarator"))
                is None
            ):
                globals_.extend(self._declaration_names(source_bytes, node))

        def visit(
            node: Node,
            namespaces: Sequence[str] = (),
            class_stack: Sequence[str] = (),
            template: bool = False,
        ) -> None:
            if node.type == "namespace_definition":
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                body = node.child_by_field_name("body")
                for child in body.named_children if body is not None else ():
                    visit(
                        child,
                        (*namespaces, name) if name else namespaces,
                        class_stack,
                        template,
                    )
                return
            if node.type == "template_declaration":
                for child in node.named_children:
                    if child.type != "template_parameter_list":
                        visit(child, namespaces, class_stack, True)
                return
            if node.type in {"class_specifier", "struct_specifier", "union_specifier"}:
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                body = node.child_by_field_name("body")
                if name and body is not None:
                    identity_name = self._identity_cpp_name(name)
                    scope = tuple(namespaces) or (parsed.module_name,)
                    qualified = ".".join((*scope, *class_stack, identity_name))
                    snippet = self._line_text(source_bytes, node)
                    bases: list[str] = []
                    for child in node.named_children:
                        if child.type == "base_class_clause":
                            for base in child.named_children:
                                if base.type == "access_specifier":
                                    continue
                                text = self._clean_cpp_name(
                                    self._text(source_bytes, base).strip()
                                )
                                if text:
                                    bases.append(text.replace("::", "."))
                    class_fact = ParsedClass(
                        name=name,
                        qualified_name=qualified,
                        file_path=relative_path,
                        source_code=snippet,
                        start_line=self._start_line(node),
                        end_line=self._end_line(node),
                        content_hash=normalized_source_hash(snippet),
                        base_classes=unique(bases),
                        docstring=self._comment_before(source_bytes, node),
                        kind=node.type.removesuffix("_specifier"),
                        is_test=parsed.is_test,
                        annotations=["template"] if template else [],
                    )
                    classes.append(class_fact)
                    for child in body.named_children:
                        visit(
                            child, namespaces, (*class_stack, identity_name), template
                        )
                return
            if node.type == "enum_specifier":
                snippet = self._line_text(source_bytes, node)
                declared_name = re.match(
                    r"\s*enum\s+(?:(?:class|struct)\s+)?([A-Za-z_]\w*)",
                    snippet,
                )
                name = declared_name.group(1) if declared_name is not None else ""
                body = node.child_by_field_name("body")
                if name and body is not None:
                    scope = tuple(namespaces) or (parsed.module_name,)
                    qualified = ".".join((*scope, *class_stack, name))
                    classes.append(
                        ParsedClass(
                            name=name,
                            qualified_name=qualified,
                            file_path=relative_path,
                            source_code=snippet,
                            start_line=self._start_line(node),
                            end_line=self._end_line(node),
                            content_hash=normalized_source_hash(snippet),
                            docstring=self._comment_before(source_bytes, node),
                            kind="enum",
                            is_test=parsed.is_test,
                        )
                    )
                return
            if node.type == "function_definition":
                key = (node.start_byte, node.end_byte)
                if key in visited:
                    return
                visited.add(key)
                function = self._parse_cpp_function(
                    source_bytes,
                    node,
                    relative_path,
                    parsed.module_name,
                    namespaces,
                    class_stack,
                    globals_,
                    parsed.is_test,
                    template,
                )
                if function is not None:
                    functions.append(function)
                return
            for child in node.named_children:
                visit(child, namespaces, class_stack, template)

        visit(root)
        for class_fact in classes:
            class_fact.methods = [
                function.symbol_identity or function.qualified_name
                for function in functions
                if function.class_qualified_name == class_fact.qualified_name
            ]

        parsed.imports = unique(imports)
        parsed.exports = unique(
            [function.name for function in functions]
            + [class_fact.name for class_fact in classes]
        )
        parsed.functions = functions
        parsed.classes = classes
        parsed.language_facts = {
            "has_syntax_errors": bool(root.has_error),
            "macros": unique(macros),
            "typedefs": unique(typedefs),
            "globals": unique(globals_),
            "namespaces": unique(
                part
                for item in classes + functions
                for part in item.qualified_name.split(".")[:-1]
            ),
        }
        return parsed

    def _parse_cpp_function(
        self,
        source: bytes,
        node: Node,
        relative_path: str,
        module_name: str,
        namespaces: Sequence[str],
        class_stack: Sequence[str],
        globals_: list[str],
        is_test: bool,
        template: bool,
    ) -> ParsedFunction | None:
        parts = self._function_parts(source, node)
        if parts is None:
            return None
        raw_name, parameters, parameter_types, function_declarator = parts
        raw_name = self._identity_cpp_name(raw_name)
        raw_parts = [part for part in raw_name.split("::") if part]
        simple_name = raw_parts[-1] if raw_parts else raw_name

        if class_stack:
            scope = tuple(namespaces) or (module_name,)
            class_qualified = ".".join((*scope, *class_stack))
            qualified = f"{class_qualified}.{simple_name}"
        elif len(raw_parts) > 1:
            scoped = [self._clean_cpp_name(part) for part in raw_parts[:-1]]
            if namespaces and scoped[: len(namespaces)] == list(namespaces):
                prefix = scoped
            else:
                prefix = [*(namespaces or (module_name,)), *scoped]
            class_qualified = ".".join(prefix)
            qualified = ".".join((*prefix, simple_name))
        elif namespaces:
            class_qualified = ""
            qualified = ".".join((*namespaces, simple_name))
        else:
            class_qualified = ""
            qualified = f"{module_name}.{simple_name}" if module_name else simple_name

        normalized_types = [self._normalize_type(item) for item in parameter_types]
        identity = f"{qualified}({','.join(normalized_types)})"
        body = node.child_by_field_name("body")
        if body is not None and function_declarator.start_byte >= body.start_byte:
            return None
        snippet = self._line_text(source, node)
        if body is not None:
            node_start = node.start_byte
            body_start = body.start_byte
            signature = (
                source[node_start:body_start].decode("utf-8", errors="replace").strip()
            )
            # Error recovery around macro-heavy template classes can surface a
            # huge class prefix as a function_definition and then select an
            # inner method declarator.  A completed type body before the method
            # body is an unambiguous false positive.
            if re.search(
                r"\b(?:class|struct|union|enum)\b[^;{]*\{",
                signature,
                flags=re.DOTALL,
            ):
                return None
        else:
            signature = snippet
        behavior = self._behavior(source, body, global_names=globals_)
        ast_summary = self._ast_summary(body)
        ast_summary.update(
            {
                "parameter_types": normalized_types,
                "template": bool(template),
                "constructor": bool(
                    class_qualified
                    and simple_name == class_qualified.rsplit(".", 1)[-1]
                ),
                "destructor": simple_name.startswith("~"),
            }
        )
        return ParsedFunction(
            name=simple_name,
            qualified_name=qualified,
            signature=signature,
            file_path=relative_path,
            module_name=module_name,
            source_code=snippet,
            start_line=self._start_line(node),
            end_line=self._end_line(node),
            content_hash=normalized_source_hash(snippet),
            docstring=self._comment_before(source, node),
            parameters=parameters,
            return_type=self._text(source, node.child_by_field_name("type")).strip(),
            reads=behavior["reads"],
            writes=behavior["writes"],
            calls=behavior["calls"],
            side_effects=behavior["side_effects"],
            exceptions=behavior["exceptions"],
            config_references=behavior["config_references"],
            ast_summary=ast_summary,
            class_qualified_name=class_qualified,
            is_test=is_test or simple_name.lower().startswith("test"),
            segments=self._segments(source, body, qualified),
            symbol_identity=identity,
            language=self.language,
        )

    @staticmethod
    def _clean_cpp_name(value: str) -> str:
        previous = None
        while previous != value:
            previous = value
            value = re.sub(r"<[^<>]*>", "", value)
        return re.sub(r"\s+", "", value)

    @staticmethod
    def _identity_cpp_name(value: str) -> str:
        """Normalize spacing while preserving overload/template identity."""
        return re.sub(r"\s+", "", value)

    @classmethod
    def _normalize_type(cls, value: str) -> str:
        value = re.sub(r"\s*=.*$", "", value).strip()
        return cls._identity_cpp_name(value)


__all__ = ["CppParser"]
