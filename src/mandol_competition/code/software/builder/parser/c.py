"""C repository parser backed by the bundled tree-sitter C grammar."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .tree_sitter import TreeSitterParserBase, unique


class CParser(TreeSitterParserBase):
    """Extract conservative C declarations, definitions, includes, and calls."""

    language = "c"
    grammar_name = "c"

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        # Keep the Tree alive while any of its Node objects are in use.  The
        # tree-sitter Python binding does not guarantee that a detached Node
        # keeps the native tree storage alive.
        tree = self._tree(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        module_name = parsed.module_name

        imports: list[str] = []
        macros: list[str] = []
        typedefs: list[str] = []
        globals_: list[str] = []
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []

        for node in self._walk(root, stop_types=("function_definition",)):
            if node is root:
                continue
            if node.type == "preproc_include":
                path_node = node.child_by_field_name("path")
                value = self._text(source_bytes, path_node).strip().strip('<>"')
                if value:
                    imports.append(value)
            elif node.type in {"preproc_def", "preproc_function_def"}:
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                if name:
                    macros.append(name)
            elif node.type == "type_definition":
                name = self._identifier(
                    source_bytes, node.child_by_field_name("declarator")
                )
                if name:
                    typedefs.append(name)
            elif node.type == "declaration" and not self._inside_function(node):
                declarator = node.child_by_field_name("declarator")
                if self._function_declarator(declarator) is None:
                    globals_.extend(self._declaration_names(source_bytes, node))

        seen_classes: set[tuple[int, int]] = set()
        for node in self._walk(root):
            if node.type not in {
                "struct_specifier",
                "enum_specifier",
                "union_specifier",
            }:
                continue
            body = node.child_by_field_name("body")
            name_node = node.child_by_field_name("name")
            node_span = (node.start_byte, node.end_byte)
            if body is None or name_node is None or node_span in seen_classes:
                continue
            seen_classes.add(node_span)
            name = self._text(source_bytes, name_node).strip()
            qualified = f"{module_name}.{name}" if module_name else name
            snippet = self._line_text(source_bytes, node)
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
                    kind=node.type.removesuffix("_specifier"),
                    is_test=parsed.is_test,
                )
            )

        for node in self._walk(root, stop_types=("function_definition",)):
            if node.type == "function_definition":
                function = self._parse_function_definition(
                    source_bytes,
                    node,
                    relative_path,
                    module_name,
                    globals_,
                    parsed.is_test,
                )
                if function is not None:
                    functions.append(function)
            elif (
                node.type == "declaration"
                and PurePosixPath(relative_path).suffix.lower() == ".h"
                and not self._inside_function(node)
            ):
                function = self._parse_function_declaration(
                    source_bytes,
                    node,
                    relative_path,
                    module_name,
                    parsed.is_test,
                )
                if function is not None:
                    functions.append(function)

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
            "declarations": unique(
                function.name for function in functions if function.is_declaration
            ),
        }
        return parsed

    @staticmethod
    def _inside_function(node: Node) -> bool:
        parent = node.parent
        while parent is not None:
            if parent.type == "function_definition":
                return True
            parent = parent.parent
        return False

    @classmethod
    def _function_declarator(cls, declarator: Node | None) -> Node | None:
        if declarator is None:
            return None
        for node in cls._walk(declarator):
            if node.type == "function_declarator":
                return node
        return None

    @classmethod
    def _function_parts(
        cls,
        source: bytes,
        node: Node,
    ) -> tuple[str, list[str], list[str], Node] | None:
        declarator = node.child_by_field_name("declarator")
        function_declarator = cls._function_declarator(declarator)
        if function_declarator is None:
            return None
        name_node = function_declarator.child_by_field_name("declarator")
        name = cls._identifier(source, name_node)
        if not name:
            return None
        parameters_node = function_declarator.child_by_field_name("parameters")
        parameters: list[str] = []
        parameter_types: list[str] = []
        if parameters_node is not None:
            for parameter in parameters_node.named_children:
                if parameter.type == "variadic_parameter":
                    parameters.append("...")
                    parameter_types.append("...")
                    continue
                if parameter.type != "parameter_declaration":
                    continue
                parameter_name = cls._identifier(
                    source, parameter.child_by_field_name("declarator")
                )
                parameter_text = cls._text(source, parameter).strip()
                type_node = parameter.child_by_field_name("type")
                type_text = cls._text(source, type_node).strip()
                if parameter_name:
                    parameters.append(parameter_name)
                    if not type_text:
                        type_text = re.sub(
                            rf"\b{re.escape(parameter_name)}\b", "", parameter_text
                        ).strip()
                elif parameter_text and parameter_text != "void":
                    parameters.append(parameter_text)
                if parameter_text != "void":
                    parameter_types.append(type_text or parameter_text)
        return name, parameters, parameter_types, function_declarator

    def _parse_function_definition(
        self,
        source: bytes,
        node: Node,
        relative_path: str,
        module_name: str,
        globals_: list[str],
        is_test: bool,
    ) -> ParsedFunction | None:
        snippet = self._line_text(source, node)
        start_line = self._start_line(node)
        end_line = self._end_line(node)
        docstring = self._comment_before(source, node)
        macro_terminator = re.match(r"\s*END_TEST\s*", snippet)
        if macro_terminator is not None:
            start_line += macro_terminator.group(0).count("\n")
            snippet = snippet[macro_terminator.end() :]
        parts = self._function_parts(source, node)
        test_macro = re.search(r"\bSTART_TEST\(\s*([A-Za-z_]\w*)\s*\)", snippet)
        if test_macro is not None:
            name = test_macro.group(1)
            parameters: list[str] = []
            parameter_types: list[str] = []
            start_line += snippet[: test_macro.start()].count("\n")
            snippet = snippet[test_macro.start() :]
        elif parts is not None:
            name, parameters, parameter_types, _ = parts
        else:
            return None
        qualified = f"{module_name}.{name}" if module_name else name
        body = node.child_by_field_name("body")
        if test_macro is not None:
            signature = test_macro.group(0)
        elif body is not None:
            node_start = node.start_byte
            body_start = body.start_byte
            signature = (
                source[node_start:body_start].decode("utf-8", errors="replace").strip()
            )
            signature = re.sub(r"^END_TEST\s+", "", signature)
        else:
            signature = snippet
        return_type = self._text(source, node.child_by_field_name("type")).strip()
        behavior = self._behavior(source, body, global_names=globals_)
        summary = self._ast_summary(body)
        summary["parameter_types"] = parameter_types
        return ParsedFunction(
            name=name,
            qualified_name=qualified,
            signature=signature,
            file_path=relative_path,
            module_name=module_name,
            source_code=snippet,
            start_line=start_line,
            end_line=end_line,
            content_hash=normalized_source_hash(snippet),
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            reads=behavior["reads"],
            writes=behavior["writes"],
            calls=behavior["calls"],
            side_effects=behavior["side_effects"],
            exceptions=behavior["exceptions"],
            config_references=behavior["config_references"],
            ast_summary=summary,
            is_test=is_test or name.lower().startswith("test"),
            segments=self._segments(source, body, qualified),
            symbol_identity=qualified,
            language=self.language,
        )

    def _parse_function_declaration(
        self,
        source: bytes,
        node: Node,
        relative_path: str,
        module_name: str,
        is_test: bool,
    ) -> ParsedFunction | None:
        snippet = self._line_text(source, node)
        start_line = self._start_line(node)
        end_line = self._end_line(node)
        docstring = self._comment_before(source, node)
        parts = self._function_parts(source, node)
        if parts is None:
            return None
        name, parameters, parameter_types, _ = parts
        qualified = f"{module_name}.{name}" if module_name else name
        type_node = node.child_by_field_name("type")
        return ParsedFunction(
            name=name,
            qualified_name=qualified,
            signature=snippet.rstrip(";").strip(),
            file_path=relative_path,
            module_name=module_name,
            source_code=snippet,
            start_line=start_line,
            end_line=end_line,
            content_hash=normalized_source_hash(snippet),
            docstring=docstring,
            parameters=parameters,
            return_type=self._text(source, type_node).strip(),
            ast_summary={
                "branches": 0,
                "loops": 0,
                "returns": 0,
                "awaits": 0,
                "has_try": False,
                "parameter_types": parameter_types,
            },
            is_test=is_test or name.lower().startswith("test"),
            symbol_identity=f"{qualified}@declaration",
            is_declaration=True,
            language=self.language,
        )

    @classmethod
    def _declaration_names(cls, source: bytes, node: Node) -> list[str]:
        names: list[str] = []
        for candidate in cls._walk(node):
            if candidate.type in {
                "init_declarator",
                "pointer_declarator",
                "array_declarator",
            }:
                declarator = candidate.child_by_field_name("declarator")
                name = cls._identifier(source, declarator)
                if name:
                    names.append(name)
            elif candidate.type == "identifier" and candidate.parent == node:
                names.append(cls._text(source, candidate).strip())
        return unique(names)


__all__ = ["CParser"]
