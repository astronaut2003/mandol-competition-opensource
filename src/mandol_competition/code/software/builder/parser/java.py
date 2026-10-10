"""Java repository parser backed by the bundled tree-sitter Java grammar."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .tree_sitter import TreeSitterParserBase, unique


class JavaParser(TreeSitterParserBase):
    """Extract package/class/interface/enum/method facts and stable overload UIDs."""

    language = "java"
    grammar_name = "java"

    _CLASS_TYPES = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "record",
        "annotation_type_declaration": "annotation",
    }

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        tree = self._tree(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        package = ""
        imports: list[str] = []
        aliases: dict[str, str] = {}

        for node in root.named_children:
            if node.type == "package_declaration":
                package = re.sub(
                    r"^package\s+|;\s*$", "", self._text(source_bytes, node).strip()
                )
            elif node.type == "import_declaration":
                imported = re.sub(
                    r"^import\s+(?:static\s+)?|;\s*$",
                    "",
                    self._text(source_bytes, node).strip(),
                )
                if imported:
                    imports.append(imported)
                    aliases[imported.rsplit(".", 1)[-1]] = imported

        if package:
            parsed.module_name = package
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []

        def visit_class(node: Node, class_stack: Sequence[str] = ()) -> None:
            kind = self._CLASS_TYPES[node.type]
            name = self._text(source_bytes, node.child_by_field_name("name")).strip()
            if not name:
                return
            prefix = [part for part in (package, *class_stack) if part]
            qualified = ".".join((*prefix, name))
            body = node.child_by_field_name("body")
            snippet = self._line_text(source_bytes, node)
            bases, interfaces = self._class_bases(source_bytes, node, kind)
            annotations = self._annotations(source_bytes, self._modifiers_node(node))
            class_is_test = (
                parsed.is_test
                or name.endswith(("Test", "Tests"))
                or "Test" in annotations
            )
            class_fact = ParsedClass(
                name=name,
                qualified_name=qualified,
                file_path=relative_path,
                source_code=snippet,
                start_line=self._start_line(node),
                end_line=self._end_line(node),
                content_hash=normalized_source_hash(snippet),
                base_classes=bases,
                docstring=self._comment_before(source_bytes, node),
                interfaces=interfaces,
                kind=kind,
                is_test=class_is_test,
                annotations=annotations,
            )
            classes.append(class_fact)
            if body is None:
                return
            for child in body.named_children:
                if child.type in {"method_declaration", "constructor_declaration"}:
                    function = self._parse_java_function(
                        source_bytes,
                        child,
                        relative_path,
                        parsed.module_name,
                        qualified,
                        class_is_test,
                    )
                    if function is not None:
                        functions.append(function)
                        class_fact.methods.append(
                            function.symbol_identity or function.qualified_name
                        )
                elif child.type in self._CLASS_TYPES:
                    visit_class(child, (*class_stack, name))

        for node in root.named_children:
            if node.type in self._CLASS_TYPES:
                visit_class(node)

        parsed.imports = unique(imports)
        parsed.import_aliases = aliases
        parsed.exports = unique(
            [class_fact.name for class_fact in classes]
            + [function.name for function in functions if not function.is_declaration]
        )
        parsed.functions = functions
        parsed.classes = classes
        parsed.language_facts = {
            "has_syntax_errors": bool(root.has_error),
            "package": package,
            "annotations": unique(
                annotation
                for class_fact in classes
                for annotation in class_fact.annotations
            ),
        }
        return parsed

    @classmethod
    def _class_bases(
        cls,
        source: bytes,
        node: Node,
        kind: str,
    ) -> tuple[list[str], list[str]]:
        bases: list[str] = []
        interfaces: list[str] = []
        superclass = node.child_by_field_name("superclass")
        if superclass is not None:
            text = re.sub(r"^extends\s+", "", cls._text(source, superclass).strip())
            if text:
                bases.append(text)
        implemented = node.child_by_field_name("interfaces")
        if implemented is not None:
            text = re.sub(
                r"^(?:implements|extends)\s+",
                "",
                cls._text(source, implemented).strip(),
            )
            interfaces.extend(item.strip() for item in text.split(",") if item.strip())
        for child in node.named_children:
            if child.type == "extends_interfaces":
                text = re.sub(r"^extends\s+", "", cls._text(source, child).strip())
                targets = [item.strip() for item in text.split(",") if item.strip()]
                if kind == "interface":
                    bases.extend(targets)
                else:
                    interfaces.extend(targets)
        return unique(bases), unique(interfaces)

    def _parse_java_function(
        self,
        source: bytes,
        node: Node,
        relative_path: str,
        module_name: str,
        class_qualified: str,
        is_test: bool,
    ) -> ParsedFunction | None:
        name = self._text(source, node.child_by_field_name("name")).strip()
        if not name:
            return None
        parameters_node = node.child_by_field_name("parameters")
        parameters: list[str] = []
        parameter_types: list[str] = []
        if parameters_node is not None:
            for parameter in parameters_node.named_children:
                if parameter.type not in {
                    "formal_parameter",
                    "spread_parameter",
                    "receiver_parameter",
                }:
                    continue
                name_node = parameter.child_by_field_name("name")
                type_node = parameter.child_by_field_name("type")
                parameter_name = self._text(source, name_node).strip()
                type_name = self._text(source, type_node).strip()
                if parameter.type == "spread_parameter":
                    type_name += "..."
                parameters.append(
                    parameter_name or self._text(source, parameter).strip()
                )
                parameter_types.append(self._normalize_type(type_name))

        qualified = f"{class_qualified}.{name}"
        identity = f"{qualified}({','.join(parameter_types)})"
        body = node.child_by_field_name("body")
        snippet = self._line_text(source, node)
        signature = (
            source[node.start_byte : body.start_byte]
            .decode("utf-8", errors="replace")
            .strip()
            if body is not None
            else snippet.rstrip(";").strip()
        )
        behavior = self._behavior(source, body)
        throws_node = next(
            (child for child in node.named_children if child.type == "throws"),
            None,
        )
        declared_exceptions: list[str] = []
        if throws_node is not None:
            throws_text = re.sub(
                r"^throws\s+", "", self._text(source, throws_node).strip()
            )
            declared_exceptions = [
                item.strip() for item in throws_text.split(",") if item.strip()
            ]
        modifiers_node = self._modifiers_node(node)
        annotations = self._annotations(source, modifiers_node)
        modifiers = re.findall(
            r"\b(?:public|protected|private|static|final|abstract|synchronized|native|default)\b",
            self._text(source, modifiers_node),
        )
        ast_summary = self._ast_summary(body)
        ast_summary.update(
            {
                "parameter_types": parameter_types,
                "annotations": annotations,
                "modifiers": unique(modifiers),
                "constructor": node.type == "constructor_declaration",
            }
        )
        return ParsedFunction(
            name=name,
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
            exceptions=unique([*behavior["exceptions"], *declared_exceptions]),
            config_references=behavior["config_references"],
            ast_summary=ast_summary,
            class_qualified_name=class_qualified,
            is_test=is_test or name.startswith("test") or "Test" in annotations,
            segments=self._segments(source, body, qualified),
            symbol_identity=identity,
            is_declaration=body is None,
            language=self.language,
        )

    @classmethod
    def _annotations(cls, source: bytes, modifiers: Node | None) -> list[str]:
        if modifiers is None:
            return []
        result: list[str] = []
        for node in cls._walk(modifiers):
            if node.type in {"annotation", "marker_annotation"}:
                name = node.child_by_field_name("name")
                value = cls._text(source, name).strip()
                if value:
                    result.append(value)
        return unique(result)

    @staticmethod
    def _modifiers_node(node: Node) -> Node | None:
        return next(
            (child for child in node.named_children if child.type == "modifiers"),
            None,
        )

    @staticmethod
    def _normalize_type(value: str) -> str:
        return re.sub(r"\s+", "", value)


__all__ = ["JavaParser"]
