"""Shared JavaScript/TypeScript repository parsing built on tree-sitter."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .tree_sitter import TreeSitterParserBase, unique


class JsTsParserBase(TreeSitterParserBase):
    """Language-neutral ECMAScript-family extraction shared by JS and TS."""

    call_node_types = frozenset({"call_expression"})
    nested_function_types = frozenset(
        {
            "function_declaration",
            "generator_function_declaration",
            "function_expression",
            "generator_function",
            "arrow_function",
            "method_definition",
        }
    )
    branch_node_types = frozenset(
        {"if_statement", "switch_statement", "ternary_expression"}
    )
    loop_node_types = frozenset(
        {
            "for_statement",
            "for_in_statement",
            "while_statement",
            "do_statement",
        }
    )
    return_node_types = frozenset({"return_statement"})
    await_node_types = frozenset({"await_expression"})
    try_node_types = frozenset({"try_statement"})
    segment_control_types = {
        "if_statement": "branch",
        "switch_statement": "branch",
        "ternary_expression": "branch",
        "for_statement": "loop",
        "for_in_statement": "loop",
        "while_statement": "loop",
        "do_statement": "loop",
        "try_statement": "exception",
    }

    _CLASS_TYPES = {
        "class_declaration": "class",
        "class": "class",
        "abstract_class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "type_alias_declaration": "type_alias",
    }
    _FUNCTION_TYPES = {
        "function_declaration",
        "generator_function_declaration",
        "function_expression",
        "generator_function",
        "function_signature",
        "method_definition",
        "method_signature",
        "abstract_method_signature",
    }

    def _grammar_for_path(self, relative_path: str) -> str:
        del relative_path
        return self.grammar_name or self.language

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        grammar = self._grammar_for_path(relative_path)
        tree = self._parser_for(grammar).parse(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        module_name = parsed.module_name

        imports, aliases = self._imports(source_bytes, root)
        exports = self._exports(source_bytes, root)
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []
        class_nodes: dict[tuple[int, int], ParsedClass] = {}

        for node in self._walk(root):
            if node.type not in self._CLASS_TYPES:
                continue
            name = self._text(source_bytes, node.child_by_field_name("name")).strip()
            if not name and node.parent is not None and node.parent.type == "variable_declarator":
                name = self._text(
                    source_bytes, node.parent.child_by_field_name("name")
                ).strip()
            if not name and self._is_default_export(node):
                name = "default"
            if not name:
                continue
            owner = self._nearest_class_fact(node, class_nodes)
            qualified = ".".join(
                part for part in (owner.qualified_name if owner else module_name, name) if part
            )
            kind = self._CLASS_TYPES[node.type]
            snippet = self._line_text(source_bytes, node)
            bases, interfaces = self._heritage(source_bytes, node, kind)
            annotations = self._decorators(source_bytes, node)
            if node.type == "abstract_class_declaration":
                annotations.append("abstract")
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
                is_test=parsed.is_test or name.endswith(("Test", "Tests", "Spec")),
                annotations=unique(annotations),
                symbol_identity=qualified,
            )
            classes.append(class_fact)
            class_nodes[(node.start_byte, node.end_byte)] = class_fact

        for node in self._walk(root):
            source_node = node
            function_node = node
            explicit_name = ""
            if node.type == "variable_declarator":
                value = node.child_by_field_name("value")
                if value is None or value.type not in {
                    "arrow_function",
                    "function_expression",
                    "generator_function",
                }:
                    continue
                function_node = value
                explicit_name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
            elif node.type in {"function_expression", "generator_function"}:
                parent = node.parent
                if parent is not None and self._is_default_export(node):
                    explicit_name = "default"
                elif parent is None or parent.type != "assignment_expression":
                    continue
                else:
                    left = self._text(
                        source_bytes, parent.child_by_field_name("left")
                    ).strip()
                    explicit_name = self._text(
                        source_bytes, node.child_by_field_name("name")
                    ).strip()
                    if not explicit_name:
                        explicit_name = (
                            "default"
                            if left == "module.exports"
                            else left.rsplit(".", 1)[-1]
                        )
                    source_node = parent
            elif node.type not in self._FUNCTION_TYPES:
                continue
            elif not self._text(
                source_bytes, node.child_by_field_name("name")
            ).strip() and self._is_default_export(node):
                explicit_name = "default"

            owner = self._nearest_class_fact(function_node, class_nodes)
            function = self._parse_function(
                source_bytes,
                function_node,
                source_node,
                relative_path,
                module_name,
                parsed.is_test,
                owner,
                explicit_name=explicit_name,
            )
            if function is None:
                continue
            functions.append(function)
            if owner is not None:
                owner.methods.append(function.symbol_identity)

        parsed.imports = imports
        parsed.import_aliases = aliases
        parsed.exports = exports
        parsed.functions = functions
        parsed.classes = classes
        parsed.language_facts = {
            "has_syntax_errors": bool(root.has_error),
            "grammar": grammar,
            "module_systems": self._module_systems(source_bytes, root),
            "declaration_file": relative_path.endswith(".d.ts"),
        }
        return parsed

    def _parse_function(
        self,
        source: bytes,
        node: Node,
        source_node: Node,
        relative_path: str,
        module_name: str,
        file_is_test: bool,
        owner: ParsedClass | None,
        *,
        explicit_name: str = "",
    ) -> ParsedFunction | None:
        name = explicit_name or self._text(
            source, node.child_by_field_name("name")
        ).strip()
        if not name and node.type == "method_definition":
            name = self._identifier(source, node.child_by_field_name("name"))
        if not name:
            return None
        prefix = owner.qualified_name if owner else module_name
        qualified = f"{prefix}.{name}" if prefix else name
        parameters_node = node.child_by_field_name("parameters")
        parameters, parameter_types, optional_count, variadic = self._parameters(
            source, parameters_node
        )
        body = node.child_by_field_name("body")
        is_declaration = body is None or node.type in {
            "function_signature",
            "method_signature",
            "abstract_method_signature",
        }
        snippet = self._line_text(source, source_node)
        signature = self._signature(source, node, source_node, body, name)
        return_type = self._text(
            source, node.child_by_field_name("return_type")
        ).strip()
        return_type = return_type.removeprefix(":").strip()
        behavior = self._behavior(source, body)
        self._augment_behavior(source, body, behavior)
        segments = self._segments(source, body, qualified)
        call_sites = self._call_sites(source, body, segments)
        is_async = bool(re.search(r"\basync\b", signature))
        is_generator = "generator" in node.type or bool(
            re.search(r"\bfunction\s*\*", signature)
        )
        identity = qualified
        if self.language == "typescript":
            identity = f"{qualified}({','.join(parameter_types)})"
            if is_declaration:
                identity += "@declaration"
        summary = self._ast_summary(body)
        summary.update(
            {
                "parameter_types": parameter_types,
                "optional_parameters": optional_count,
                "variadic": variadic,
                "generator": is_generator,
                "type_parameters": self._text(
                    source, node.child_by_field_name("type_parameters")
                ).strip(),
            }
        )
        return ParsedFunction(
            name=name,
            qualified_name=qualified,
            signature=signature,
            file_path=relative_path,
            module_name=module_name,
            source_code=snippet,
            start_line=self._start_line(source_node),
            end_line=self._end_line(source_node),
            content_hash=normalized_source_hash(snippet),
            docstring=self._comment_before(source, source_node),
            parameters=parameters,
            return_type=return_type,
            reads=behavior["reads"],
            writes=behavior["writes"],
            calls=behavior["calls"],
            call_sites=call_sites,
            side_effects=behavior["side_effects"],
            exceptions=behavior["exceptions"],
            config_references=behavior["config_references"],
            ast_summary=summary,
            class_qualified_name=owner.qualified_name if owner else "",
            is_async=is_async,
            is_test=(
                file_is_test
                or name.lower().startswith(("test", "it", "describe"))
                or (owner is not None and owner.is_test)
            ),
            segments=segments,
            symbol_identity=identity,
            is_declaration=is_declaration,
            language=self.language,
        )

    @classmethod
    def _parameters(
        cls, source: bytes, parameters_node: Node | None
    ) -> tuple[list[str], list[str], int, bool]:
        if parameters_node is None:
            return [], [], 0, False
        parameters: list[str] = []
        parameter_types: list[str] = []
        optional_count = 0
        variadic = False
        for parameter in parameters_node.named_children:
            if parameter.type in {"comment", "type_parameter"}:
                continue
            pattern = parameter.child_by_field_name("pattern")
            if pattern is None:
                pattern = parameter.child_by_field_name("name")
            name = cls._text(source, pattern).strip()
            raw = cls._text(source, parameter).strip()
            if not name:
                name = re.split(r"[?:=]", raw, maxsplit=1)[0].strip()
            if name:
                parameters.append(name)
            type_node = parameter.child_by_field_name("type")
            type_text = cls._text(source, type_node).strip().removeprefix(":").strip()
            parameter_types.append(re.sub(r"\s+", "", type_text or "any"))
            optional_count += int(parameter.type == "optional_parameter" or "?" in raw)
            variadic = variadic or parameter.type == "rest_pattern" or raw.startswith("...")
        return parameters, parameter_types, optional_count, variadic

    @classmethod
    def _signature(
        cls,
        source: bytes,
        node: Node,
        source_node: Node,
        body: Node | None,
        name: str,
    ) -> str:
        if body is not None:
            start = source_node.start_byte
            end = body.start_byte
            prefix = source[start:end].decode("utf-8", errors="replace").strip()
            if prefix:
                return re.sub(r"\s*=>\s*$", "", prefix).strip()
        raw = cls._text(source, source_node).strip().rstrip(";")
        return raw or name

    @classmethod
    def _imports(cls, source: bytes, root: Node) -> tuple[list[str], dict[str, str]]:
        imports: list[str] = []
        aliases: dict[str, str] = {}
        for node in cls._walk(root):
            if node.type == "import_statement":
                module = cls._text(source, node.child_by_field_name("source")).strip(
                    "\"'"
                )
                if not module:
                    continue
                imports.append(module)
                for specifier in cls._walk(node):
                    if specifier.type == "import_specifier":
                        imported = cls._text(
                            source, specifier.child_by_field_name("name")
                        ).strip()
                        local = cls._text(
                            source, specifier.child_by_field_name("alias")
                        ).strip()
                        if imported:
                            aliases[local or imported] = f"{module}.{imported}"
                clause = next(
                    (child for child in node.named_children if child.type == "import_clause"),
                    None,
                )
                if clause is not None:
                    for child in clause.named_children:
                        if child.type == "identifier":
                            local = cls._text(source, child).strip()
                            if local:
                                aliases.setdefault(local, module)
            elif node.type == "call_expression" and cls._call_name(source, node) == "require":
                arguments = node.child_by_field_name("arguments")
                literal = next(cls._nodes(arguments, "string", "string_literal"), None) if arguments else None
                module = cls._text(source, literal).strip("\"'")
                if module:
                    imports.append(module)
                    parent = node.parent
                    if parent is not None and parent.type == "variable_declarator":
                        local = cls._text(
                            source, parent.child_by_field_name("name")
                        ).strip()
                        if local:
                            aliases[local] = module
        return unique(imports), dict(sorted(aliases.items()))

    @classmethod
    def _exports(cls, source: bytes, root: Node) -> list[str]:
        exports: list[str] = []
        for node in cls._walk(root):
            if node.type == "export_statement":
                declaration = node.child_by_field_name("declaration")
                variable_declaration = declaration is not None and declaration.type in {
                    "lexical_declaration",
                    "variable_declaration",
                }
                if variable_declaration:
                    for variable in declaration.named_children:
                        if variable.type != "variable_declarator":
                            continue
                        value = cls._text(
                            source, variable.child_by_field_name("name")
                        ).strip()
                        if value:
                            exports.append(value)
                name = (
                    ""
                    if variable_declaration
                    else cls._text(
                        source,
                        declaration.child_by_field_name("name")
                        if declaration is not None
                        else None,
                    ).strip()
                )
                if not name and re.search(
                    r"^export\s+default\b", cls._text(source, node).strip()
                ):
                    name = "default"
                elif not name and declaration is not None and not variable_declaration:
                    name = cls._identifier(source, declaration)
                if name:
                    exports.append(name)
                for specifier in cls._nodes(node, "export_specifier"):
                    alias = specifier.child_by_field_name("alias")
                    original = specifier.child_by_field_name("name")
                    value = cls._text(source, alias or original).strip()
                    if value:
                        exports.append(value)
            elif node.type == "assignment_expression":
                left = cls._text(source, node.child_by_field_name("left")).strip()
                if left == "module.exports":
                    exports.append("default")
                elif left.startswith("exports.") or left.startswith("module.exports."):
                    exports.append(left.rsplit(".", 1)[-1])
        return unique(exports)

    @classmethod
    def _heritage(
        cls, source: bytes, node: Node, kind: str
    ) -> tuple[list[str], list[str]]:
        bases: list[str] = []
        interfaces: list[str] = []
        for child in cls._walk(node, stop_types=("class_body", "interface_body")):
            text = cls._text(source, child).strip()
            if child.type == "extends_type_clause":
                bases.extend(cls._heritage_targets(text.removeprefix("extends")))
            elif child.type == "implements_clause":
                interfaces.extend(cls._heritage_targets(text.removeprefix("implements")))
            elif child.type == "class_heritage":
                match = re.search(r"\bextends\s+([^\s{]+)", text)
                if match:
                    bases.append(match.group(1))
        if kind == "interface" and interfaces:
            bases.extend(interfaces)
            interfaces = []
        return unique(bases), unique(interfaces)

    @staticmethod
    def _heritage_targets(value: str) -> list[str]:
        return [
            re.sub(r"<.*>", "", item).strip()
            for item in value.split(",")
            if item.strip()
        ]

    @classmethod
    def _decorators(cls, source: bytes, node: Node) -> list[str]:
        return unique(
            cls._text(source, decorator).strip().removeprefix("@").split("(", 1)[0]
            for decorator in node.named_children
            if decorator.type == "decorator"
        )

    @classmethod
    def _nearest_class_fact(
        cls,
        node: Node,
        class_nodes: dict[tuple[int, int], ParsedClass],
    ) -> ParsedClass | None:
        parent = node.parent
        while parent is not None:
            found = class_nodes.get((parent.start_byte, parent.end_byte))
            if found is not None:
                return found
            parent = parent.parent
        return None

    @classmethod
    def _is_default_export(cls, node: Node) -> bool:
        parent = node.parent
        return bool(
            parent is not None
            and parent.type == "export_statement"
            and any(
                child.type == "default"
                for child in parent.children
                if not child.is_named
            )
        )

    @classmethod
    def _augment_behavior(
        cls, source: bytes, body: Node | None, behavior: dict[str, list[str]]
    ) -> None:
        if body is None:
            return
        configs = set(behavior["config_references"])
        side_effects = set(behavior["side_effects"])
        for node in cls._walk(body, stop_types=tuple(cls.nested_function_types)):
            text = cls._text(source, node).strip()
            if node.type == "member_expression":
                match = re.fullmatch(r"process\.env\.([A-Za-z_$][\w$]*)", text)
                if match:
                    configs.add(f"environment.{match.group(1)}")
                    side_effects.add("environment")
                match = re.fullmatch(
                    r"(?:config|settings|options)\.([A-Za-z_$][\w$]*)", text,
                    flags=re.IGNORECASE,
                )
                if match:
                    configs.add(f"config.{match.group(1)}")
        for call in behavior["calls"]:
            lowered = call.lower()
            if lowered in {"fetch", "axios", "request"} or lowered.startswith(
                ("axios.", "http.", "https.")
            ):
                side_effects.add("network")
        behavior["config_references"] = unique(configs)
        behavior["side_effects"] = unique(side_effects)

    @classmethod
    def _module_systems(cls, source: bytes, root: Node) -> list[str]:
        systems: set[str] = set()
        for node in cls._walk(root):
            if node.type in {"import_statement", "export_statement"}:
                systems.add("esm")
            elif node.type == "call_expression" and cls._call_name(source, node) == "require":
                systems.add("commonjs")
            elif node.type == "assignment_expression" and "module.exports" in cls._text(source, node):
                systems.add("commonjs")
        return sorted(systems)


class JavaScriptParser(JsTsParserBase):
    """Parse JavaScript, JSX, ESM, and CommonJS source files."""

    language = "javascript"
    grammar_name = "javascript"


__all__ = ["JavaScriptParser", "JsTsParserBase"]
