"""Go repository parser backed by the bundled tree-sitter Go grammar."""

from __future__ import annotations

import re
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .tree_sitter import TreeSitterParserBase, unique


class GoParser(TreeSitterParserBase):
    """Extract packages, types, receiver methods, calls, and control segments."""

    language = "go"
    grammar_name = "go"
    call_node_types = frozenset({"call_expression"})
    nested_function_types = frozenset(
        {"function_declaration", "method_declaration", "func_literal"}
    )
    branch_node_types = frozenset(
        {
            "if_statement",
            "expression_switch_statement",
            "type_switch_statement",
            "select_statement",
        }
    )
    loop_node_types = frozenset({"for_statement"})
    return_node_types = frozenset({"return_statement"})
    segment_control_types = {
        "if_statement": "branch",
        "expression_switch_statement": "branch",
        "type_switch_statement": "branch",
        "select_statement": "branch",
        "for_statement": "loop",
    }

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        tree = self._tree(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        package = self._package(source_bytes, root)
        if package:
            parsed.module_name = package
        module_name = parsed.module_name
        imports, aliases = self._imports(source_bytes, root)
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []
        types_by_name: dict[str, ParsedClass] = {}
        named_types: list[str] = []
        type_aliases: list[str] = []

        for declaration in root.named_children:
            if declaration.type != "type_declaration":
                continue
            for node in declaration.named_children:
                if node.type == "type_alias":
                    name = self._text(
                        source_bytes, node.child_by_field_name("name")
                    ).strip()
                    if name:
                        type_aliases.append(name)
                    continue
                if node.type != "type_spec":
                    continue
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                type_node = node.child_by_field_name("type")
                if not name or type_node is None:
                    continue
                if type_node.type not in {"struct_type", "interface_type"}:
                    named_types.append(name)
                    continue
                qualified = f"{module_name}.{name}" if module_name else name
                snippet = self._line_text(source_bytes, declaration)
                class_fact = ParsedClass(
                    name=name,
                    qualified_name=qualified,
                    file_path=relative_path,
                    source_code=snippet,
                    start_line=self._start_line(declaration),
                    end_line=self._end_line(declaration),
                    content_hash=normalized_source_hash(snippet),
                    docstring=self._comment_before(source_bytes, declaration),
                    kind="struct" if type_node.type == "struct_type" else "interface",
                    is_test=parsed.is_test or name.endswith(("Test", "Suite")),
                    symbol_identity=qualified,
                )
                classes.append(class_fact)
                types_by_name[name] = class_fact
                if type_node.type == "interface_type":
                    for member in self._nodes(type_node, "method_elem"):
                        function = self._parse_function(
                            source_bytes,
                            member,
                            relative_path,
                            module_name,
                            parsed.is_test,
                            owner=class_fact,
                            receiver="",
                        )
                        if function is not None:
                            functions.append(function)
                            class_fact.methods.append(function.symbol_identity)

        for node in root.named_children:
            if node.type not in {"function_declaration", "method_declaration"}:
                continue
            receiver = ""
            owner = None
            if node.type == "method_declaration":
                receiver_node = node.child_by_field_name("receiver")
                receiver = self._receiver_type(source_bytes, receiver_node)
                owner = types_by_name.get(receiver)
            function = self._parse_function(
                source_bytes,
                node,
                relative_path,
                module_name,
                parsed.is_test,
                owner=owner,
                receiver=receiver,
            )
            if function is not None:
                functions.append(function)
                if owner is not None:
                    owner.methods.append(function.symbol_identity)

        parsed.imports = imports
        parsed.import_aliases = aliases
        parsed.exports = unique(
            [item.name for item in classes if item.name[:1].isupper()]
            + [item.name for item in functions if item.name[:1].isupper()]
        )
        parsed.functions = functions
        parsed.classes = classes
        parsed.language_facts = {
            "has_syntax_errors": bool(root.has_error),
            "package": package,
            "named_types": unique(named_types),
            "type_aliases": unique(type_aliases),
        }
        return parsed

    def _parse_function(
        self,
        source: bytes,
        node: Node,
        relative_path: str,
        module_name: str,
        file_is_test: bool,
        *,
        owner: ParsedClass | None,
        receiver: str,
    ) -> ParsedFunction | None:
        name = self._text(source, node.child_by_field_name("name")).strip()
        if not name:
            return None
        owner_name = owner.name if owner is not None else receiver
        prefix = ".".join(part for part in (module_name, owner_name) if part)
        qualified = f"{prefix}.{name}" if prefix else name
        parameters_node = node.child_by_field_name("parameters")
        parameters, parameter_types, variadic = self._parameters(
            source, parameters_node
        )
        result_node = node.child_by_field_name("result")
        return_type = self._text(source, result_node).strip()
        return_values = self._return_values(source, result_node)
        body = node.child_by_field_name("body")
        snippet = self._line_text(source, node)
        signature = (
            source[node.start_byte : body.start_byte]
            .decode("utf-8", errors="replace")
            .strip()
            if body is not None
            else self._text(source, node).strip().rstrip(";")
        )
        behavior = self._behavior(source, body)
        self._augment_behavior(source, body, behavior)
        segments = self._segments(source, body, qualified)
        call_sites = self._call_sites(source, body, segments)
        summary = self._ast_summary(body)
        node_types = [item.type for item in self._walk(body)] if body else []
        summary.update(
            {
                "parameter_types": parameter_types,
                "receiver": self._text(
                    source, node.child_by_field_name("receiver")
                ).strip(),
                "receiver_type": receiver,
                "return_values": return_values,
                "multiple_returns": len(return_values) > 1,
                "variadic": variadic,
                "goroutines": node_types.count("go_statement"),
                "defers": node_types.count("defer_statement"),
                "channel_operations": sum(
                    item in {"send_statement", "receive_expression"}
                    for item in node_types
                ),
                "selects": node_types.count("select_statement"),
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
            return_type=return_type,
            reads=behavior["reads"],
            writes=behavior["writes"],
            calls=behavior["calls"],
            call_sites=call_sites,
            side_effects=behavior["side_effects"],
            exceptions=behavior["exceptions"],
            config_references=behavior["config_references"],
            ast_summary=summary,
            class_qualified_name=(
                f"{module_name}.{owner_name}" if owner_name and module_name else owner_name
            ),
            is_test=file_is_test or name.startswith("Test"),
            segments=segments,
            symbol_identity=qualified,
            is_declaration=body is None,
            language=self.language,
        )

    @classmethod
    def _package(cls, source: bytes, root: Node) -> str:
        clause = next(
            (child for child in root.named_children if child.type == "package_clause"),
            None,
        )
        if clause is None:
            return ""
        identifier = next(cls._nodes(clause, "package_identifier"), None)
        return cls._text(source, identifier).strip()

    @classmethod
    def _imports(cls, source: bytes, root: Node) -> tuple[list[str], dict[str, str]]:
        imports: list[str] = []
        aliases: dict[str, str] = {}
        for node in cls._nodes(root, "import_spec"):
            path = cls._text(source, node.child_by_field_name("path")).strip('"`')
            name = cls._text(source, node.child_by_field_name("name")).strip()
            if path:
                imports.append(path)
                aliases[name or path.rsplit("/", 1)[-1]] = path
        for declaration in cls._nodes(root, "import_declaration"):
            if any(item.type == "import_spec" for item in cls._walk(declaration)):
                continue
            path = next(
                cls._nodes(
                    declaration,
                    "interpreted_string_literal",
                    "raw_string_literal",
                ),
                None,
            )
            value = cls._text(source, path).strip('"`')
            if value:
                imports.append(value)
                aliases[value.rsplit("/", 1)[-1]] = value
        return unique(imports), dict(sorted(aliases.items()))

    @classmethod
    def _receiver_type(cls, source: bytes, receiver: Node | None) -> str:
        if receiver is None:
            return ""
        type_names = [
            cls._text(source, node).strip()
            for node in cls._nodes(receiver, "type_identifier")
        ]
        return type_names[0] if type_names else ""

    @classmethod
    def _parameters(
        cls, source: bytes, parameters: Node | None
    ) -> tuple[list[str], list[str], bool]:
        if parameters is None:
            return [], [], False
        names: list[str] = []
        types: list[str] = []
        variadic = False
        for parameter in parameters.named_children:
            if parameter.type not in {
                "parameter_declaration",
                "variadic_parameter_declaration",
            }:
                continue
            type_node = parameter.child_by_field_name("type")
            type_text = cls._text(source, type_node).strip()
            variadic_parameter = parameter.type == "variadic_parameter_declaration"
            if variadic_parameter:
                type_text = f"...{type_text}"
                variadic = True
            parameter_names = [
                cls._text(source, child).strip()
                for child in parameter.named_children
                if child.type == "identifier"
            ]
            if parameter_names:
                names.extend(parameter_names)
                types.extend(type_text for _ in parameter_names)
            else:
                raw = cls._text(source, parameter).strip()
                names.append(raw)
                types.append(type_text or raw)
        return names, [re.sub(r"\s+", "", item) for item in types], variadic

    @classmethod
    def _return_values(cls, source: bytes, result: Node | None) -> list[str]:
        if result is None:
            return []
        if result.type != "parameter_list":
            value = cls._text(source, result).strip()
            return [value] if value else []
        values = [
            cls._text(source, child.child_by_field_name("type") or child).strip()
            for child in result.named_children
        ]
        return [value for value in values if value]

    @classmethod
    def _augment_behavior(
        cls, source: bytes, body: Node | None, behavior: dict[str, list[str]]
    ) -> None:
        if body is None:
            return
        writes = set(behavior["writes"])
        configs = set(behavior["config_references"])
        effects = set(behavior["side_effects"])
        exceptions = set(behavior["exceptions"])
        for node in cls._walk(body, stop_types=tuple(cls.nested_function_types)):
            if node.type in {
                "assignment_statement",
                "short_var_declaration",
                "inc_statement",
                "dec_statement",
            }:
                left = node.child_by_field_name("left") or node
                for identifier in cls._nodes(left, "identifier"):
                    writes.add(cls._text(source, identifier).strip())
            elif node.type in {"go_statement", "send_statement"}:
                effects.add("concurrency")
        for call_node in cls._nodes(body, "call_expression"):
            call = cls._call_name(source, call_node)
            lowered = call.lower()
            arguments = call_node.child_by_field_name("arguments")
            literal = (
                next(
                    cls._nodes(
                        arguments,
                        "interpreted_string_literal",
                        "raw_string_literal",
                    ),
                    None,
                )
                if arguments is not None
                else None
            )
            key = cls._text(source, literal).strip('"`')
            if lowered in {"os.getenv", "os.lookupenv"} and key:
                configs.add(f"environment.{key}")
                effects.add("environment")
            if lowered == "panic":
                exceptions.add("panic")
        behavior["writes"] = unique(writes)
        behavior["config_references"] = unique(configs)
        behavior["side_effects"] = unique(effects)
        behavior["exceptions"] = unique(exceptions)


__all__ = ["GoParser"]
