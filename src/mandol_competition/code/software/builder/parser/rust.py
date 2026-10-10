"""Rust repository parser backed by the bundled tree-sitter Rust grammar."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath

from tree_sitter import Node

from .base import ParsedClass, ParsedFile, ParsedFunction, normalized_source_hash
from .tree_sitter import TreeSitterParserBase, unique


class RustParser(TreeSitterParserBase):
    """Extract Rust items with conservative impl/trait method identities."""

    language = "rust"
    grammar_name = "rust"
    call_node_types = frozenset({"call_expression", "macro_invocation"})
    nested_function_types = frozenset({"function_item", "closure_expression"})
    branch_node_types = frozenset({"if_expression", "match_expression"})
    loop_node_types = frozenset(
        {"for_expression", "while_expression", "loop_expression"}
    )
    return_node_types = frozenset({"return_expression"})
    await_node_types = frozenset({"await_expression"})
    segment_control_types = {
        "if_expression": "branch",
        "match_expression": "branch",
        "for_expression": "loop",
        "while_expression": "loop",
        "loop_expression": "loop",
    }
    _CLASS_TYPES = {
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
    }

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        relative_path = PurePosixPath(relative_path).as_posix()
        source_bytes = source.encode("utf-8")
        tree = self._tree(source_bytes)
        root = tree.root_node
        parsed = self._empty_file(source, relative_path, root)
        module_name = parsed.module_name
        imports, aliases = self._uses(source_bytes, root)
        module_declarations: list[str] = []
        classes: list[ParsedClass] = []
        functions: list[ParsedFunction] = []
        types_by_name: dict[tuple[str, str], ParsedClass] = {}
        scopes = self._scopes(source_bytes, root, module_name, module_declarations)

        for scope, scope_module in scopes:
            for node in scope.named_children:
                if node.type not in self._CLASS_TYPES:
                    continue
                name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                if not name:
                    continue
                qualified = f"{scope_module}.{name}" if scope_module else name
                snippet = self._line_text(source_bytes, node)
                class_fact = ParsedClass(
                    name=name,
                    qualified_name=qualified,
                    file_path=relative_path,
                    source_code=snippet,
                    start_line=self._start_line(node),
                    end_line=self._end_line(node),
                    content_hash=normalized_source_hash(snippet),
                    docstring=self._comment_before(source_bytes, node),
                    kind=self._CLASS_TYPES[node.type],
                    is_test=parsed.is_test or name.endswith(("Test", "Tests")),
                    annotations=self._item_annotations(source_bytes, node),
                    symbol_identity=qualified,
                )
                classes.append(class_fact)
                types_by_name[(scope_module, name)] = class_fact

        # Trait declarations are class-owned declaration functions.
        for scope, scope_module in scopes:
            for node in scope.named_children:
                if node.type != "trait_item":
                    continue
                trait_name = self._text(
                    source_bytes, node.child_by_field_name("name")
                ).strip()
                owner = types_by_name.get((scope_module, trait_name))
                body = node.child_by_field_name("body")
                if owner is None or body is None:
                    continue
                for member in body.named_children:
                    if member.type not in {
                        "function_signature_item",
                        "function_item",
                    }:
                        continue
                    function = self._parse_function(
                        source_bytes,
                        member,
                        relative_path,
                        scope_module,
                        parsed.is_test,
                        owner_name=trait_name,
                        owner=owner,
                        trait_name=trait_name,
                        trait_declaration=True,
                    )
                    if function is not None:
                        functions.append(function)
                        owner.methods.append(function.symbol_identity)
                        self._append_nested_functions(
                            source_bytes,
                            member,
                            function,
                            relative_path,
                            parsed.is_test,
                            functions,
                        )

        impls: list[dict[str, str]] = []
        for scope, scope_module in scopes:
            for node in scope.named_children:
                if node.type != "impl_item":
                    continue
                type_name = self._type_identity(
                    source_bytes, node.child_by_field_name("type")
                )
                trait_name = self._type_identity(
                    source_bytes, node.child_by_field_name("trait")
                )
                if not type_name:
                    continue
                impls.append(
                    {"module": scope_module, "type": type_name, "trait": trait_name}
                )
                owner = types_by_name.get(
                    (scope_module, self._simple_type_name(type_name))
                )
                if owner is not None and trait_name:
                    owner.interfaces = unique([*owner.interfaces, trait_name])
                body = node.child_by_field_name("body")
                if body is None:
                    continue
                for member in body.named_children:
                    if member.type != "function_item":
                        continue
                    function = self._parse_function(
                        source_bytes,
                        member,
                        relative_path,
                        scope_module,
                        parsed.is_test,
                        owner_name=type_name,
                        owner=owner,
                        trait_name=trait_name,
                        trait_declaration=False,
                    )
                    if function is not None:
                        functions.append(function)
                        if owner is not None:
                            owner.methods.append(function.symbol_identity)
                        self._append_nested_functions(
                            source_bytes,
                            member,
                            function,
                            relative_path,
                            parsed.is_test,
                            functions,
                        )

        for scope, scope_module in scopes:
            for node in scope.named_children:
                if node.type != "function_item":
                    continue
                function = self._parse_function(
                    source_bytes,
                    node,
                    relative_path,
                    scope_module,
                    parsed.is_test,
                    owner_name="",
                    owner=None,
                    trait_name="",
                    trait_declaration=False,
                )
                if function is not None:
                    functions.append(function)
                    self._append_nested_functions(
                        source_bytes,
                        node,
                        function,
                        relative_path,
                        parsed.is_test,
                        functions,
                    )

        parsed.imports = imports
        parsed.import_aliases = aliases
        parsed.exports = unique(
            [item.name for item in classes if self._is_public_item(source_bytes, item)]
            + [
                item.name
                for item in functions
                if item.signature.lstrip().startswith("pub ")
            ]
        )
        parsed.functions = functions
        parsed.classes = classes
        parsed.language_facts = {
            "has_syntax_errors": bool(root.has_error),
            "modules": unique(module_declarations),
            "impls": impls,
            "macro_invocations": unique(
                self._call_name(source_bytes, node)
                for node in self._nodes(root, "macro_invocation")
            ),
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
        owner_name: str,
        owner: ParsedClass | None,
        trait_name: str,
        trait_declaration: bool,
    ) -> ParsedFunction | None:
        name = self._text(source, node.child_by_field_name("name")).strip()
        if not name:
            return None
        owner_identity = self._simple_type_name(owner_name)
        prefix = ".".join(part for part in (module_name, owner_identity) if part)
        qualified = f"{prefix}.{name}" if prefix else name
        parameters_node = node.child_by_field_name("parameters")
        parameters, parameter_types = self._parameters(source, parameters_node)
        return_type = self._text(
            source, node.child_by_field_name("return_type")
        ).strip()
        body = node.child_by_field_name("body")
        snippet = self._line_text(source, node)
        signature = (
            source[node.start_byte : body.start_byte]
            .decode("utf-8", errors="replace")
            .strip()
            if body is not None
            else self._text(source, node).rstrip(";").strip()
        )
        behavior = self._behavior(source, body)
        self._augment_behavior(source, body, behavior)
        segments = self._segments(source, body, qualified)
        call_sites = self._call_sites(source, body, segments)
        summary = self._ast_summary(body)
        raw_header = signature.lower()
        attributes = self._item_annotations(source, node)
        summary.update(
            {
                "parameter_types": parameter_types,
                "trait_impl": trait_name if owner_name and not trait_declaration else "",
                "trait_declaration": trait_declaration,
                "unsafe": bool(re.search(r"\bunsafe\b", raw_header)),
                "const": bool(re.search(r"\bconst\s+fn\b", raw_header)),
                "attributes": attributes,
                "macros": unique(
                    self._call_name(source, call)
                    for call in self._nodes(body, "macro_invocation")
                )
                if body is not None
                else [],
            }
        )
        identity = qualified
        if trait_declaration:
            identity = f"{qualified}@trait"
        elif trait_name:
            identity = f"{qualified}@impl:{trait_name}"
        cfg_attributes = [
            attribute for attribute in attributes if attribute.startswith("#[cfg")
        ]
        if cfg_attributes:
            cfg_hash = hashlib.sha256(
                "\n".join(cfg_attributes).encode("utf-8")
            ).hexdigest()[:12]
            identity = f"{identity}@cfg:{cfg_hash}"
        class_qualified = (
            owner.qualified_name
            if owner is not None
            else (
                f"{module_name}.{owner_identity}"
                if owner_identity and module_name
                else owner_identity
            )
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
            class_qualified_name=class_qualified,
            is_async=bool(re.search(r"\basync\s+fn\b", raw_header)),
            is_test=(
                file_is_test
                or name.startswith("test_")
                or any(attribute.startswith("#[test") for attribute in attributes)
            ),
            segments=segments,
            symbol_identity=identity,
            is_declaration=body is None,
            language=self.language,
        )

    def _append_nested_functions(
        self,
        source: bytes,
        parent_node: Node,
        parent_function: ParsedFunction,
        relative_path: str,
        file_is_test: bool,
        functions: list[ParsedFunction],
    ) -> None:
        body = parent_node.child_by_field_name("body")
        if body is None:
            return
        for node in self._walk(body, stop_types=("function_item",)):
            if node is body or node.type != "function_item":
                continue
            nested = self._parse_function(
                source,
                node,
                relative_path,
                f"{parent_function.qualified_name}.<locals>",
                file_is_test or parent_function.is_test,
                owner_name="",
                owner=None,
                trait_name="",
                trait_declaration=False,
            )
            if nested is None:
                continue
            nested.parent_function_qualified_name = parent_function.qualified_name
            functions.append(nested)
            self._append_nested_functions(
                source,
                node,
                nested,
                relative_path,
                file_is_test,
                functions,
            )

    @classmethod
    def _uses(cls, source: bytes, root: Node) -> tuple[list[str], dict[str, str]]:
        imports: list[str] = []
        aliases: dict[str, str] = {}
        for node in cls._nodes(root, "use_declaration"):
            argument = node.child_by_field_name("argument")
            value = cls._text(source, argument).strip()
            if value:
                imports.append(value)
            for alias_node in cls._nodes(node, "use_as_clause"):
                path = cls._text(
                    source, alias_node.child_by_field_name("path")
                ).strip()
                alias = cls._text(
                    source, alias_node.child_by_field_name("alias")
                ).strip()
                if path and alias:
                    aliases[alias] = path
        return unique(imports), dict(sorted(aliases.items()))

    @classmethod
    def _scopes(
        cls,
        source: bytes,
        root: Node,
        module_name: str,
        module_declarations: list[str],
    ) -> list[tuple[Node, str]]:
        """Return root and inline-module bodies with deterministic module names."""
        result: list[tuple[Node, str]] = []

        def visit(scope: Node, prefix: str) -> None:
            result.append((scope, prefix))
            for item in scope.named_children:
                if item.type != "mod_item":
                    continue
                name = cls._text(
                    source, item.child_by_field_name("name")
                ).strip()
                if not name:
                    continue
                qualified = f"{prefix}.{name}" if prefix else name
                module_declarations.append(qualified)
                body = item.child_by_field_name("body")
                if body is not None:
                    visit(body, qualified)

        visit(root, module_name)
        return result

    @classmethod
    def _parameters(
        cls, source: bytes, parameters: Node | None
    ) -> tuple[list[str], list[str]]:
        if parameters is None:
            return [], []
        names: list[str] = []
        types: list[str] = []
        for parameter in parameters.named_children:
            if parameter.type == "self_parameter":
                names.append("self")
                types.append(re.sub(r"\s+", "", cls._text(source, parameter)))
                continue
            if parameter.type != "parameter":
                continue
            pattern = parameter.child_by_field_name("pattern")
            type_node = parameter.child_by_field_name("type")
            name = cls._text(source, pattern).strip()
            type_name = cls._text(source, type_node).strip()
            if name:
                names.append(name)
                types.append(re.sub(r"\s+", "", type_name))
        return names, types

    @classmethod
    def _augment_behavior(
        cls, source: bytes, body: Node | None, behavior: dict[str, list[str]]
    ) -> None:
        if body is None:
            return
        configs = set(behavior["config_references"])
        effects = set(behavior["side_effects"])
        exceptions = set(behavior["exceptions"])
        for call in cls._nodes(body, "call_expression", "macro_invocation"):
            name = cls._call_name(source, call)
            lowered = name.lower()
            arguments = call.child_by_field_name("arguments")
            if arguments is None and call.type == "macro_invocation":
                arguments = next(cls._nodes(call, "token_tree"), None)
            literal = (
                next(cls._nodes(arguments, "string_literal"), None)
                if arguments is not None
                else None
            )
            key = cls._text(source, literal).strip('"')
            if lowered in {"std::env::var", "env::var"} and key:
                configs.add(f"environment.{key}")
                effects.add("environment")
            if lowered in {"panic!", "todo!", "unimplemented!"}:
                exceptions.add(lowered.rstrip("!"))
        behavior["config_references"] = unique(configs)
        behavior["side_effects"] = unique(effects)
        behavior["exceptions"] = unique(exceptions)

    @classmethod
    def _item_annotations(cls, source: bytes, node: Node) -> list[str]:
        result = [
            cls._text(source, child).strip()
            for child in node.named_children
            if child.type in {"attribute_item", "visibility_modifier"}
        ]
        previous = node.prev_named_sibling
        while previous is not None and previous.type == "attribute_item":
            result.append(cls._text(source, previous).strip())
            previous = previous.prev_named_sibling
        return unique(result)

    @staticmethod
    def _simple_type_name(value: str) -> str:
        value = re.sub(r"<.*>", "", value).strip()
        return value.rsplit("::", 1)[-1]

    @classmethod
    def _type_identity(cls, source: bytes, node: Node | None) -> str:
        return re.sub(r"\s+", "", cls._text(source, node).strip())

    @staticmethod
    def _is_public_item(source: bytes, item: ParsedClass) -> bool:
        del source
        return item.source_code.lstrip().startswith("pub ")


__all__ = ["RustParser"]
