"""Deterministic, lightweight Python source analysis based on ``ast``."""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Sequence
from functools import cached_property
from pathlib import Path, PurePosixPath

from .base import (
    ParsedCall,
    ParsedClass,
    ParsedCodeSegment,
    ParsedFile,
    ParsedFunction,
    normalized_source_hash,
)


def _dotted_name(node: ast.AST | None) -> str:
    if node is None:
        return ""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Call):
        return _dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return _dotted_name(node.value)
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _sorted(values: Iterable[str]) -> list[str]:
    return sorted({value for value in values if value})


def _source_span(source: str, node: ast.AST) -> str:
    """Return the exact repository lines represented by an AST source span."""
    start = int(getattr(node, "lineno", 1))
    end = int(getattr(node, "end_lineno", start))
    return "\n".join(source.splitlines()[start - 1 : end])


class _SourceText(str):
    """File-local byte lines; AST columns are UTF-8 offsets, not characters."""

    @cached_property
    def ast_lines(self) -> list[bytes]:
        # Unlike str.splitlines(), Python source line boundaries do not include
        # form feed/vertical tab characters inside strings. Keep CRLF intact.
        return [line.encode("utf-8") for line in re.split(r"(?<=\n)|(?<=\r)(?!\n)", self)]

    def segment(self, node: ast.AST) -> str | None:
        attributes = [getattr(node, name, None) for name in ("lineno", "end_lineno", "col_offset", "end_col_offset")]
        if any(value is None for value in attributes):
            return None
        start, end, column, end_column = attributes
        lines = self.ast_lines
        if start == end:
            return lines[start - 1][column:end_column].decode("utf-8")
        return (lines[start - 1][column:] + b"".join(lines[start:end - 1]) + lines[end - 1][:end_column]).decode("utf-8")


class _FactVisitor(ast.NodeVisitor):
    """Collect local facts without descending into nested symbol definitions."""

    def __init__(self, source: str = "") -> None:
        self.source = source if isinstance(source, _SourceText) else _SourceText(source)
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.calls: set[str] = set()
        self.call_sites: list[ParsedCall] = []
        self.side_effects: set[str] = set()
        self.exceptions: set[str] = set()
        self.config_references: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.reads.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.writes.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = _dotted_name(node)
        if isinstance(node.ctx, ast.Load):
            self.reads.add(name)
        elif isinstance(node.ctx, (ast.Store, ast.Del)):
            self.writes.add(name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.side_effects.add("global_state")
        self.writes.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.side_effects.add("global_state")
        self.writes.update(node.names)

    def visit_Raise(self, node: ast.Raise) -> None:
        exception_name = _dotted_name(node.exc)
        if exception_name:
            self.exceptions.add(exception_name)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        base = _dotted_name(node.value)
        key = ""
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            key = node.slice.value
        if base == "os.environ":
            self.side_effects.add("environment")
            if key:
                self.config_references.add(f"environment.{key}")
        elif key and (
            base.lower().endswith(("config", "settings"))
            or ".config" in base.lower()
            or ".settings" in base.lower()
        ):
            self.config_references.add(f"{base}.{key}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        call = _dotted_name(node.func)
        if call:
            self.calls.add(call)
            expression = self.source.segment(node) if self.source else ""
            self.call_sites.append(
                ParsedCall(
                    target=call,
                    source_order=-1,
                    start_line=int(getattr(node, "lineno", 0)),
                    end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 0))),
                    start_column=getattr(node, "col_offset", None),
                    end_column=getattr(node, "end_col_offset", None),
                    expression=expression or call,
                )
            )
            lowered = call.lower()
            if call == "open" or any(
                token in lowered
                for token in (
                    ".read_text",
                    ".write_text",
                    ".read_bytes",
                    ".write_bytes",
                )
            ):
                self.side_effects.add("file_io")
            call_parts = lowered.split(".")
            looks_like_http_client = (
                len(call_parts) >= 2
                and call_parts[-1]
                in {"get", "post", "put", "patch", "delete", "request"}
                and call_parts[-2].endswith(("client", "session"))
            )
            if (
                lowered.startswith(("requests.", "httpx.", "urllib.", "socket."))
                or looks_like_http_client
            ):
                self.side_effects.add("network")
            if any(
                token in lowered
                for token in (
                    ".execute",
                    ".executemany",
                    ".commit",
                    "sqlalchemy",
                    ".cursor",
                )
            ):
                self.side_effects.add("database")
            if lowered.startswith("subprocess.") or call in {
                "os.system",
                "os.spawn",
                "os.execv",
            }:
                self.side_effects.add("process")
            if call in {"os.getenv", "os.putenv", "os.unsetenv"}:
                self.side_effects.add("environment")
                if node.args and isinstance(node.args[0], ast.Constant):
                    self.config_references.add(f"environment.{node.args[0].value}")
            if (
                lowered.startswith("logging.")
                or lowered.startswith("logger.")
                or ".logger." in lowered
            ):
                self.side_effects.add("logging")
        self.generic_visit(node)


class _MetricVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.branches = 0
        self.loops = 0
        self.returns = 0
        self.awaits = 0
        self.has_try = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        del node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        del node

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        del node

    def visit_If(self, node: ast.If) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.branches += 1
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self.branches += len(node.cases)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loops += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.loops += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.loops += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        self.returns += 1
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        self.awaits += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.has_try = True
        self.generic_visit(node)

    def as_dict(self) -> dict[str, object]:
        return {
            "branches": self.branches,
            "loops": self.loops,
            "returns": self.returns,
            "awaits": self.awaits,
            "has_try": self.has_try,
        }


class PythonParser:
    """Python implementation of the extensible language parser contract."""

    language = "python"

    def __init__(
        self,
        *,
        build_code_segments: bool = True,
        max_segments_per_function: int = 8,
        build_semantic_blocks: bool | None = None,
        max_blocks_per_function: int | None = None,
    ):
        if build_semantic_blocks is not None:
            build_code_segments = build_semantic_blocks
        if max_blocks_per_function is not None:
            max_segments_per_function = max_blocks_per_function
        self.build_code_segments = build_code_segments
        self.max_segments_per_function = max(0, max_segments_per_function)
        # Deprecated attributes for external parser integrations.
        self.build_semantic_blocks = self.build_code_segments
        self.max_blocks_per_function = self.max_segments_per_function

    def parse_file(self, path: Path, relative_path: str) -> ParsedFile:
        source = path.read_text(encoding="utf-8", errors="replace")
        return self.parse_source(source, relative_path)

    def parse_source(self, source: str, relative_path: str) -> ParsedFile:
        source = _SourceText(source)
        relative_path = PurePosixPath(relative_path).as_posix()
        module_name = self._module_name(relative_path)
        is_test = self._is_test_path(relative_path)
        file_hash = normalized_source_hash(source)
        try:
            tree = ast.parse(source, filename=relative_path, type_comments=True)
        except SyntaxError as exc:
            return ParsedFile(
                path=relative_path,
                language=self.language,
                module_name=module_name,
                content_hash=file_hash,
                source_loc=len(source.splitlines()),
                is_test=is_test,
                parse_error=f"{exc.msg} at line {exc.lineno}",
            )

        imports, aliases = self._imports(tree)
        functions: list[ParsedFunction] = []
        classes: list[ParsedClass] = []
        exports: list[str] = []

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = f"{module_name}.{node.name}" if module_name else node.name
                parsed_function = self._parse_function(
                    node, source, relative_path, module_name, qualified, "", is_test
                )
                functions.append(parsed_function)
                functions.extend(
                    self._nested_functions(
                        node,
                        source,
                        relative_path,
                        module_name,
                        qualified,
                        is_test,
                    )
                )
                exports.append(node.name)
            elif isinstance(node, ast.ClassDef):
                class_qualified = (
                    f"{module_name}.{node.name}" if module_name else node.name
                )
                parsed_class, methods = self._parse_class(
                    node, source, relative_path, module_name, class_qualified, is_test
                )
                classes.append(parsed_class)
                functions.extend(methods)
                exports.append(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets: Sequence[ast.AST]
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        value = node.value
                        if isinstance(value, (ast.List, ast.Tuple)):
                            exports.extend(
                                item.value
                                for item in value.elts
                                if isinstance(item, ast.Constant)
                                and isinstance(item.value, str)
                            )

        return ParsedFile(
            path=relative_path,
            language=self.language,
            module_name=module_name,
            content_hash=file_hash,
            source_loc=len(source.splitlines()),
            is_test=is_test,
            imports=imports,
            import_aliases=aliases,
            exports=_sorted(exports),
            functions=functions,
            classes=classes,
        )

    def _parse_class(
        self,
        node: ast.ClassDef,
        source: str,
        relative_path: str,
        module_name: str,
        qualified_name: str,
        is_test: bool,
    ) -> tuple[ParsedClass, list[ParsedFunction]]:
        class_source = _source_span(source, node)
        methods: list[ParsedFunction] = []
        class_is_test = is_test or node.name.startswith("Test")
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_qualified = f"{qualified_name}.{child.name}"
                method = self._parse_function(
                    child,
                    source,
                    relative_path,
                    module_name,
                    method_qualified,
                    qualified_name,
                    class_is_test,
                )
                methods.append(method)
                methods.extend(
                    self._nested_functions(
                        child,
                        source,
                        relative_path,
                        module_name,
                        method_qualified,
                        class_is_test,
                    )
                )
        parsed = ParsedClass(
            name=node.name,
            qualified_name=qualified_name,
            file_path=relative_path,
            source_code=class_source,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            content_hash=normalized_source_hash(class_source),
            base_classes=_sorted(_dotted_name(base) for base in node.bases),
            methods=[
                method.symbol_identity or method.qualified_name
                for method in methods
                if not method.parent_function_qualified_name
            ],
            docstring=ast.get_docstring(node, clean=False) or "",
            kind="class",
            is_test=class_is_test,
            annotations=_sorted(_dotted_name(item) for item in node.decorator_list),
        )
        return parsed, methods

    def _nested_functions(
        self,
        parent: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        relative_path: str,
        module_name: str,
        parent_qualified_name: str,
        is_test: bool,
    ) -> list[ParsedFunction]:
        """Return lexical nested functions without attributing their facts to parents."""
        result: list[ParsedFunction] = []
        for statement in parent.body:
            candidates: list[ast.AST] = [statement]
            while candidates:
                candidate = candidates.pop()
                if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested_qualified = (
                        f"{parent_qualified_name}.<locals>.{candidate.name}"
                    )
                    nested = self._parse_function(
                        candidate,
                        source,
                        relative_path,
                        module_name,
                        nested_qualified,
                        "",
                        is_test,
                        parent_function_qualified_name=parent_qualified_name,
                    )
                    result.append(nested)
                    result.extend(
                        self._nested_functions(
                            candidate,
                            source,
                            relative_path,
                            module_name,
                            nested_qualified,
                            is_test,
                        )
                    )
                    continue
                if isinstance(candidate, (ast.ClassDef, ast.Lambda)):
                    continue
                candidates.extend(reversed(list(ast.iter_child_nodes(candidate))))
        return result

    def _parse_function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        relative_path: str,
        module_name: str,
        qualified_name: str,
        class_qualified_name: str,
        is_test_file: bool,
        parent_function_qualified_name: str = "",
    ) -> ParsedFunction:
        function_source = _source_span(source, node)
        visitor = _FactVisitor(source)
        metrics = _MetricVisitor()
        for statement in node.body:
            visitor.visit(statement)
            metrics.visit(statement)

        parameters = self._parameters(node.args)
        return_type = ast.unparse(node.returns) if node.returns is not None else ""
        prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        signature = f"{prefix}{node.name}({ast.unparse(node.args)})"
        if return_type:
            signature += f" -> {return_type}"
        docstring = ast.get_docstring(node, clean=False) or ""
        preconditions = [
            ast.unparse(statement.test)
            for statement in node.body
            if isinstance(statement, ast.Assert)
        ]
        segments = self._segments(node, source, qualified_name)
        call_sites = sorted(
            visitor.call_sites,
            key=lambda call: (
                call.start_line,
                call.start_column if call.start_column is not None else -1,
                call.end_line,
                call.end_column if call.end_column is not None else -1,
            ),
        )
        for source_order, call_site in enumerate(call_sites):
            call_site.source_order = source_order
            containing_segments = [
                segment
                for segment in segments
                if segment.start_line <= call_site.start_line <= segment.end_line
            ]
            if containing_segments:
                call_site.enclosing_segment_id = min(
                    containing_segments,
                    key=lambda segment: (
                        segment.end_line - segment.start_line,
                        segment.end_column or 0,
                    ),
                ).segment_id
        decorators = _sorted(
            _dotted_name(decorator) for decorator in node.decorator_list
        )
        ast_summary = metrics.as_dict()
        ast_summary["decorators"] = decorators
        is_overload = any(
            decorator == "overload" or decorator.endswith(".overload")
            for decorator in decorators
        )
        symbol_identity = qualified_name
        if is_overload:
            parameter_types = self._parameter_types(node.args)
            overload_key = (
                ",".join(parameter_types)
                or normalized_source_hash(function_source)[:12]
            )
            return_key = return_type.replace(" ", "") or "Any"
            symbol_identity = (
                f"{qualified_name}({overload_key})->{return_key}@overload:"
                f"{normalized_source_hash(function_source)[:12]}"
            )
        return ParsedFunction(
            name=node.name,
            qualified_name=qualified_name,
            signature=signature,
            file_path=relative_path,
            module_name=module_name,
            source_code=function_source,
            start_line=node.lineno,
            end_line=getattr(node, "end_lineno", node.lineno),
            content_hash=normalized_source_hash(function_source),
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            reads=_sorted(visitor.reads),
            writes=_sorted(visitor.writes),
            calls=_sorted(visitor.calls),
            call_sites=call_sites,
            side_effects=_sorted(visitor.side_effects),
            exceptions=_sorted(visitor.exceptions),
            config_references=_sorted(visitor.config_references),
            preconditions=preconditions,
            ast_summary=ast_summary,
            class_qualified_name=class_qualified_name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_test=is_test_file or node.name.startswith("test_"),
            segments=segments,
            symbol_identity=symbol_identity,
            is_declaration=is_overload,
            parent_function_qualified_name=parent_function_qualified_name,
            language=self.language,
        )

    def _segments(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        qualified_name: str,
    ) -> list[ParsedCodeSegment]:
        if not self.build_code_segments or not self.max_segments_per_function:
            return []
        candidates: list[tuple[str, ast.AST, list[ast.stmt]]] = []

        def add(segment_type: str, node: ast.AST, body: list[ast.stmt] | None = None):
            candidates.append((segment_type, node, list(body or [])))

        def visit(statements: list[ast.stmt], *, elif_context: bool = False) -> None:
            for statement in statements:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(statement, ast.If):
                    add("elif" if elif_context else "if", statement)
                    visit(statement.body)
                    if len(statement.orelse) == 1 and isinstance(statement.orelse[0], ast.If):
                        visit(statement.orelse, elif_context=True)
                    elif statement.orelse:
                        add("else", statement.orelse[0], statement.orelse)
                        visit(statement.orelse)
                elif isinstance(statement, (ast.For, ast.AsyncFor)):
                    add("for", statement)
                    visit(statement.body)
                    if statement.orelse:
                        add("else", statement.orelse[0], statement.orelse)
                        visit(statement.orelse)
                elif isinstance(statement, ast.While):
                    add("while", statement)
                    visit(statement.body)
                    if statement.orelse:
                        add("else", statement.orelse[0], statement.orelse)
                        visit(statement.orelse)
                elif isinstance(statement, (ast.Try, getattr(ast, "TryStar", ast.Try))):
                    add("try", statement)
                    visit(statement.body)
                    for handler in statement.handlers:
                        add("except", handler, handler.body)
                        visit(handler.body)
                    if statement.orelse:
                        add("else", statement.orelse[0], statement.orelse)
                        visit(statement.orelse)
                    if statement.finalbody:
                        add("finally", statement.finalbody[0], statement.finalbody)
                        visit(statement.finalbody)
                elif isinstance(statement, (ast.With, ast.AsyncWith)):
                    add("with", statement)
                    visit(statement.body)
                elif isinstance(statement, ast.Match):
                    add("match", statement)
                    for case in statement.cases:
                        anchor = case.pattern
                        add("case", anchor, case.body)
                        visit(case.body)

        visit(list(function.body))
        function_lines = max(
            1, getattr(function, "end_lineno", function.lineno) - function.lineno + 1
        )
        if function_lines >= 8:
            for statement in function.body:
                if any(candidate[1] is statement for candidate in candidates):
                    continue
                fact_visitor = _FactVisitor()
                fact_visitor.visit(statement)
                if bool(
                    fact_visitor.side_effects
                    or any("." in write for write in fact_visitor.writes)
                ):
                    add("statement_group", statement)

        segments: list[ParsedCodeSegment] = []
        for source_order, (segment_type, node, body) in enumerate(
            sorted(
                candidates,
                key=lambda item: (
                    int(getattr(item[1], "lineno", 0)),
                    int(getattr(item[1], "col_offset", 0)),
                    int(getattr(item[1], "end_lineno", getattr(item[1], "lineno", 0))),
                    item[0],
                ),
            )[: self.max_segments_per_function]
        ):
            start_line = int(getattr(node, "lineno", function.lineno))
            end_anchor = body[-1] if body else node
            end_line = int(getattr(end_anchor, "end_lineno", start_line))
            start_column = getattr(node, "col_offset", None)
            end_column = getattr(end_anchor, "end_col_offset", None)
            snippet = "\n".join(source.splitlines()[start_line - 1 : end_line])
            facts = _FactVisitor()
            if body:
                for statement in body:
                    facts.visit(statement)
            else:
                facts.visit(node)
            control_transfer = ""
            if segment_type in {"if", "elif", "else", "match", "case"}:
                control_transfer = "branch"
            elif segment_type in {"for", "while"}:
                control_transfer = "loop"
            elif segment_type in {"try", "except", "finally"}:
                control_transfer = "exception"
            segment_id = (
                f"{segment_type}:{start_line}:{start_column or 0}-"
                f"{end_line}:{end_column or 0}"
            )
            segments.append(
                ParsedCodeSegment(
                    segment_id=segment_id,
                    code_snippet=snippet,
                    segment_type=segment_type,
                    start_line=start_line,
                    end_line=end_line,
                    start_column=start_column,
                    end_column=end_column,
                    source_order=source_order,
                    content_hash=normalized_source_hash(snippet),
                    reads=_sorted(facts.reads),
                    writes=_sorted(facts.writes),
                    control_transfer=control_transfer,
                    semantic_summary=f"{segment_type} segment in {qualified_name} at lines {start_line}-{end_line}.",
                )
            )
        return segments

    def _blocks(
        self,
        function: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        qualified_name: str,
    ) -> list[ParsedCodeSegment]:
        """Deprecated parser hook."""
        return self._segments(function, source, qualified_name)

    @staticmethod
    def _parameters(arguments: ast.arguments) -> list[str]:
        result = [argument.arg for argument in arguments.posonlyargs + arguments.args]
        if arguments.vararg:
            result.append(f"*{arguments.vararg.arg}")
        result.extend(argument.arg for argument in arguments.kwonlyargs)
        if arguments.kwarg:
            result.append(f"**{arguments.kwarg.arg}")
        return result

    @staticmethod
    def _parameter_types(arguments: ast.arguments) -> list[str]:
        result: list[str] = []
        for argument in arguments.posonlyargs + arguments.args:
            result.append(
                ast.unparse(argument.annotation) if argument.annotation else "Any"
            )
        if arguments.vararg:
            annotation = (
                ast.unparse(arguments.vararg.annotation)
                if arguments.vararg.annotation
                else "Any"
            )
            result.append(f"*{annotation}")
        for argument in arguments.kwonlyargs:
            result.append(
                ast.unparse(argument.annotation) if argument.annotation else "Any"
            )
        if arguments.kwarg:
            annotation = (
                ast.unparse(arguments.kwarg.annotation)
                if arguments.kwarg.annotation
                else "Any"
            )
            result.append(f"**{annotation}")
        return [value.replace(" ", "") for value in result]

    @staticmethod
    def _imports(tree: ast.Module) -> tuple[list[str], dict[str, str]]:
        imports: set[str] = set()
        aliases: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                for item in node.names:
                    imports.add(item.name)
                    aliases[item.asname or item.name.split(".")[0]] = item.name
            elif isinstance(node, ast.ImportFrom):
                prefix = "." * node.level + (node.module or "")
                imports.add(prefix)
                for item in node.names:
                    target = f"{prefix}.{item.name}" if prefix else item.name
                    aliases[item.asname or item.name] = target.strip(".")
        return _sorted(imports), aliases

    @staticmethod
    def _module_name(relative_path: str) -> str:
        path = PurePosixPath(relative_path)
        parts = list(path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _is_test_path(relative_path: str) -> bool:
        path = PurePosixPath(relative_path)
        return (
            any(part.lower() in {"test", "tests"} for part in path.parts)
            or path.name.startswith("test_")
            or path.stem.endswith(("_test", "Test"))
        )


__all__ = ["PythonParser", "normalized_source_hash"]
