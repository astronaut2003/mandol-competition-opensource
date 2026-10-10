"""Convert parsed repository facts into MemoryUnits and typed graph edges."""

from __future__ import annotations

import configparser
import json
import posixpath
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from mandol_substrate.representation import MemoryUnit
from ..relations import SoftwareRelation
from ..schemas import (
    ClassSchema,
    CodeSegmentSchema,
    ConfigSchema,
    DirectorySchema,
    FileSchema,
    FunctionSchema,
    RepositorySchema,
)
from ..spaces import software_space_for_type
from .parser import ParsedClass, ParsedFile, ParsedFunction
from .scanner import ManifestEntry, RepositoryManifest


@dataclass(frozen=True)
class SoftwareEdge:
    source_uid: str
    target_uid: str
    relation: SoftwareRelation
    properties: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


@dataclass
class BaseBuildFacts:
    units: list[MemoryUnit] = field(default_factory=list)
    spaces: list[list[str]] = field(default_factory=list)
    edges: list[SoftwareEdge] = field(default_factory=list)
    unit_uids_by_file: dict[str, list[str]] = field(default_factory=dict)
    symbol_hashes_by_file: dict[str, dict[str, str]] = field(default_factory=dict)
    module_to_uids: dict[str, list[str]] = field(default_factory=dict)
    config_uids: list[str] = field(default_factory=list)


class RelationBuilder:
    """Build deterministic base units and repository-local relationships."""

    def __init__(self, manifest: RepositoryManifest, repo_uid: str) -> None:
        self.manifest = manifest
        self.repo_uid = repo_uid
        self.root = Path(manifest.root)

    def build(
        self,
        parsed_files: dict[str, ParsedFile],
        *,
        repository_schema: RepositorySchema,
        unit_paths: set[str] | None = None,
        relation_paths: set[str] | None = None,
    ) -> BaseBuildFacts:
        """Build graph facts, optionally restricting materialized file facts.

        ``unit_paths`` limits File/Class/Function/CodeSegment construction while
        retaining repository-wide symbol indexes for conservative resolution.
        ``relation_paths`` limits source-oriented relations.  This lets an
        incremental update rebuild a changed file and its known dependants
        without recreating tens of thousands of unchanged MemoryUnits.
        """
        facts = BaseBuildFacts()
        edge_keys: set[tuple[Any, ...]] = set()
        seen_uids: set[str] = set()
        self._disambiguate_class_identities(parsed_files)
        self._disambiguate_function_identities(parsed_files)

        def add_unit(
            unit: MemoryUnit, path: str | None = None, module: str = ""
        ) -> None:
            unit.metadata.update(
                {
                    "repo_uid": self.repo_uid,
                    "repo_path": self.manifest.root,
                }
            )
            if unit.uid in seen_uids:
                raise ValueError(f"Duplicate software UID generated: {unit.uid}")
            seen_uids.add(unit.uid)
            facts.units.append(unit)
            software_type = unit.metadata["software_type"]
            facts.spaces.append([software_space_for_type(software_type).value])
            if path:
                facts.unit_uids_by_file.setdefault(path, []).append(unit.uid)
            if module:
                facts.module_to_uids.setdefault(module, []).append(unit.uid)

        def add_edge(
            source: str, target: str, relation: SoftwareRelation, **properties: Any
        ) -> None:
            call_site_identity = (
                properties.get("call_site_id")
                if relation is SoftwareRelation.CALLS
                else None
            )
            key = (source, target, relation, call_site_identity)
            if (source == target and relation is not SoftwareRelation.CALLS) or key in edge_keys:
                return
            edge_keys.add(key)
            facts.edges.append(SoftwareEdge(source, target, relation, properties))

        repository_unit = repository_schema.to_memory_unit()
        add_unit(repository_unit)

        for directory in self.manifest.directories:
            unit = DirectorySchema(
                path=directory,
                metadata={"repo_uid": self.repo_uid},
            ).to_memory_unit()
            add_unit(unit)
            parent = PurePosixPath(directory).parent.as_posix()
            parent_uid = self.repo_uid if parent == "." else f"directory:{parent}"
            add_edge(parent_uid, unit.uid, SoftwareRelation.CONTAINS)

        module_to_file_uid: dict[str, str] = {}
        path_to_file_uid: dict[str, str] = {}
        basename_to_file_uids: dict[str, list[str]] = {}
        stem_to_file_uids: dict[str, list[str]] = {}
        function_by_identity: dict[tuple[str, str], str] = {}
        function_by_qualified: dict[str, list[str]] = {}
        function_by_name: dict[str, list[str]] = {}
        class_by_qualified: dict[tuple[str, str], str] = {}
        class_candidates: dict[tuple[str, str], list[tuple[str, int, int, str]]] = {}
        functions: dict[str, ParsedFunction] = {}
        config_values: dict[str, tuple[Any, str]] = {}

        # Build lightweight repository-wide resolution indexes once.  The
        # expensive schema/MemoryUnit conversion below can then stay local to
        # changed files without reducing call or inheritance resolution quality.
        for path, parsed in parsed_files.items():
            for class_fact in parsed.classes:
                class_uid = self._class_uid(
                    parsed.language,
                    class_fact.symbol_identity or class_fact.qualified_name,
                )
                class_key = (parsed.language, class_fact.qualified_name)
                previous_class = class_by_qualified.get(class_key)
                if previous_class is None:
                    class_by_qualified[class_key] = class_uid
                elif previous_class != class_uid:
                    class_by_qualified[class_key] = ""
                class_candidates.setdefault(class_key, []).append(
                    (path, class_fact.start_line, class_fact.end_line, class_uid)
                )
            for function in parsed.functions:
                function_uid = self._function_uid(function)
                function_by_identity[
                    (
                        function.language,
                        function.symbol_identity or function.qualified_name,
                    )
                ] = function_uid
                function_by_qualified.setdefault(function.qualified_name, []).append(
                    function_uid
                )
                function_by_name.setdefault(function.name, []).append(function_uid)
                functions[function_uid] = function

        for entry in self.manifest.files:
            parsed = parsed_files.get(entry.path)
            module_name = (
                parsed.module_name if parsed else self._module_name(entry.path)
            )
            imports = parsed.imports if parsed else []
            exports = parsed.exports if parsed else []
            summary = (
                f"{entry.language.title()} {'test ' if entry.is_test else ''}module {module_name}."
                if entry.kind == "source"
                else f"Configuration file {entry.path}."
            )
            file_uid = f"file:{entry.path}"
            path_to_file_uid[entry.path] = file_uid
            basename_to_file_uids.setdefault(PurePosixPath(entry.path).name, []).append(
                file_uid
            )
            stem_to_file_uids.setdefault(
                PurePosixPath(entry.path).with_suffix("").as_posix(), []
            ).append(file_uid)
            module_to_file_uid.setdefault(module_name, file_uid)
            module_to_file_uid.setdefault(
                ".".join(PurePosixPath(entry.path).with_suffix("").parts),
                f"file:{entry.path}",
            )
            if parsed and parsed.language == "java":
                java_key = ".".join(
                    part
                    for part in (parsed.module_name, PurePosixPath(entry.path).stem)
                    if part
                )
                module_to_file_uid[java_key] = f"file:{entry.path}"

            parsed_config = (
                self._parse_config(entry) if entry.kind == "config" else {}
            )
            for key, value in parsed_config.items():
                config_values.setdefault(key, (value, entry.path))

            if unit_paths is not None and entry.path not in unit_paths:
                continue

            file_unit = FileSchema(
                path=entry.path,
                language=entry.language,
                imports=imports,
                exports=exports,
                semantic_summary=summary,
                content_hash=entry.content_hash,
                source_loc=entry.source_loc,
                is_test=entry.is_test,
                is_config=entry.kind == "config",
                module_name=module_name,
                parse_error=parsed.parse_error if parsed else "",
                language_facts=parsed.language_facts if parsed else {},
                metadata={
                    "repo_uid": self.repo_uid,
                    "source_kind": entry.kind,
                    "is_test": entry.is_test,
                    "module_name": module_name,
                },
            ).to_memory_unit()
            add_unit(file_unit, entry.path, module_name)
            parent = PurePosixPath(entry.path).parent.as_posix()
            parent_uid = self.repo_uid if parent == "." else f"directory:{parent}"
            add_edge(parent_uid, file_unit.uid, SoftwareRelation.CONTAINS)

            if entry.kind == "config":
                for key, value in parsed_config.items():
                    if config_values.get(key) != (value, entry.path):
                        continue
                    config_unit = ConfigSchema(
                        key=key,
                        value=value,
                        source_file=entry.path,
                        scope=key.split(".", 1)[0] if "." in key else module_name,
                        semantic_summary=f"Configuration {key} from {entry.path}.",
                        metadata={
                            "repo_uid": self.repo_uid,
                            "module_name": module_name,
                        },
                    ).to_memory_unit()
                    add_unit(config_unit, entry.path, module_name)
                    facts.config_uids.append(config_unit.uid)
                    add_edge(file_unit.uid, config_unit.uid, SoftwareRelation.CONTAINS)

        for path, parsed in parsed_files.items():
            if unit_paths is not None and path not in unit_paths:
                continue
            file_uid = f"file:{path}"
            for class_fact in parsed.classes:
                class_unit = ClassSchema(
                    name=class_fact.name,
                    qualified_name=class_fact.qualified_name,
                    base_classes=class_fact.base_classes,
                    methods=class_fact.methods,
                    semantic_summary=(
                        class_fact.docstring.strip().splitlines()[0]
                        if class_fact.docstring.strip()
                        else f"Class {class_fact.qualified_name}."
                    ),
                    file_path=path,
                    start_line=class_fact.start_line,
                    end_line=class_fact.end_line,
                    content_hash=class_fact.content_hash,
                    language=parsed.language,
                    kind=class_fact.kind,
                    interfaces=class_fact.interfaces,
                    docstring=class_fact.docstring,
                    symbol_identity=class_fact.symbol_identity
                    or class_fact.qualified_name,
                    metadata={
                        "repo_uid": self.repo_uid,
                        "file_path": path,
                        "module_name": parsed.module_name,
                        "class_kind": class_fact.kind,
                        "is_test": class_fact.is_test,
                        "annotations": class_fact.annotations,
                    },
                    uid=self._class_uid(
                        parsed.language,
                        class_fact.symbol_identity or class_fact.qualified_name,
                    ),
                ).to_memory_unit()
                add_unit(class_unit, path, parsed.module_name)
                facts.symbol_hashes_by_file.setdefault(path, {})[class_unit.uid] = (
                    class_fact.content_hash
                )
                add_edge(file_uid, class_unit.uid, SoftwareRelation.CONTAINS)
                add_edge(class_unit.uid, file_uid, SoftwareRelation.CONTAINED_IN)

            for function in parsed.functions:
                function_uid = self._function_uid(function)
                function_unit = FunctionSchema(
                    name=function.name,
                    qualified_name=function.qualified_name,
                    signature=function.signature,
                    language=parsed.language,
                    file_path=path,
                    docstring=function.docstring,
                    parameters=function.parameters,
                    return_type=function.return_type,
                    reads=function.reads,
                    writes=function.writes,
                    calls=function.calls,
                    call_sites=[asdict(call_site) for call_site in function.call_sites],
                    side_effects=function.side_effects,
                    preconditions=function.preconditions,
                    postconditions=function.postconditions,
                    semantic_summary=(
                        function.docstring.strip().splitlines()[0]
                        if function.docstring.strip()
                        else f"Function {function.qualified_name}."
                    ),
                    source_code=function.source_code,
                    start_line=function.start_line,
                    end_line=function.end_line,
                    content_hash=function.content_hash,
                    exceptions=function.exceptions,
                    ast_summary=function.ast_summary,
                    config_references=function.config_references,
                    symbol_identity=function.symbol_identity or function.qualified_name,
                    is_declaration=function.is_declaration,
                    uid=function_uid,
                    metadata={
                        "repo_uid": self.repo_uid,
                        "file_path": path,
                        "module_name": parsed.module_name,
                        "class_qualified_name": function.class_qualified_name,
                        "is_test": function.is_test,
                        "is_declaration": function.is_declaration,
                        "parent_function_qualified_name": function.parent_function_qualified_name,
                    },
                ).to_memory_unit()
                add_unit(function_unit, path, parsed.module_name)
                facts.symbol_hashes_by_file.setdefault(path, {})[function_unit.uid] = (
                    function.content_hash
                )
                if function.parent_function_qualified_name:
                    parent_candidates = function_by_qualified.get(
                        function.parent_function_qualified_name, []
                    )
                    if len(parent_candidates) == 1:
                        add_edge(
                            parent_candidates[0],
                            function_unit.uid,
                            SoftwareRelation.CONTAINS,
                        )
                        add_edge(
                            function_unit.uid,
                            parent_candidates[0],
                            SoftwareRelation.CONTAINED_IN,
                        )
                    else:
                        add_edge(file_uid, function_unit.uid, SoftwareRelation.CONTAINS)
                elif function.class_qualified_name:
                    candidates = class_candidates.get(
                        (parsed.language, function.class_qualified_name), []
                    )
                    containing = [
                        candidate
                        for candidate in candidates
                        if candidate[0] == path
                        and candidate[1] <= function.start_line <= candidate[2]
                    ]
                    if containing:
                        class_uid = min(
                            containing,
                            key=lambda candidate: candidate[2] - candidate[1],
                        )[3]
                    elif len(candidates) == 1:
                        class_uid = candidates[0][3]
                    else:
                        class_uid = ""
                    if class_uid:
                        add_edge(
                            class_uid, function_unit.uid, SoftwareRelation.HAS_METHOD
                        )
                        add_edge(
                            function_unit.uid, class_uid, SoftwareRelation.CONTAINED_IN
                        )
                    else:
                        add_edge(file_uid, function_unit.uid, SoftwareRelation.CONTAINS)
                        add_edge(
                            function_unit.uid, file_uid, SoftwareRelation.CONTAINED_IN
                        )
                else:
                    add_edge(file_uid, function_unit.uid, SoftwareRelation.CONTAINS)
                    add_edge(function_unit.uid, file_uid, SoftwareRelation.CONTAINED_IN)

                segment_units: list[MemoryUnit] = []
                for segment in sorted(
                    function.segments,
                    key=lambda item: (item.source_order, item.start_line),
                ):
                    segment_unit = CodeSegmentSchema(
                        segment_id=segment.segment_id,
                        segment_type=segment.segment_type,
                        language=parsed.language,
                        file_path=path,
                        parent_uid=function_unit.uid,
                        function_uid=function_unit.uid,
                        code_snippet=segment.code_snippet,
                        reads=segment.reads,
                        writes=segment.writes,
                        control_transfer=segment.control_transfer,
                        semantic_summary=segment.semantic_summary,
                        start_line=segment.start_line,
                        end_line=segment.end_line,
                        start_column=segment.start_column,
                        end_column=segment.end_column,
                        content_hash=segment.content_hash,
                        metadata={
                            "repo_uid": self.repo_uid,
                            "file_path": path,
                            "module_name": parsed.module_name,
                            "source_order": segment.source_order,
                        },
                    ).to_memory_unit()
                    add_unit(segment_unit, path, parsed.module_name)
                    segment_units.append(segment_unit)
                    add_edge(
                        segment_unit.uid,
                        function_unit.uid,
                        SoftwareRelation.CONTAINED_IN,
                    )
                for previous, current in zip(segment_units, segment_units[1:]):
                    add_edge(
                        previous.uid,
                        current.uid,
                        SoftwareRelation.NEXT_SEGMENT,
                    )

        for path, parsed in parsed_files.items():
            if relation_paths is not None and path not in relation_paths:
                continue
            source_file_uid = f"file:{path}"
            for imported in parsed.imports:
                target_file = self._resolve_import_file(
                    imported,
                    parsed,
                    module_to_file_uid,
                    path_to_file_uid,
                    basename_to_file_uids,
                    stem_to_file_uids,
                )
                if target_file:
                    add_edge(source_file_uid, target_file, SoftwareRelation.IMPORTS)

            for class_fact in parsed.classes:
                source_class = self._class_uid(
                    parsed.language,
                    class_fact.symbol_identity or class_fact.qualified_name,
                )
                for base in class_fact.base_classes:
                    target_class = self._resolve_class(base, parsed, class_by_qualified)
                    if target_class:
                        add_edge(
                            source_class, target_class, SoftwareRelation.INHERITS_FROM
                        )
                for interface in class_fact.interfaces:
                    target_interface = self._resolve_class(
                        interface, parsed, class_by_qualified
                    )
                    if target_interface:
                        add_edge(
                            source_class,
                            target_interface,
                            SoftwareRelation.IMPLEMENTS,
                        )

            for function in parsed.functions:
                source_function = self._function_uid(function)
                call_sites = function.call_sites or [
                    {
                        "target": call,
                        "source_order": source_order,
                        "start_line": 0,
                        "end_line": 0,
                        "start_column": None,
                        "end_column": None,
                        "expression": call,
                        "enclosing_segment_id": None,
                    }
                    for source_order, call in enumerate(function.calls)
                ]
                for call_site in call_sites:
                    if isinstance(call_site, dict):
                        call = str(call_site["target"])
                        properties = dict(call_site)
                    else:
                        call = call_site.target
                        properties = asdict(call_site)
                    properties.pop("target", None)
                    target_function = self._resolve_call(
                        call,
                        function,
                        parsed,
                        function_by_identity,
                        function_by_qualified,
                        function_by_name,
                        functions,
                    )
                    if not target_function:
                        continue
                    source_order = int(properties.get("source_order", 0))
                    properties["call_site_id"] = (
                        f"{source_function}@{properties.get('start_line', 0)}:"
                        f"{properties.get('start_column')}:{source_order}"
                    )
                    segment_id = properties.pop("enclosing_segment_id", None)
                    properties.pop("enclosing_block_id", None)
                    properties["enclosing_segment_uid"] = (
                        f"segment:{source_function}:{segment_id}"
                        if segment_id
                        else ""
                    )
                    add_edge(
                        source_function,
                        target_function,
                        SoftwareRelation.CALLS,
                        **properties,
                    )
                    if function.is_test and not functions[target_function].is_test:
                        add_edge(
                            target_function, source_function, SoftwareRelation.TESTED_BY
                        )
                for reference in function.config_references:
                    config_uid = self._resolve_config(reference, config_values)
                    if config_uid:
                        add_edge(
                            config_uid,
                            source_function,
                            SoftwareRelation.AFFECTS_BEHAVIOR_OF,
                        )

        return facts

    def _parse_config(self, entry: ManifestEntry) -> dict[str, Any]:
        path = Path(entry.absolute_path)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            suffix = path.suffix.lower()
            if suffix == ".json":
                return self._flatten(json.loads(text))
            if suffix == ".toml":
                return self._flatten(tomllib.loads(text))
            if suffix in {".ini", ".cfg"} or path.name in {"setup.cfg", "tox.ini"}:
                parser = configparser.ConfigParser()
                parser.read_string(text)
                return {
                    f"{section}.{key}": value
                    for section in parser.sections()
                    for key, value in parser.items(section)
                }
            if suffix == ".env" or path.name == ".env":
                return {
                    key.strip(): value.strip().strip("\"'")
                    for line in text.splitlines()
                    if line.strip()
                    and not line.lstrip().startswith("#")
                    and "=" in line
                    for key, value in [line.split("=", 1)]
                }
            if suffix in {".yaml", ".yml"}:
                return self._simple_yaml(text)
        except (OSError, ValueError, configparser.Error):
            return {}
        return {}

    @classmethod
    def _flatten(cls, value: Any, prefix: str = "") -> dict[str, Any]:
        result: dict[str, Any] = {}
        if isinstance(value, dict):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                result.update(cls._flatten(item, child))
        elif isinstance(value, list) and prefix or prefix:
            result[prefix] = value
        return result

    @staticmethod
    def _simple_yaml(text: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        stack: list[tuple[int, str]] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            indent = len(line) - len(line.lstrip())
            key, raw_value = stripped.split(":", 1)
            key = key.strip()
            if not key:
                continue
            while stack and stack[-1][0] >= indent:
                stack.pop()
            prefix = ".".join(item[1] for item in stack)
            full_key = f"{prefix}.{key}" if prefix else key
            raw_value = raw_value.strip()
            if not raw_value:
                stack.append((indent, key))
                continue
            value: Any = raw_value.strip("\"'")
            if value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
            result[full_key] = value
        return result

    @staticmethod
    def _module_name(path: str) -> str:
        pure_path = PurePosixPath(path)
        parts = list(pure_path.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    @staticmethod
    def _function_uid(function: ParsedFunction) -> str:
        identity = function.symbol_identity or function.qualified_name
        language_prefix = (
            "" if function.language == "python" else f"{function.language}:"
        )
        return f"function:{language_prefix}{identity}"

    @classmethod
    def _disambiguate_function_identities(
        cls,
        parsed_files: dict[str, ParsedFile],
    ) -> None:
        """Resolve deterministic parser ambiguities without content-based UIDs.

        C/C++ projects commonly keep mutually exclusive platform definitions in
        one un-preprocessed syntax tree.  Return type normally is not needed for
        overload identity, so it is added only when an otherwise duplicate group
        exists; file/span is the conservative final tie breaker.
        """
        groups: dict[str, list[ParsedFunction]] = {}
        for parsed in parsed_files.values():
            for function in parsed.functions:
                groups.setdefault(cls._function_uid(function), []).append(function)
        for functions in groups.values():
            if len(functions) < 2:
                continue
            return_candidates: dict[str, list[ParsedFunction]] = {}
            for function in functions:
                base = function.symbol_identity or function.qualified_name
                return_type = re.sub(r"\s+", "", function.return_type or "void")
                candidate = f"{base}->{return_type}"
                function.symbol_identity = candidate
                return_candidates.setdefault(candidate, []).append(function)
            for candidate, remaining in return_candidates.items():
                if len(remaining) < 2:
                    continue
                for ordinal, function in enumerate(
                    sorted(
                        remaining,
                        key=lambda item: (
                            item.file_path,
                            item.start_line,
                            item.end_line,
                        ),
                    ),
                    start=1,
                ):
                    path = PurePosixPath(function.file_path).as_posix()
                    function.symbol_identity = f"{candidate}@{path}:{ordinal}"

    @classmethod
    def _disambiguate_class_identities(
        cls,
        parsed_files: dict[str, ParsedFile],
    ) -> None:
        groups: dict[str, list[ParsedClass]] = {}
        for parsed in parsed_files.values():
            for class_fact in parsed.classes:
                identity = class_fact.symbol_identity or class_fact.qualified_name
                groups.setdefault(cls._class_uid(parsed.language, identity), []).append(
                    class_fact
                )
        for classes in groups.values():
            if len(classes) < 2:
                continue
            per_file: dict[str, list[ParsedClass]] = {}
            for class_fact in classes:
                per_file.setdefault(class_fact.file_path, []).append(class_fact)
            for path, variants in per_file.items():
                base = variants[0].symbol_identity or variants[0].qualified_name
                for ordinal, class_fact in enumerate(
                    sorted(variants, key=lambda item: (item.start_line, item.end_line)),
                    start=1,
                ):
                    suffix = f"@{PurePosixPath(path).as_posix()}"
                    if len(variants) > 1:
                        suffix += f":{ordinal}"
                    class_fact.symbol_identity = f"{base}{suffix}"

    @staticmethod
    def _class_uid(language: str, qualified_name: str) -> str:
        language_prefix = "" if language == "python" else f"{language}:"
        return f"class:{language_prefix}{qualified_name}"

    @staticmethod
    def _resolve_import_file(
        imported: str,
        parsed: ParsedFile,
        module_to_file_uid: dict[str, str],
        path_to_file_uid: dict[str, str],
        basename_to_file_uids: dict[str, list[str]],
        stem_to_file_uids: dict[str, list[str]],
    ) -> str | None:
        imported = imported.strip().strip('<>"')
        # ``from . import name`` / ``from .. import name`` has no explicit
        # module target. PythonParser preserves the relative prefix as a
        # conservative fact; there is no reliable file endpoint to resolve.
        if imported and not imported.strip("."):
            return None
        if imported in path_to_file_uid:
            return path_to_file_uid[imported]
        relative_include = (
            PurePosixPath(parsed.path).parent / PurePosixPath(imported)
        ).as_posix()
        if relative_include in path_to_file_uid:
            return path_to_file_uid[relative_include]
        if imported.startswith("."):
            relative_stem = PurePosixPath(
                posixpath.normpath(relative_include)
            ).with_suffix("").as_posix()
            local_matches = list(stem_to_file_uids.get(relative_stem, ()))
            local_matches.extend(
                stem_to_file_uids.get(posixpath.join(relative_stem, "index"), ())
            )
            if len(local_matches) == 1:
                return local_matches[0]
        basename_matches = basename_to_file_uids.get(PurePosixPath(imported).name, [])
        if len(basename_matches) == 1:
            return basename_matches[0]

        imported = imported.lstrip(".")
        candidates = [imported]
        if "." in parsed.module_name:
            candidates.append(f"{parsed.module_name.rsplit('.', 1)[0]}.{imported}")
        for candidate in candidates:
            while candidate:
                if candidate in module_to_file_uid:
                    return module_to_file_uid[candidate]
                candidate = candidate.rsplit(".", 1)[0] if "." in candidate else ""
        return None

    @staticmethod
    def _resolve_class(
        base: str,
        parsed: ParsedFile,
        classes: dict[tuple[str, str], str],
    ) -> str | None:
        normalized = base.replace("::", ".")
        candidates = [normalized, f"{parsed.module_name}.{normalized}"]
        alias = parsed.import_aliases.get(normalized.split(".", 1)[0])
        if alias:
            suffix = normalized.split(".", 1)[1] if "." in normalized else ""
            candidates.append(f"{alias}.{suffix}".rstrip("."))
        for candidate in candidates:
            uid = classes.get((parsed.language, candidate))
            if uid:
                return uid
        simple = normalized.rsplit(".", 1)[-1]
        matches = [
            uid
            for (language, qualified), uid in classes.items()
            if language == parsed.language and qualified.rsplit(".", 1)[-1] == simple
        ]
        if len(matches) == 1:
            return matches[0]
        return None

    @staticmethod
    def _resolve_call(
        call: str,
        function: ParsedFunction,
        parsed: ParsedFile,
        identities: dict[tuple[str, str], str],
        functions: dict[str, list[str]],
        functions_by_name: dict[str, list[str]],
        parsed_functions: dict[str, ParsedFunction],
    ) -> str | None:
        call = re.sub(r"<[^<>]*>", "", call).replace("::", ".")
        candidates = [call]
        if (
            call.startswith(("self.", "cls.", "this."))
            and function.class_qualified_name
        ):
            candidates.append(
                f"{function.class_qualified_name}.{call.split('.', 1)[1]}"
            )
        if function.class_qualified_name:
            candidates.append(f"{function.class_qualified_name}.{call}")
        candidates.append(f"{function.module_name}.{call}")
        head, separator, tail = call.partition(".")
        alias = parsed.import_aliases.get(head)
        if alias:
            candidates.append(f"{alias}.{tail}" if separator else alias)
        for candidate in candidates:
            identity_uid = identities.get((function.language, candidate))
            if identity_uid:
                return identity_uid
            resolved = RelationBuilder._prefer_definition(
                functions.get(candidate, []), parsed_functions, function.language
            )
            if resolved:
                return resolved
        simple = call.rsplit(".", 1)[-1]
        matches = functions_by_name.get(simple, [])
        return RelationBuilder._prefer_definition(
            matches, parsed_functions, function.language
        )

    @staticmethod
    def _prefer_definition(
        uids: list[str],
        functions: dict[str, ParsedFunction],
        language: str,
    ) -> str | None:
        uids = [uid for uid in uids if functions[uid].language == language]
        definitions = [uid for uid in uids if not functions[uid].is_declaration]
        if len(definitions) == 1:
            return definitions[0]
        return uids[0] if not definitions and len(uids) == 1 else None

    @staticmethod
    def _resolve_config(
        reference: str, config_values: dict[str, tuple[Any, str]]
    ) -> str | None:
        normalized = reference.lower().replace("_", ".")
        exact = [
            key for key in config_values if key.lower().replace("_", ".") == normalized
        ]
        if len(exact) == 1:
            return f"config:{exact[0]}"
        suffix = normalized.rsplit(".", 1)[-1]
        matches = [
            key
            for key in config_values
            if key.lower().replace("_", ".").rsplit(".", 1)[-1] == suffix
        ]
        return f"config:{matches[0]}" if len(matches) == 1 else None


__all__ = ["BaseBuildFacts", "RelationBuilder", "SoftwareEdge"]
