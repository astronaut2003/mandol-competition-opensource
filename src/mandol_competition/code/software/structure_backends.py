"""Optional structural fact sources that preserve Mandol's graph/schema layer.

CodeGraph is an external, explicitly prepared SQLite export, never a package
installed or process launched by this module. No graph database dependency is
added to Coda. Experimental imports should use an isolated repository graph.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from mandol_substrate.representation import MemoryUnit
from .builder import CParser, CppParser, GoParser, JavaParser, JavaScriptParser, PythonParser, RustParser, TypeScriptParser, RepositoryScanner
from .builder.relation_builder import BaseBuildFacts, RelationBuilder, SoftwareEdge
from .relations import SoftwareRelation
from .schemas import ClassSchema, FileSchema, FunctionSchema, RepositorySchema
from .spaces import software_space_for_type


@dataclass
class StructureResult:
    facts: BaseBuildFacts
    backend: str
    diagnostics: dict = field(default_factory=dict)


class StructureBackend(Protocol):
    def extract(self, repository_path: Path) -> StructureResult: ...


class NativeCodaBackend:
    """Reuse the existing native parsers AND relation resolver, without models."""

    def __init__(self):
        classes = (PythonParser, CParser, CppParser, JavaParser, JavaScriptParser, TypeScriptParser, GoParser, RustParser)
        self.parsers = {parser.language: parser for parser in (cls(build_code_segments=False) for cls in classes)}
        self.cache = {}

    def extract(self, repository_path: Path):
        root = repository_path.resolve()
        manifest = RepositoryScanner(root).scan()
        # Generated external indexes are never repository evidence.
        manifest.files = [entry for entry in manifest.files if not entry.path.startswith(".codegraph/")]
        manifest.directories = [path for path in manifest.directories if path != ".codegraph" and not path.startswith(".codegraph/")]
        parsed, errors, hits = {}, {}, 0
        from tree_sitter_language_pack import downloaded_languages
        prepared = set(downloaded_languages())
        for entry in manifest.files:
            if entry.kind != "source" or entry.language not in self.parsers:
                continue
            grammar = None if entry.language == "python" else (self.parsers[entry.language].grammar_name or entry.language)
            if grammar is not None and grammar not in prepared:
                raise RuntimeError(f"Prepare the pinned {grammar} grammar cache before offline extraction")
            key = (str(root), entry.path, entry.content_hash)
            if key in self.cache:
                parsed[entry.path] = self.cache[key]
                hits += 1
            else:
                parsed[entry.path] = self.parsers[entry.language].parse_file(Path(entry.absolute_path), entry.path)
                self.cache[key] = parsed[entry.path]
            if parsed[entry.path].parse_error:
                errors[entry.path] = parsed[entry.path].parse_error
        repo = RepositorySchema(name=root.name, path=str(root))
        facts = RelationBuilder(manifest, repo.to_memory_unit().uid).build(parsed, repository_schema=repo)
        return StructureResult(facts, "native-coda", {"parsed_files": len(parsed), "parse_errors": errors, "parse_cache_hits": hits})


class CodeGraphBackend:
    """Normalize the pinned colbymchenry/codegraph v1.6 SQLite schema.

    Structural UIDs depend on repository/path/kind/qualified symbol/signature,
    not source line number. Raw source spans are rechecked against the checkout.
    Unsupported external node/edge kinds are counted, not guessed into ontology.
    """

    relation_map = {"contains": SoftwareRelation.CONTAINS, "calls": SoftwareRelation.CALLS,
                    "imports": SoftwareRelation.IMPORTS, "inherits": SoftwareRelation.INHERITS_FROM,
                    "extends": SoftwareRelation.INHERITS_FROM, "implements": SoftwareRelation.IMPLEMENTS}

    def __init__(self, database: Path, *, repository: str, revision: str, backend_revision: str):
        self.database = Path(database).resolve()
        self.repository, self.revision, self.backend_revision = repository, revision, backend_revision

    def extract(self, repository_path: Path):
        root = repository_path.resolve()
        if not self.database.is_file():
            raise FileNotFoundError("Prepare an explicit local CodeGraph export first")
        connection = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("BEGIN")
            rows = [dict(row) for row in connection.execute("SELECT * FROM nodes ORDER BY id")]
            edges = [dict(row) for row in connection.execute("SELECT * FROM edges ORDER BY id")]
            files = [dict(row) for row in connection.execute("SELECT * FROM files ORDER BY path")]
        finally:
            connection.close()
        repo_uid = "repo:codegraph:" + hashlib.sha256(self.repository.encode()).hexdigest()
        repo = RepositorySchema(name=self.repository, uid=repo_uid, path=str(root), revision=self.revision).to_memory_unit()
        facts = BaseBuildFacts(units=[repo], spaces=[[software_space_for_type("repository").value]])
        mapping, rejected, ignored = {}, [], {}
        file_hashes = {row["path"]: row["content_hash"] for row in files}
        # Conditional/overloaded declarations can share a public signature.
        # Preserve every variant; source order disambiguates only within that
        # group. Reordering variants is an explicitly documented UID limit.
        groups = defaultdict(list)
        for row in rows:
            groups[(row["file_path"], row["language"], row["kind"], row["qualified_name"], row.get("signature"))].append(row)
        ordinals = {row["id"]: index for variants in groups.values()
                    for index, row in enumerate(sorted(variants, key=lambda row: (row["start_line"], row["start_column"], row["id"])))}
        for row in rows:
            kind = row["kind"]
            if kind not in {"file", "function", "method", "class", "interface", "struct"}:
                ignored[kind] = ignored.get(kind, 0) + 1
                continue
            path = (root / row["file_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                rejected.append({"id": row["id"], "reason": "missing or outside repository source"})
                continue
            raw_bytes = path.read_bytes()
            if hashlib.sha256(raw_bytes).hexdigest() != file_hashes.get(row["file_path"]):
                rejected.append({"id": row["id"], "reason": "stale CodeGraph source hash; sync required"})
                continue
            text = raw_bytes.decode("utf-8", errors="strict")
            lines = text.splitlines(keepends=True)
            start, end = row["start_line"], row["end_line"]
            if kind != "file" and not 1 <= start <= end <= len(lines):
                rejected.append({"id": row["id"], "reason": "invalid source span"})
                continue
            relative = path.relative_to(root).as_posix()
            identity = json.dumps([self.repository, row["language"], relative, kind, row["qualified_name"], row.get("signature"), ordinals[row["id"]]], separators=(",", ":"))
            uid = "structural:" + hashlib.sha256(identity.encode()).hexdigest()
            metadata = {"repo_uid": repo_uid, "repository": self.repository, "revision": self.revision,
                        "structure_backend": "codegraph", "structure_backend_revision": self.backend_revision,
                        "external_id": row["id"], "file_path": relative, "source_file_sha256": hashlib.sha256(raw_bytes).hexdigest()}
            qualified = row["qualified_name"]
            if row["language"] == "python" and kind != "file":
                module = relative.removesuffix(".py").replace("/", ".").removesuffix(".__init__")
                qualified = f"{module}.{qualified.replace('::', '.')}"
                metadata["external_qualified_name"] = row["qualified_name"]
            if kind == "file":
                unit = FileSchema(path=relative, language=row["language"], source_loc=len(lines), uid=uid, metadata=metadata).to_memory_unit()
            elif kind in {"function", "method"}:
                unit = FunctionSchema(name=row["name"], qualified_name=qualified, signature=row.get("signature") or "",
                                      file_path=relative, language=row["language"], start_line=start, end_line=end,
                                      source_code="".join(lines[start - 1:end]), uid=uid, metadata=metadata).to_memory_unit()
            else:
                unit = ClassSchema(name=row["name"], qualified_name=qualified, file_path=relative, language=row["language"],
                                   start_line=start, end_line=end, kind=kind, uid=uid, metadata=metadata).to_memory_unit()
            if uid in mapping.values():
                rejected.append({"id": row["id"], "reason": "ambiguous normalized symbol identity"})
                continue
            mapping[row["id"]] = uid
            facts.units.append(unit)
            facts.spaces.append([software_space_for_type(unit.metadata["software_type"]).value])
            facts.unit_uids_by_file.setdefault(relative, []).append(uid)
            if kind == "file":
                facts.edges.append(SoftwareEdge(repo_uid, uid, SoftwareRelation.CONTAINS))
        skipped_edges = 0
        for edge in edges:
            relation = self.relation_map.get(edge["kind"])
            if relation is None or edge["source"] not in mapping or edge["target"] not in mapping:
                skipped_edges += 1
                continue
            facts.edges.append(SoftwareEdge(mapping[edge["source"]], mapping[edge["target"]], relation,
                                            {"structure_backend": "codegraph", "external_provenance": edge.get("provenance")}))
        return StructureResult(facts, "codegraph", {"external_nodes": len(rows), "external_edges": len(edges),
                                                    "files": files, "rejected_nodes": rejected, "unmapped_node_kinds": ignored,
                                                    "unmapped_edges": skipped_edges, "backend_revision": self.backend_revision,
                                                    "order_disambiguated_groups": sum(len(group) > 1 for group in groups.values())})
