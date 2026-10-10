"""Small cold/warm/incremental repository construction benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..core.semantic_graph import SemanticGraph
from ..core.semantic_map import SemanticMap
from ..utils.model_manager import global_model_manager
from .builder import RepositoryScanner
from .facade import SemanticCode
from .validation import evaluate_repository_quality


class _BenchmarkEmbeddingModel:
    dimension = 128

    def _one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % self.dimension] += 1.0
        return vector

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._one(texts)
        return np.stack([self._one(text) for text in texts])


def _copy_manifest(source: Path, target: Path) -> list[str]:
    manifest = RepositoryScanner(source).scan()
    for entry in manifest.files:
        destination = target / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry.absolute_path, destination)
    return [entry.path for entry in manifest.files if entry.kind == "source"]


def benchmark_repository(repo_path: str) -> list[tuple[str, dict]]:
    """Measure cold, unchanged warm, and one-file incremental builds."""
    source = Path(repo_path).expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="semantic-code-benchmark-") as temporary:
        target = Path(temporary) / source.name
        target.mkdir()
        source_files = _copy_manifest(source, target)
        with patch.object(
            global_model_manager,
            "get_or_load_model",
            return_value=_BenchmarkEmbeddingModel(),
        ):
            semantic_map = SemanticMap(
                embedding_model_name="semantic-code/benchmark",
                embedding_dim=_BenchmarkEmbeddingModel.dimension,
                use_flash_attention=False,
            )
        enrichment_invocations: list[tuple[str, str]] = []

        def semantic_enricher(kind: str, payload: dict) -> dict[str, str]:
            path = str(payload.get("path", ""))
            enrichment_invocations.append((kind, path))
            return {
                "responsibility": f"Deterministic benchmark summary for {path}.",
                "semantic_summary": f"File-level semantic memory for {path}.",
            }

        semantic_code = SemanticCode.from_repository(
            str(target),
            semantic_graph=SemanticGraph(semantic_map),
            include_git_history=False,
            build_high_level_memory=True,
            build_indexes=True,
            semantic_enricher=semantic_enricher,
            semantic_enricher_version="benchmark-v1",
        )
        cold_uids = {unit.uid for unit in semantic_code.semantic_graph.get_all_units()}
        cold_edges = semantic_code.semantic_graph.rx_graph.num_edges()
        cold_report = semantic_code.last_build_report.to_dict()
        cold_report.update(
            {
                "uid_stable": True,
                "unit_delta": 0,
                "edge_delta": 0,
                "quality": evaluate_repository_quality(
                    semantic_code, target
                ).to_dict(),
            }
        )
        rows = [("cold", cold_report)]

        semantic_code.update_repository(include_git_history=False)
        warm_uids = {unit.uid for unit in semantic_code.semantic_graph.get_all_units()}
        warm_report = semantic_code.last_build_report.to_dict()
        warm_report.update(
            {
                "uid_stable": warm_uids == cold_uids,
                "unit_delta": len(warm_uids) - len(cold_uids),
                "edge_delta": semantic_code.semantic_graph.rx_graph.num_edges()
                - cold_edges,
                "quality": evaluate_repository_quality(
                    semantic_code, target
                ).to_dict(),
            }
        )
        rows.append(("warm", warm_report))

        if source_files:
            changed_relative = source_files[0]
            changed_file = target / changed_relative
            original = changed_file.read_text(encoding="utf-8", errors="replace")
            changed_file.write_text(
                original + _benchmark_change(changed_file.suffix.lower()),
                encoding="utf-8",
            )
            semantic_code.update_repository(include_git_history=False)
            incremental_uids = {
                unit.uid for unit in semantic_code.semantic_graph.get_all_units()
            }
            incremental_report = semantic_code.last_build_report.to_dict()
            incremental_report.update(
                {
                    "uid_stable": cold_uids.issubset(incremental_uids),
                    "unit_delta": len(incremental_uids) - len(cold_uids),
                    "edge_delta": semantic_code.semantic_graph.rx_graph.num_edges()
                    - cold_edges,
                    "changed_source_file": changed_relative,
                    "quality": evaluate_repository_quality(
                        semantic_code, target
                    ).to_dict(),
                }
            )
            rows.append(("single-file", incremental_report))
        return rows


def _benchmark_change(suffix: str) -> str:
    if suffix == ".py":
        return "\n\ndef semantic_code_benchmark_change():\n    return None\n"
    if suffix in {".c", ".h"}:
        return "\nint semantic_code_benchmark_change(void) { return 0; }\n"
    if suffix in {".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx", ".inl", ".ipp"}:
        return "\nint semantic_code_benchmark_change() { return 0; }\n"
    if suffix == ".java":
        return (
            "\nclass SemanticCodeBenchmarkChange {\n"
            "  static int value() { return 0; }\n}\n"
        )
    return "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "repo_paths", nargs="+", help="One to three local repository paths"
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optionally persist the measured scenario reports as JSON.",
    )
    args = parser.parse_args()
    collected: dict[str, list[dict]] = {}
    for repo_path in args.repo_paths[:3]:
        resolved = str(Path(repo_path).resolve())
        print(f"repository: {resolved}")
        print(
            "scenario files loc functions segments file_memories units edges high "
            "parsed cache_hits enrich_calls enrich_cache_hits uid_stable unit_delta "
            "edge_delta relation_mode relation_files scan_s parse_s relation_s "
            "edge_insert_s dense_s base_s file_memory_s index_s high_s total_s "
            "embedding_device embedding_batches peak_ram_mb"
        )
        rows = benchmark_repository(repo_path)
        collected[resolved] = [
            {"scenario": scenario, **report} for scenario, report in rows
        ]
        for scenario, report in rows:
            print(
                scenario,
                report["repository_files"],
                report["source_loc"],
                report["functions"],
                report["code_segments"],
                report["file_memories"],
                report["memory_units"],
                report["relationships"],
                report["high_level_memories"],
                report["parsed_files"],
                report["cache_hits"],
                report["semantic_enrichment_calls"],
                report["semantic_enrichment_cache_hits"],
                report["uid_stable"],
                report["unit_delta"],
                report["edge_delta"],
                report["relation_build_mode"],
                report["relation_files_processed"],
                f"{report['scan_time']:.6f}",
                f"{report['parse_time']:.6f}",
                f"{report['relation_fact_build_time']:.6f}",
                f"{report['relationship_insert_time']:.6f}",
                f"{report['dense_embedding_time']:.6f}",
                f"{report['base_memory_build_time']:.6f}",
                f"{report['file_memory_build_time']:.6f}",
                f"{report['embedding_index_time']:.6f}",
                f"{report['high_level_memory_build_time']:.6f}",
                f"{report['total_build_time']:.6f}",
                report["embedding_device"],
                report["embedding_batches"],
                report["peak_ram_mb"],
            )
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(collected, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
