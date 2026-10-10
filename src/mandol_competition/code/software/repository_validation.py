"""Independent-process repository persistence validation utility."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ..core.semantic_graph import SemanticGraph
from ..core.semantic_map import SemanticMap
from ..utils.model_manager import global_model_manager
from .facade import SemanticCode
from .validation import (
    RepositorySnapshotFingerprint,
    evaluate_repository_quality,
    query_snapshot,
    relationship_samples,
    spot_check_samples,
)


class _ValidationEmbeddingModel:
    """Offline deterministic model for reproducible validation only."""

    dimension = 96

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


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _build(repo_path: Path, output: Path, *, include_git_history: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "graph"
    with patch.object(
        global_model_manager,
        "get_or_load_model",
        return_value=_ValidationEmbeddingModel(),
    ):
        semantic_map = SemanticMap(
            embedding_model_name="semantic-code/validation",
            embedding_dim=_ValidationEmbeddingModel.dimension,
            use_flash_attention=False,
        )
    semantic_code = SemanticCode.from_repository(
        str(repo_path),
        semantic_graph=SemanticGraph(semantic_map),
        include_git_history=include_git_history,
        max_git_commits=50,
        build_high_level_memory=True,
        build_indexes=True,
        cache_path=output / "builder_cache.json",
    )
    result = {
        "fingerprint": RepositorySnapshotFingerprint.from_graph(
            semantic_code.semantic_graph
        ).to_dict(),
        "queries": query_snapshot(semantic_code),
        "quality": evaluate_repository_quality(semantic_code, repo_path).to_dict(),
        "relationship_samples": relationship_samples(semantic_code.semantic_graph),
        "spot_checks": spot_check_samples(semantic_code, repo_path),
        "build_report": semantic_code.last_build_report.to_dict(),
    }
    semantic_code.semantic_graph.save_graph(
        str(snapshot_path),
        retriever_methods_to_build=[],
        build_sparse_vectors=False,
    )
    result["artifacts"] = sorted(
        str(path.relative_to(snapshot_path))
        for path in snapshot_path.rglob("*")
        if path.is_file()
    )
    _write_json(output / "before.json", result)
    semantic_code.semantic_graph.close()
    return result


def _load(repo_path: Path, output: Path) -> dict:
    with patch.object(
        global_model_manager,
        "get_or_load_model",
        return_value=_ValidationEmbeddingModel(),
    ):
        graph = SemanticGraph.load_graph(str(output / "graph"))
    semantic_code = SemanticCode(graph, initialize_spaces=False)
    repository = semantic_code.get_repository()
    if repository is not None:
        semantic_code.repository_uid = repository.uid
        semantic_code.repository_path = str(repo_path)
    result = {
        "fingerprint": RepositorySnapshotFingerprint.from_graph(graph).to_dict(),
        "queries": query_snapshot(semantic_code),
        "quality": evaluate_repository_quality(semantic_code, repo_path).to_dict(),
    }
    _write_json(output / "after.json", result)
    graph.close()
    return result


def run_round_trip(
    repo_path: str | Path,
    output_path: str | Path,
    *,
    include_git_history: bool = False,
) -> dict:
    """Build and load in two fresh child processes, then compare exact snapshots."""
    repo = Path(repo_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, "-m", "mandol.software.repository_validation"]
    build_command = [*common, "_build", str(repo), str(output)]
    if include_git_history:
        build_command.append("--include-git-history")
    subprocess.run(build_command, check=True)
    subprocess.run([*common, "_load", str(repo), str(output)], check=True)
    before = json.loads((output / "before.json").read_text(encoding="utf-8"))
    after = json.loads((output / "after.json").read_text(encoding="utf-8"))
    result = {
        "pass": before["fingerprint"] == after["fingerprint"]
        and before["queries"] == after["queries"],
        "fingerprint_match": before["fingerprint"] == after["fingerprint"],
        "query_match": before["queries"] == after["queries"],
        "before": before["fingerprint"],
        "after": after["fingerprint"],
        "queries_before": before["queries"],
        "queries_after": after["queries"],
        "artifacts": before["artifacts"],
    }
    _write_json(output / "roundtrip_result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("roundtrip", "_build", "_load"))
    parser.add_argument("repo_path")
    parser.add_argument("output_path")
    parser.add_argument("--include-git-history", action="store_true")
    args = parser.parse_args()
    repo = Path(args.repo_path).expanduser().resolve()
    output = Path(args.output_path).expanduser().resolve()
    if args.mode == "_build":
        result = _build(
            repo,
            output,
            include_git_history=args.include_git_history,
        )
    elif args.mode == "_load":
        result = _load(repo, output)
    else:
        result = run_round_trip(
            repo,
            output,
            include_git_history=args.include_git_history,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["run_round_trip"]
