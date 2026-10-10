"""Build and query SemanticCode from a real local repository."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from ..core.semantic_graph import SemanticGraph
from ..core.semantic_map import SemanticMap
from ..utils.model_manager import global_model_manager
from .demo import _DemoEmbeddingModel
from .facade import SemanticCode


def build_repository_demo(repo_path: str) -> SemanticCode:
    """Build an offline demo using Mandol's real batch and graph paths."""
    with patch.object(
        global_model_manager,
        "get_or_load_model",
        return_value=_DemoEmbeddingModel(),
    ):
        semantic_map = SemanticMap(
            embedding_model_name="semantic-code/repository-demo",
            embedding_dim=_DemoEmbeddingModel.dimension,
            use_flash_attention=False,
        )
    return SemanticCode.from_repository(
        repo_path,
        semantic_graph=SemanticGraph(semantic_map),
        max_git_commits=20,
        build_indexes=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_path", nargs="?", default=".")
    args = parser.parse_args()
    repo_path = str(Path(args.repo_path).expanduser().resolve())
    semantic_code = build_repository_demo(repo_path)

    functions = semantic_code.search(
        "authentication timeout handling",
        software_types=["function"],
        top_k=3,
    )
    selected = functions[0] if functions else None
    print("repository:", semantic_code.get_repository())
    print("function:", selected)
    if selected:
        print("symbol_memory:", semantic_code.get_symbol_memory(selected.uid))
        print(
            "history:", [unit.uid for unit in semantic_code.get_history(selected.uid)]
        )
    module_name = (
        selected.metadata.get("module_name", "").split(".", 1)[0] if selected else ""
    )
    print(
        "module_memory:",
        semantic_code.get_module_memory(module_name) if module_name else None,
    )
    print("architecture_memory:", semantic_code.get_architecture_memory())
    print("search:", [unit.uid for unit in functions])
    print("build_report:", semantic_code.last_build_report.to_dict())


if __name__ == "__main__":
    main()
