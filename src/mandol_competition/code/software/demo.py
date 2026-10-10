"""Runnable SemanticCode reference demo without external model downloads."""

from __future__ import annotations

import hashlib
import re
from unittest.mock import patch

import numpy as np

from ..core.semantic_graph import SemanticGraph
from ..core.semantic_map import SemanticMap
from ..utils.model_manager import global_model_manager
from .facade import SemanticCode
from .relations import SoftwareRelation
from .schemas import (
    CommitSchema,
    ConfigSchema,
    EvolutionSchema,
    ExecutionTraceSchema,
    FileSchema,
    FunctionSchema,
    IssueSchema,
    ModuleMemorySchema,
    RepositorySchema,
)


class _DemoEmbeddingModel:
    """Small deterministic token hashing model used only by this demo."""

    dimension = 64

    def _encode_one(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token in re.findall(r"[a-z0-9_]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:2], "big") % self.dimension] += 1.0
        return vector

    def encode(self, texts, **kwargs):
        del kwargs
        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.stack([self._encode_one(text) for text in texts])


def build_demo() -> SemanticCode:
    """Construct the demo graph and return its facade."""
    with patch.object(
        global_model_manager,
        "get_or_load_model",
        return_value=_DemoEmbeddingModel(),
    ):
        semantic_map = SemanticMap(
            embedding_model_name="semantic-code/demo",
            embedding_dim=_DemoEmbeddingModel.dimension,
            use_flash_attention=False,
        )
    semantic_code = SemanticCode(SemanticGraph(semantic_map))

    units = [
        RepositorySchema(
            name="example",
            path=".",
            default_branch="main",
            description="Example authentication service repository.",
        ),
        FileSchema(
            path="auth.py",
            language="python",
            exports=["authenticate", "validate_credentials"],
            semantic_summary="Authentication policy and credential validation.",
        ),
        FunctionSchema(
            name="authenticate",
            qualified_name="auth.authenticate",
            signature="authenticate(user, password) -> bool",
            language="python",
            file_path="auth.py",
            parameters=["user", "password"],
            return_type="bool",
            reads=["auth.timeout"],
            calls=["auth.validate_credentials"],
            semantic_summary="Authenticate a user with the configured authentication timeout policy.",
        ),
        FunctionSchema(
            name="validate_credentials",
            qualified_name="auth.validate_credentials",
            signature="validate_credentials(user, password) -> bool",
            language="python",
            file_path="auth.py",
            parameters=["user", "password"],
            return_type="bool",
            semantic_summary="Validate stored credentials.",
        ),
        IssueSchema(
            issue_id="123",
            title="Authentication timeout rejects valid logins",
            body="The authentication timeout is applied before credential validation.",
            state="closed",
            labels=["bug", "authentication"],
            created_at="2026-01-10T09:00:00Z",
            closed_at="2026-01-12T10:00:00Z",
        ),
        CommitSchema(
            commit_hash="abc123",
            message="Fix authentication timeout ordering",
            author="Demo Developer",
            timestamp="2026-01-12T10:00:00Z",
            changed_files=["auth.py"],
        ),
        ConfigSchema(
            key="auth.timeout",
            value=30,
            source_file="config.yaml",
            scope="authentication",
            semantic_summary="Maximum authentication operation time in seconds.",
        ),
        ExecutionTraceSchema(
            trace_id="login-1",
            entry_function="function:auth.authenticate",
            call_sequence=[
                "function:auth.authenticate",
                "function:auth.validate_credentials",
            ],
            metrics={"duration_ms": 42},
        ),
        EvolutionSchema(
            subject_uid="function:auth.authenticate",
            change_type="bug_fix",
            timestamp="2026-01-12T10:00:00Z",
            diff_summary="Validate credentials before enforcing the timeout result.",
            source_evidence=["commit:abc123", "issue:123"],
            impacted_units=["function:auth.authenticate", "config:auth.timeout"],
            confidence=0.95,
        ),
        ModuleMemorySchema(
            module_name="auth",
            responsibilities=["Authenticate users", "Validate credentials"],
            key_entities=[
                "function:auth.authenticate",
                "function:auth.validate_credentials",
            ],
            important_dependencies=["config:auth.timeout"],
            known_risks=["Timeout ordering can reject valid logins"],
            semantic_summary="The auth module owns login authentication policy.",
            evolution_summary="Commit abc123 corrected timeout ordering.",
            derived_from=[
                "file:auth.py",
                "commit:abc123",
                "issue:123",
            ],
        ),
    ]
    for schema in units:
        semantic_code.add_software_unit(
            schema,
            index_update_mode="none",
            generate_sparse_embedding=False,
        )

    relationships = [
        ("repo:example", "file:auth.py", SoftwareRelation.CONTAINS),
        ("file:auth.py", "function:auth.authenticate", SoftwareRelation.CONTAINS),
        (
            "function:auth.authenticate",
            "function:auth.validate_credentials",
            SoftwareRelation.CALLS,
        ),
        ("commit:abc123", "function:auth.authenticate", SoftwareRelation.MODIFIES),
        ("issue:123", "commit:abc123", SoftwareRelation.FIXED_BY),
        (
            "config:auth.timeout",
            "function:auth.authenticate",
            SoftwareRelation.AFFECTS_BEHAVIOR_OF,
        ),
        (
            "function:auth.authenticate",
            "trace:login-1",
            SoftwareRelation.RELATED_TO_TRACE,
        ),
        (
            "function:auth.authenticate",
            "evolution:auth.authenticate:abc123",
            SoftwareRelation.HAS_EVOLUTION,
        ),
        (
            "evolution:auth.authenticate:abc123",
            "commit:abc123",
            SoftwareRelation.CAUSED_BY_EVIDENCE,
        ),
        (
            "evolution:auth.authenticate:abc123",
            "issue:123",
            SoftwareRelation.CAUSED_BY_EVIDENCE,
        ),
    ]
    for source_uid, target_uid, relation in relationships:
        semantic_code.add_relation(source_uid, target_uid, relation)

    return semantic_code


def main() -> None:
    semantic_code = build_demo()
    query_results = semantic_code.search("authentication timeout", top_k=3)
    commits = semantic_code.get_related_commits("function:auth.authenticate")
    issues = semantic_code.get_related_issues("function:auth.authenticate")
    calls = semantic_code.traverse(
        "function:auth.authenticate",
        SoftwareRelation.CALLS,
        direction="outgoing",
    )
    history = semantic_code.get_history("function:auth.authenticate")
    module_memory = semantic_code.get_module_memory("auth")

    print("search:", [unit.uid for unit in query_results])
    print("commits:", [unit.uid for unit in commits])
    print("issues:", [unit.uid for unit in issues])
    print("calls:", [unit.uid for unit in calls])
    print("history:", [unit.uid for unit in history])
    print("module_memory:", module_memory.uid if module_memory else None)


if __name__ == "__main__":
    main()
