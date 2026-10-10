"""Convenience factories for software-domain MemoryUnits."""

from __future__ import annotations

from typing import Any

from mandol_substrate.representation import MemoryUnit
from .schemas import (
    ArchitectureMemorySchema,
    ClassSchema,
    CodeSegmentSchema,
    CommitSchema,
    ConfigSchema,
    DirectorySchema,
    EvolutionSchema,
    ExecutionTraceSchema,
    FileSchema,
    FileMemorySchema,
    FunctionSchema,
    IssueSchema,
    LogSchema,
    ModuleMemorySchema,
    RepairMemorySchema,
    RepositorySchema,
    SymbolMemorySchema,
)


def _create(schema: type[Any], **kwargs: Any) -> MemoryUnit:
    return schema(**kwargs).to_memory_unit()


def create_repository_unit(**kwargs: Any) -> MemoryUnit:
    return _create(RepositorySchema, **kwargs)


def create_directory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(DirectorySchema, **kwargs)


def create_file_unit(**kwargs: Any) -> MemoryUnit:
    return _create(FileSchema, **kwargs)


def create_class_unit(**kwargs: Any) -> MemoryUnit:
    return _create(ClassSchema, **kwargs)


def create_function_unit(**kwargs: Any) -> MemoryUnit:
    return _create(FunctionSchema, **kwargs)


def create_code_segment_unit(**kwargs: Any) -> MemoryUnit:
    return _create(CodeSegmentSchema, **kwargs)


def create_semantic_block_unit(**kwargs: Any) -> MemoryUnit:
    """Deprecated compatibility wrapper that creates a CodeSegment."""
    return create_code_segment_unit(**kwargs)


def create_issue_unit(**kwargs: Any) -> MemoryUnit:
    return _create(IssueSchema, **kwargs)


def create_commit_unit(**kwargs: Any) -> MemoryUnit:
    return _create(CommitSchema, **kwargs)


def create_config_unit(**kwargs: Any) -> MemoryUnit:
    return _create(ConfigSchema, **kwargs)


def create_log_unit(**kwargs: Any) -> MemoryUnit:
    return _create(LogSchema, **kwargs)


def create_execution_trace_unit(**kwargs: Any) -> MemoryUnit:
    return _create(ExecutionTraceSchema, **kwargs)


def create_evolution_unit(**kwargs: Any) -> MemoryUnit:
    return _create(EvolutionSchema, **kwargs)


def create_symbol_memory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(SymbolMemorySchema, **kwargs)


def create_file_memory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(FileMemorySchema, **kwargs)


def create_repair_memory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(RepairMemorySchema, **kwargs)


def create_module_memory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(ModuleMemorySchema, **kwargs)


def create_architecture_memory_unit(**kwargs: Any) -> MemoryUnit:
    return _create(ArchitectureMemorySchema, **kwargs)


__all__ = [name for name in globals() if name.startswith("create_")]
