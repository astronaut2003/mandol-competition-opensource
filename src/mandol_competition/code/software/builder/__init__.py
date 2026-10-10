"""Automatic local repository ingestion for SemanticCode."""

from .cache import BUILDER_VERSION, BuilderCache
from .git_analyzer import DiffHunk, GitAnalyzer, GitCommitFacts
from .high_level_builder import HighLevelMemoryBuilder, SemanticEnricher
from .parser import (
    CParser,
    CppParser,
    GoParser,
    JavaScriptParser,
    JavaParser,
    LanguageParser,
    ParsedCall,
    ParsedClass,
    ParsedCodeSegment,
    ParsedFile,
    ParsedFunction,
    PythonParser,
    RustParser,
    TypeScriptParser,
)
from .repository_builder import RepositoryBuildReport, RepositoryMemoryBuilder
from .scanner import ManifestEntry, RepositoryManifest, RepositoryScanner

__all__ = [
    "BUILDER_VERSION",
    "BuilderCache",
    "CParser",
    "CppParser",
    "DiffHunk",
    "GitAnalyzer",
    "GitCommitFacts",
    "HighLevelMemoryBuilder",
    "GoParser",
    "JavaScriptParser",
    "JavaParser",
    "LanguageParser",
    "ManifestEntry",
    "ParsedCall",
    "ParsedClass",
    "ParsedCodeSegment",
    "ParsedFile",
    "ParsedFunction",
    "PythonParser",
    "RustParser",
    "RepositoryBuildReport",
    "RepositoryManifest",
    "RepositoryMemoryBuilder",
    "RepositoryScanner",
    "SemanticEnricher",
    "TypeScriptParser",
]
