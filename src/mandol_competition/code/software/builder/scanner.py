"""Single-pass repository discovery with conservative ignore rules."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath

DEFAULT_IGNORED_DIRECTORIES = {
    ".git",
    ".hg",
    ".svn",
    ".mandol",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "target",
    "vendor",
    "venv",
}

IGNORED_SUFFIXES = {
    ".a",
    ".avi",
    ".bin",
    ".bmp",
    ".class",
    ".dll",
    ".dylib",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".lockb",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".obj",
    ".pdf",
    ".png",
    ".pyc",
    ".pyo",
    ".so",
    ".tar",
    ".tiff",
    ".wav",
    ".webp",
    ".whl",
    ".xz",
    ".zip",
}

CONFIG_SUFFIXES = {".cfg", ".conf", ".env", ".ini", ".json", ".toml", ".yaml", ".yml"}
CONFIG_NAMES = {
    "Cargo.toml",
    "Dockerfile",
    "Makefile",
    "Pipfile",
    "go.mod",
    "go.sum",
    "package.json",
    "pyproject.toml",
    "setup.cfg",
    "tsconfig.json",
    "tox.ini",
}
SOURCE_LANGUAGES = {
    ".py": "python",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".inl": "cpp",
    ".ipp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}


@dataclass(frozen=True)
class ManifestEntry:
    path: str
    absolute_path: str
    kind: str
    language: str
    content_hash: str
    size: int
    source_loc: int
    is_test: bool = False


@dataclass
class RepositoryManifest:
    root: str
    name: str
    files: list[ManifestEntry] = field(default_factory=list)
    directories: list[str] = field(default_factory=list)
    language_statistics: dict[str, int] = field(default_factory=dict)
    file_count: int = 0
    source_file_count: int = 0
    test_file_count: int = 0
    source_loc: int = 0


class _GitIgnoreMatcher:
    def __init__(self, patterns: Iterable[str]) -> None:
        self.patterns: list[tuple[str, bool, bool]] = []
        for raw in patterns:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            directory_only = line.endswith("/")
            line = line.strip("/")
            if line:
                self.patterns.append((line, negated, directory_only))

    def ignored(self, relative_path: str, *, is_dir: bool) -> bool:
        relative_path = PurePosixPath(relative_path).as_posix().strip("/")
        path = PurePosixPath(relative_path)
        ignored = False
        for pattern, negated, directory_only in self.patterns:
            if directory_only and not is_dir:
                # Files below an ignored directory are handled when os.walk prunes it.
                continue
            if "/" in pattern:
                matched = fnmatch.fnmatch(relative_path, pattern) or path.match(pattern)
                if is_dir:
                    matched = matched or relative_path.startswith(
                        f"{pattern.rstrip('/')}/"
                    )
            else:
                matched = any(fnmatch.fnmatch(part, pattern) for part in path.parts)
            if matched:
                ignored = not negated
        return ignored


class RepositoryScanner:
    """Discover useful repository files in one filesystem traversal."""

    def __init__(
        self, repo_path: str | Path, *, max_file_size: int = 1_000_000
    ) -> None:
        self.root = Path(repo_path).expanduser().resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository path is not a directory: {self.root}")
        self.max_file_size = max(1, int(max_file_size))
        gitignore = self.root / ".gitignore"
        patterns = (
            gitignore.read_text(encoding="utf-8", errors="replace").splitlines()
            if gitignore.is_file()
            else []
        )
        self.ignore_matcher = _GitIgnoreMatcher(patterns)

    def scan(self) -> RepositoryManifest:
        files: list[ManifestEntry] = []
        directories: set[str] = set()
        language_statistics: dict[str, int] = {}

        for current_root, dir_names, file_names in os.walk(
            self.root, followlinks=False
        ):
            current = Path(current_root)
            relative_current = current.relative_to(self.root)
            retained_dirs = []
            for name in sorted(dir_names):
                relative = (relative_current / name).as_posix()
                candidate = current / name
                if name in DEFAULT_IGNORED_DIRECTORIES or candidate.is_symlink():
                    continue
                if self.ignore_matcher.ignored(relative, is_dir=True):
                    continue
                retained_dirs.append(name)
            dir_names[:] = retained_dirs

            for name in sorted(file_names):
                path = current / name
                relative = path.relative_to(self.root).as_posix()
                if path.is_symlink() or self.ignore_matcher.ignored(
                    relative, is_dir=False
                ):
                    continue
                entry = self._entry(path, relative)
                if entry is None:
                    continue
                files.append(entry)
                parent = PurePosixPath(relative).parent
                while parent.as_posix() not in (".", ""):
                    directories.add(parent.as_posix())
                    parent = parent.parent
                if entry.kind == "source" and entry.language:
                    language_statistics[entry.language] = (
                        language_statistics.get(entry.language, 0) + 1
                    )

        # In repositories containing explicit C++ translation units, plain .h
        # headers are parsed with the C++ grammar (which remains a C superset).
        if any(entry.language == "cpp" for entry in files):
            files = [
                replace(entry, language="cpp")
                if PurePosixPath(entry.path).suffix.lower() == ".h"
                and entry.language == "c"
                else entry
                for entry in files
            ]
            language_statistics = {}
            for entry in files:
                if entry.kind == "source" and entry.language:
                    language_statistics[entry.language] = (
                        language_statistics.get(entry.language, 0) + 1
                    )

        source_files = [entry for entry in files if entry.kind == "source"]
        return RepositoryManifest(
            root=str(self.root),
            name=self.root.name,
            files=files,
            directories=sorted(directories, key=lambda item: (item.count("/"), item)),
            language_statistics=dict(sorted(language_statistics.items())),
            file_count=len(files),
            source_file_count=len(source_files),
            test_file_count=sum(entry.is_test for entry in source_files),
            source_loc=sum(entry.source_loc for entry in source_files),
        )

    def _entry(self, path: Path, relative: str) -> ManifestEntry | None:
        try:
            size = path.stat().st_size
        except OSError:
            return None
        suffix = path.suffix.lower()
        if size > self.max_file_size or suffix in IGNORED_SUFFIXES:
            return None
        if path.name.endswith((".min.js", ".generated.py", "_pb2.py", "_pb2_grpc.py")):
            return None
        try:
            content = path.read_bytes()
        except OSError:
            return None
        if b"\x00" in content[:8192]:
            return None

        language = SOURCE_LANGUAGES.get(suffix, "")
        if language:
            kind = "source"
        elif suffix in CONFIG_SUFFIXES or path.name in CONFIG_NAMES:
            kind = "config"
        else:
            # Keep documentation and text artifacts out of the graph in phase one.
            return None
        path_obj = PurePosixPath(relative)
        stem_lower = path_obj.stem.lower()
        is_test = (
            any(part.lower() in {"test", "tests"} for part in path_obj.parts)
            or stem_lower.startswith("test_")
            or stem_lower.endswith(("_test", "test", "tests"))
            or stem_lower.endswith((".test", ".spec", "_spec"))
        )
        return ManifestEntry(
            path=relative,
            absolute_path=str(path),
            kind=kind,
            language=language or "configuration",
            content_hash=hashlib.sha256(content).hexdigest(),
            size=size,
            source_loc=content.count(b"\n") + (1 if content else 0),
            is_test=is_test,
        )


__all__ = ["ManifestEntry", "RepositoryManifest", "RepositoryScanner"]
