"""Local Git history analysis and diff-hunk to parsed symbol mapping."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .parser import LanguageParser, ParsedFile, PythonParser


@dataclass(frozen=True)
class DiffHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    added_lines: int = 0
    removed_lines: int = 0


@dataclass
class GitCommitFacts:
    commit_hash: str
    parents: list[str]
    author: str
    timestamp: str
    message: str
    changed_files: list[str] = field(default_factory=list)
    hunks: dict[str, list[DiffHunk]] = field(default_factory=dict)
    modified_symbols: list[str] = field(default_factory=list)
    diff_summary: str = ""


class GitAnalyzer:
    """Read Git metadata through the local command-line client, never the network."""

    _HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    def __init__(
        self,
        repo_path: str | Path,
        *,
        revision: str = "HEAD",
        max_commits: int | None = 100,
        parser: PythonParser | None = None,
        parsers: Mapping[str, LanguageParser] | None = None,
    ) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.revision = revision
        self.max_commits = max_commits
        self.parsers: dict[str, LanguageParser] = dict(parsers or {})
        self.parsers.setdefault(
            "python", parser or PythonParser(build_code_segments=False)
        )
        self.git_root: Path | None = None
        self.repo_prefix = ""

    @property
    def available(self) -> bool:
        return self._discover_git_root()

    def revision_hash(self) -> str:
        if not self._discover_git_root():
            return self.revision
        return (
            self._run("rev-parse", "--verify", f"{self.revision}^{{commit}}")
            or self.revision
        )

    def default_branch(self) -> str:
        if not self._discover_git_root():
            return "main"
        remote_head = self._run(
            "symbolic-ref", "--quiet", "--short", "refs/remotes/origin/HEAD"
        )
        if remote_head:
            return remote_head.rsplit("/", 1)[-1]
        branch = self._run("branch", "--show-current")
        return branch or "main"

    def analyze(self, current_files: dict[str, ParsedFile]) -> list[GitCommitFacts]:
        if not self._discover_git_root():
            return []
        revision_hash = self.revision_hash()
        args = ["log", "--format=%H%x09%P%x09%an%x09%aI%x09%s"]
        if self.max_commits is not None:
            args.extend(["-n", str(max(0, self.max_commits))])
        args.append(revision_hash)
        if self.repo_prefix:
            args.extend(["--", self.repo_prefix])
        rows = self._run(*args).splitlines()
        commits: list[GitCommitFacts] = []
        for row in rows:
            fields = row.split("\t", 4)
            if len(fields) != 5:
                continue
            commit_hash, parents_text, author, timestamp, message = fields
            parents = parents_text.split() if parents_text else []
            changed_git_paths = self._changed_paths(commit_hash)
            changed_files = [
                path
                for git_path in changed_git_paths
                if (path := self._repo_relative(git_path)) is not None
            ]
            hunks: dict[str, list[DiffHunk]] = {}
            modified_symbols: set[str] = set()
            added_total = 0
            removed_total = 0
            for relative_path in changed_files:
                git_path = self._git_relative(relative_path)
                file_hunks = self._diff_hunks(
                    commit_hash, parents[0] if parents else None, git_path
                )
                hunks[relative_path] = file_hunks
                added_total += sum(hunk.added_lines for hunk in file_hunks)
                removed_total += sum(hunk.removed_lines for hunk in file_hunks)
                if relative_path in current_files:
                    current_file = current_files[relative_path]
                    snapshot = self._snapshot(
                        commit_hash,
                        git_path,
                        relative_path,
                        current_file.language,
                    )
                    parsed = snapshot or current_files[relative_path]
                    for function in parsed.functions:
                        if any(
                            self._overlaps(function.start_line, function.end_line, hunk)
                            for hunk in file_hunks
                        ):
                            current_qualified_names = {
                                item.symbol_identity or item.qualified_name
                                for item in current_files[relative_path].functions
                            }
                            identity = (
                                function.symbol_identity or function.qualified_name
                            )
                            if identity in current_qualified_names:
                                language_prefix = (
                                    ""
                                    if current_file.language == "python"
                                    else f"{current_file.language}:"
                                )
                                modified_symbols.add(
                                    f"function:{language_prefix}{identity}"
                                )
                    for class_fact in parsed.classes:
                        if any(
                            self._overlaps(
                                class_fact.start_line, class_fact.end_line, hunk
                            )
                            for hunk in file_hunks
                        ):
                            current_classes = {
                                item.qualified_name
                                for item in current_files[relative_path].classes
                            }
                            if class_fact.qualified_name in current_classes:
                                language_prefix = (
                                    ""
                                    if current_file.language == "python"
                                    else f"{current_file.language}:"
                                )
                                modified_symbols.add(
                                    f"class:{language_prefix}{class_fact.qualified_name}"
                                )
            summary = f"{message}: {len(changed_files)} file(s), +{added_total}/-{removed_total} lines."
            commits.append(
                GitCommitFacts(
                    commit_hash=commit_hash,
                    parents=parents,
                    author=author,
                    timestamp=timestamp,
                    message=message,
                    changed_files=changed_files,
                    hunks=hunks,
                    modified_symbols=sorted(modified_symbols),
                    diff_summary=summary,
                )
            )
        return commits

    @staticmethod
    def classify_change(message: str) -> str:
        lowered = message.lower()
        if any(
            token in lowered
            for token in (
                "fix",
                "bug",
                "regression",
                "crash",
                "error",
                "incorrect",
                "timeout",
                "race",
            )
        ):
            return "bug_fix"
        if any(token in lowered for token in ("feat", "feature", "add ", "implement")):
            return "feature"
        if "refactor" in lowered:
            return "refactor"
        if "test" in lowered:
            return "test"
        if "doc" in lowered:
            return "documentation"
        return "change"

    @staticmethod
    def issue_ids(message: str) -> list[str]:
        return sorted(set(re.findall(r"(?<![\w])#(\d+)", message)))

    def _discover_git_root(self) -> bool:
        if self.git_root is not None:
            return True
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.repo_path), "rev-parse", "--show-toplevel"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            return False
        root_text = completed.stdout.strip()
        if not root_text:
            return False
        self.git_root = Path(root_text).resolve()
        try:
            relative = self.repo_path.relative_to(self.git_root)
        except ValueError:
            return False
        self.repo_prefix = "" if relative.as_posix() == "." else relative.as_posix()
        return True

    def _run(self, *args: str, allow_failure: bool = False) -> str:
        if self.git_root is None and not self._discover_git_root():
            return ""
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.git_root), *args],
                check=not allow_failure,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return completed.stdout.strip()

    def _changed_paths(self, commit_hash: str) -> list[str]:
        output = self._run(
            "diff-tree",
            "--root",
            "--no-commit-id",
            "--name-only",
            "-r",
            commit_hash,
        )
        return [line for line in output.splitlines() if line]

    def _diff_hunks(
        self, commit_hash: str, parent: str | None, git_path: str
    ) -> list[DiffHunk]:
        if parent:
            output = self._run(
                "diff",
                "--unified=0",
                "--no-renames",
                parent,
                commit_hash,
                "--",
                git_path,
            )
        else:
            output = self._run(
                "show",
                "--format=",
                "--unified=0",
                "--no-renames",
                commit_hash,
                "--",
                git_path,
            )
        hunks: list[DiffHunk] = []
        current: dict[str, int] | None = None
        for line in output.splitlines():
            match = self._HUNK_RE.match(line)
            if match:
                if current:
                    hunks.append(DiffHunk(**current))
                current = {
                    "old_start": int(match.group(1)),
                    "old_count": int(match.group(2) or "1"),
                    "new_start": int(match.group(3)),
                    "new_count": int(match.group(4) or "1"),
                    "added_lines": 0,
                    "removed_lines": 0,
                }
            elif current and line.startswith("+") and not line.startswith("+++"):
                current["added_lines"] += 1
            elif current and line.startswith("-") and not line.startswith("---"):
                current["removed_lines"] += 1
        if current:
            hunks.append(DiffHunk(**current))
        return hunks

    def _snapshot(
        self,
        commit_hash: str,
        git_path: str,
        relative_path: str,
        language: str,
    ) -> ParsedFile | None:
        source = self._run("show", f"{commit_hash}:{git_path}", allow_failure=True)
        parser = self.parsers.get(language)
        if not source or parser is None:
            return None
        parse_source = getattr(parser, "parse_source", None)
        if not callable(parse_source):
            return None
        return parse_source(source + "\n", relative_path)

    def _repo_relative(self, git_path: str) -> str | None:
        path = PurePosixPath(git_path)
        if not self.repo_prefix:
            return path.as_posix()
        prefix = PurePosixPath(self.repo_prefix)
        try:
            return path.relative_to(prefix).as_posix()
        except ValueError:
            return None

    def _git_relative(self, repo_relative_path: str) -> str:
        if not self.repo_prefix:
            return repo_relative_path
        return (PurePosixPath(self.repo_prefix) / repo_relative_path).as_posix()

    @staticmethod
    def _overlaps(start_line: int, end_line: int, hunk: DiffHunk) -> bool:
        hunk_start = hunk.new_start
        hunk_end = hunk.new_start + max(1, hunk.new_count) - 1
        return start_line <= hunk_end and hunk_start <= end_line


__all__ = ["DiffHunk", "GitAnalyzer", "GitCommitFacts"]
