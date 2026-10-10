"""TypeScript and TSX parser using the shared ECMAScript-family backend."""

from __future__ import annotations

from pathlib import PurePosixPath

from .javascript import JsTsParserBase


class TypeScriptParser(JsTsParserBase):
    """Parse TypeScript declarations and implementations with stable overload IDs."""

    language = "typescript"
    grammar_name = "typescript"

    def _grammar_for_path(self, relative_path: str) -> str:
        return "tsx" if PurePosixPath(relative_path).suffix.lower() == ".tsx" else "typescript"


__all__ = ["TypeScriptParser"]
