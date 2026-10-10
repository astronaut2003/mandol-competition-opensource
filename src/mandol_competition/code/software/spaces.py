"""Dynamic Mandol MemorySpaces for software-domain units."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .schemas import SoftwareUnitType


class SoftwareSpace(str, Enum):
    """Software spaces kept separate from Mandol's canonical tower registry."""

    SOFTWARE = "software"
    CODE = "software:code"
    ISSUES = "software:issues"
    COMMITS = "software:commits"
    RUNTIME = "software:runtime"
    EVOLUTION = "software:evolution"
    HIGH_LEVEL = "software:high_level"
    HIGH_LEVEL_FILE = "software:high_level:file"
    HIGH_LEVEL_SYMBOL = "software:high_level:symbol"
    HIGH_LEVEL_REPAIR = "software:high_level:repair"
    HIGH_LEVEL_MODULE = "software:high_level:module"
    HIGH_LEVEL_ARCHITECTURE = "software:high_level:architecture"
    HIGH_LEVEL_EPISODE = "software:high_level:episode"


_TYPE_TO_SPACE = {
    SoftwareUnitType.REPOSITORY: SoftwareSpace.CODE,
    SoftwareUnitType.DIRECTORY: SoftwareSpace.CODE,
    SoftwareUnitType.FILE: SoftwareSpace.CODE,
    SoftwareUnitType.CLASS: SoftwareSpace.CODE,
    SoftwareUnitType.FUNCTION: SoftwareSpace.CODE,
    SoftwareUnitType.CODE_SEGMENT: SoftwareSpace.CODE,
    SoftwareUnitType.SEMANTIC_BLOCK: SoftwareSpace.CODE,
    SoftwareUnitType.ISSUE: SoftwareSpace.ISSUES,
    SoftwareUnitType.COMMIT: SoftwareSpace.COMMITS,
    SoftwareUnitType.CONFIG: SoftwareSpace.RUNTIME,
    SoftwareUnitType.LOG: SoftwareSpace.RUNTIME,
    SoftwareUnitType.EXECUTION_TRACE: SoftwareSpace.RUNTIME,
    SoftwareUnitType.EVOLUTION: SoftwareSpace.EVOLUTION,
    SoftwareUnitType.FILE_MEMORY: SoftwareSpace.HIGH_LEVEL_FILE,
    SoftwareUnitType.SYMBOL_MEMORY: SoftwareSpace.HIGH_LEVEL_SYMBOL,
    SoftwareUnitType.REPAIR_MEMORY: SoftwareSpace.HIGH_LEVEL_REPAIR,
    SoftwareUnitType.MODULE_MEMORY: SoftwareSpace.HIGH_LEVEL_MODULE,
    SoftwareUnitType.ARCHITECTURE_MEMORY: SoftwareSpace.HIGH_LEVEL_ARCHITECTURE,
    SoftwareUnitType.ENGINEERING_EPISODE: SoftwareSpace.HIGH_LEVEL_EPISODE,
}


def software_space_for_type(software_type: SoftwareUnitType | str) -> SoftwareSpace:
    """Return the leaf software space for an entity type."""
    return _TYPE_TO_SPACE[SoftwareUnitType(software_type)]


def initialize_software_spaces(semantic_system: Any) -> None:
    """Create software spaces using an existing SemanticGraph or SemanticMap."""
    semantic_map = getattr(semantic_system, "semantic_map", semantic_system)
    if not hasattr(semantic_map, "create_memory_space") or not hasattr(
        semantic_map, "add_space_to_space"
    ):
        raise TypeError(
            "initialize_software_spaces() expects a SemanticGraph or SemanticMap-like object"
        )

    for space in SoftwareSpace:
        semantic_map.create_memory_space(space.value)
    top_level_children = (
        SoftwareSpace.CODE,
        SoftwareSpace.ISSUES,
        SoftwareSpace.COMMITS,
        SoftwareSpace.RUNTIME,
        SoftwareSpace.EVOLUTION,
        SoftwareSpace.HIGH_LEVEL,
    )
    for child in top_level_children:
        semantic_map.add_space_to_space(child.value, SoftwareSpace.SOFTWARE.value)
    for child in (
        SoftwareSpace.HIGH_LEVEL_FILE,
        SoftwareSpace.HIGH_LEVEL_SYMBOL,
        SoftwareSpace.HIGH_LEVEL_REPAIR,
        SoftwareSpace.HIGH_LEVEL_MODULE,
        SoftwareSpace.HIGH_LEVEL_ARCHITECTURE,
        SoftwareSpace.HIGH_LEVEL_EPISODE,
    ):
        semantic_map.add_space_to_space(child.value, SoftwareSpace.HIGH_LEVEL.value)


__all__ = [
    "SoftwareSpace",
    "initialize_software_spaces",
    "software_space_for_type",
]
