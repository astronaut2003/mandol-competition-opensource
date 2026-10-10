"""Relationship names used by the software-domain facade."""

from enum import Enum


class SoftwareRelation(str, Enum):
    """Software relationships stored as ordinary SemanticGraph edge types."""

    CONTAINS = "CONTAINS"
    HAS_METHOD = "HAS_METHOD"
    IMPORTS = "IMPORTS"

    CALLS = "CALLS"
    INHERITS_FROM = "INHERITS_FROM"
    IMPLEMENTS = "IMPLEMENTS"
    CONTAINED_IN = "CONTAINED_IN"
    NEXT_SEGMENT = "NEXT_SEGMENT"
    # Deprecated read/API compatibility; new builders emit NEXT_SEGMENT.
    NEXT_BLOCK = "NEXT_BLOCK"
    DATA_DEPENDS_ON = "DATA_DEPENDS_ON"

    MODIFIES = "MODIFIES"
    FIXED_BY = "FIXED_BY"
    MENTIONS = "MENTIONS"
    TESTED_BY = "TESTED_BY"
    AFFECTS_BEHAVIOR_OF = "AFFECTS_BEHAVIOR_OF"
    AUTHORED_BY = "AUTHORED_BY"
    LOGGED_BY = "LOGGED_BY"
    RELATED_TO_TRACE = "RELATED_TO_TRACE"

    HAS_EVOLUTION = "HAS_EVOLUTION"
    EVOLVES_FROM = "EVOLVES_FROM"
    CAUSED_BY_EVIDENCE = "CAUSED_BY_EVIDENCE"
    IMPACTS = "IMPACTS"

    DERIVED_FROM = "DERIVED_FROM"
    SUPPORTED_BY = "SUPPORTED_BY"


__all__ = ["SoftwareRelation"]
