"""Conservative, reversible evidence groups; similarity never owns identity.

A group is an immutable navigation/packing plan, not a canonical merge. Callers
must supply already eligible items. Persistence is a BuildJob proposal/receipt,
not a second memory store. Profile lifecycle stays in its existing A/C protocol.
"""

from dataclasses import asdict, dataclass
from itertools import combinations

from mandol_substrate.state.models import digest


@dataclass(frozen=True)
class ConsolidationGroup:
    group_id: str
    kind: str
    operation: str
    source_derived: tuple[tuple[str, int, str], ...]
    supporting_raw_uids: tuple[str, ...]
    contextual_raw_uids: tuple[str, ...]
    reason: str
    method: str
    recipe: str = "conservative-dedup-v3"

    @property
    def backlinks(self):
        return tuple(
            (self.group_id, uid, "GROUP_HAS_MEMBER", "MEMBER_OF_GROUP")
            for uid, _, _ in self.source_derived
        )


def evidence_key(item):
    return tuple(
        sorted(
            (e["source_uid"], e["quote"])
            for e in item.get("evidence", item.get("mentions", []))
        )
    )


def structural_key(kind, memory):
    """Hard partition before any optional similarity/LLM candidate selection."""
    i = memory.metadata["hlm_item"]
    common = (memory.metadata["user_id"], memory.metadata["raw_namespace"], kind)
    if kind == "profile":
        p = i["profile_item"]
        return (
            *common,
            *(
                p.get(k)
                for k in ("subject", "attribute", "scope", "condition", "lifetime")
            ),
        )
    if kind == "episode":
        return (
            *common,
            *(
                i.get(k)
                for k in (
                    "actor",
                    "action",
                    "object",
                    "event_type",
                    "assertion_kind",
                    "modality",
                    "discourse_mode",
                    "status",
                    "location",
                )
            ),
            tuple(sorted(i["participants"])),
        )
    if kind == "entity":
        # Names are only candidate keys. They are never merge authority.
        return (*common, i["kind"], i["name"], i["identity_context"])
    if kind == "relation":
        # Local E numbers alone are NOT canonical endpoint IDs across batches.
        return (
            *common,
            memory.metadata["hlm_job_id"],
            *(
                i.get(k)
                for k in (
                    "subject_id",
                    "predicate",
                    "object_id",
                    "status",
                    "condition",
                    "valid_from",
                    "valid_to",
                    "modality",
                )
            ),
        )
    raise ValueError("No clustering route for this representation")


def classify_pair(kind, a, b, *, mode="structural"):
    if mode not in ("exact", "structural"):
        raise ValueError("Only predeclared deterministic merge modes")
    if structural_key(kind, a) != structural_key(kind, b):
        return "KEEP_SEPARATE", "INCOMPATIBLE_STRUCTURAL_PARTITION"
    x, y = a.metadata["hlm_item"], b.metadata["hlm_item"]
    if kind == "profile":
        return "KEEP_SEPARATE", "EXISTING_PROFILE_LIFECYCLE_OWNS_MUTATION"
    if a.metadata["source_versions"] != b.metadata["source_versions"]:
        shared = set(a.metadata["source_versions"]) & set(b.metadata["source_versions"])
        if any(
            a.metadata["source_versions"][u] != b.metadata["source_versions"][u]
            for u in shared
        ):
            return "KEEP_SEPARATE", "SOURCE_VERSION_CONFLICT"
    if (
        digest(x) == digest(y)
        and a.metadata["source_versions"] == b.metadata["source_versions"]
    ):
        return "EXACT_DUPLICATE", "IDENTICAL_ASSERTION_AND_VERSIONED_EVIDENCE"
    if mode == "exact":
        return "KEEP_SEPARATE", "NOT_EXACT"
    if kind == "entity":
        if set(evidence_key(x)) & set(evidence_key(y)):
            return "ENTITY_RESOLUTION", "SHARED_EXACT_MENTION_AND_IDENTITY_CONTEXT"
        return "KEEP_SEPARATE", "NAME_ONLY_OR_UNPROVEN_ALIAS"
    if kind == "relation":
        if x["text"] == y["text"] and evidence_key(x) == evidence_key(y):
            return "SAME_ASSERTION", "EXACT_ENDPOINTS_QUALIFIERS_AND_EVIDENCE"
        return "KEEP_SEPARATE", "UNPROVEN_SAME_ASSERTION"
    if kind == "episode":
        tx, ty = (
            (x["time_precision"], x["event_time"]),
            (y["time_precision"], y["event_time"]),
        )
        if tx != ty:
            if (
                tx[0] in ("EXACT", "DAY")
                and ty[0] in ("EXACT", "DAY")
                and tx[1]
                and ty[1]
            ):
                return (
                    "REPEATED_OCCURRENCE_GROUP",
                    "DIFFERENT_DATES_OR_OCCURRENCES_RETAINED",
                )
            return "KEEP_SEPARATE", "TIME_NOT_COMPARABLE"
        if (
            set(evidence_key(x)) & set(evidence_key(y))
            and x["state_before"] == y["state_before"]
            and x["state_after"] == y["state_after"]
        ):
            return "SAME_OCCURRENCE_MERGE", "SHARED_OCCURRENCE_EVIDENCE_AND_STATE"
        return "KEEP_SEPARATE", "SAME_TIME_ALONE_DOES_NOT_PROVE_SAME_OCCURRENCE"
    return "KEEP_SEPARATE", "NO_AUTHORITY"


def groups(kind, memories, *, mode="structural", pair_limit=4096):
    partitions = {}
    for uid, m in memories.items():
        if uid != m.logical_uid or not m.active:
            raise ValueError("Only admitted current records can be grouped")
        partitions.setdefault(structural_key(kind, m), []).append(m)
    result = []
    examined = 0
    for key in sorted(partitions, key=repr):
        for a, b in combinations(
            sorted(partitions[key], key=lambda m: m.logical_uid), 2
        ):
            if examined >= pair_limit:
                raise ValueError("Explicit consolidation pair budget exceeded")
            examined += 1
            operation, reason = classify_pair(kind, a, b, mode=mode)
            if operation == "KEEP_SEPARATE":
                continue
            originals = tuple(
                (m.logical_uid, m.version, digest(m.metadata["hlm_item"]))
                for m in (a, b)
            )
            support = tuple(
                sorted(
                    {u for m in (a, b) for u, _ in evidence_key(m.metadata["hlm_item"])}
                )
            )
            context = tuple(
                sorted(
                    {
                        u
                        for m in (a, b)
                        for u in m.metadata["hlm_item"].get("context_source_uids", [])
                    }
                    - set(support)
                )
            )
            result.append(
                ConsolidationGroup(
                    "group_" + digest([kind, operation, originals]),
                    kind,
                    operation,
                    originals,
                    support,
                    context,
                    reason,
                    "STRUCTURAL" if mode == "structural" else "EXACT",
                )
            )
    return result


def representatives(kind, memories, *, mode="structural"):
    """Read-only redundancy control. Repeated occurrences never lose a slot here."""
    plans = groups(kind, memories, mode=mode)
    # No transitive union: A~B and B~C cannot silently authorize A~C.
    retained = dict(memories)
    removed = {}
    for plan in plans:
        if plan.operation == "REPEATED_OCCURRENCE_GROUP":
            continue
        a, b = (u for u, _, _ in plan.source_derived)
        if a in retained and b in retained:
            removed[b] = a
            retained.pop(b)
    return retained, {
        "groups": [asdict(p) for p in plans],
        "backlinks": [link for p in plans for link in p.backlinks],
        "representatives": removed,
        "source_records_deleted": 0,
    }
