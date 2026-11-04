"""Opt-in raw-evidence metadata, bound to one canonical publication view.

Observed Add order is not event time, validity or truth priority. No retrieval,
ranking, source text, model input, canonical write or baseline policy changes.
"""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re

from mandol_substrate.state.models import ProfileMismatch, StateUnavailable, digest


HISTORY = re.compile(
    r"\b(?:order|sequence|timeline|history|historical|chronolog\w*|evolv\w*|"
    r"progression|develop\w*|initially|originally|previously|formerly|current|"
    r"currently|latest|still|now|again)\b"
    r"|\b(?:used to|no longer|over time|at first|throughout our conversations)\b", re.I)


def renderer_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ObservationOrderPolicy:
    renderer_sha256: str
    version: str = "textual-observation-history-order-v1"
    max_added_chars: int = 2000

    def __post_init__(self):
        if (self.version != "textual-observation-history-order-v1"
                or type(self.max_added_chars) is not int or self.max_added_chars != 2000
                or self.renderer_sha256 != renderer_hash()):
            raise ProfileMismatch("Observation-order recipe/source fingerprint differs")

    @property
    def hash(self):
        return digest(asdict(self))

    def render(self, scope, hits, canonical, *, query, choices, published_seq):
        choices = [] if choices is None else choices
        if (not isinstance(query, str) or not isinstance(choices, list)
                or not all(isinstance(o, str) for o in choices)):
            raise ValueError("Observation-order routing needs actual query/options")
        if not HISTORY.search(query + "\n" + "\n".join(choices)):
            return hits, {"policy_hash": self.hash, "activated": False}
        if type(published_seq) is not int or published_seq < 0 or len(hits) > 100:
            raise StateUnavailable("Bounded held publication frontier required")
        ids = [hit["id"] for hit in hits]
        if len(ids) != len(set(ids)) or not set(ids) <= set(canonical):
            raise StateUnavailable("Observation-order hydration differs")
        output, positions, used = [], [], 0
        for hit in hits:
            memory = canonical[hit["id"]]
            if (memory.logical_uid != hit["id"] or memory.metadata.get("user_id") != scope.user_id
                    or not memory.metadata.get("aml") or not memory.active or memory.lifecycle == "deleted"
                    or memory.memory_level != "L0_Observation"
                    or type(memory.commit_seq) is not int or not 1 <= memory.commit_seq <= published_seq
                    or type(memory.turn_index) is not int or memory.turn_index < 0
                    or scope.memory_uid(memory.session_id, memory.request_id, memory.turn_index) != memory.logical_uid):
                raise StateUnavailable("Observation-order source identity/frontier is invalid")
            # A revision's commit_seq identifies its UPDATE, not its original
            # observation. Keep its existing evidence unchanged rather than
            # assigning it a fabricated original order. Clipped values also stay.
            if memory.version != 1 or not hit["content"].endswith(memory.text_content):
                output.append(dict(hit))
                continue
            session = digest([scope.key, memory.session_id])[:12]
            prefix = (f"[Source observation order: request {memory.commit_seq}, message {memory.turn_index + 1}; "
                      f"session {session}. Order observed, NOT event time or truth priority.]\n")
            if used + len(prefix) > self.max_added_chars:
                output.append(dict(hit))
                continue
            output.append(dict(hit, content=prefix + hit["content"]))
            used += len(prefix)
            positions.append(dict(id=hit["id"], source_version=memory.version,
                                  source_sha256=digest(memory.text_content),
                                  request=memory.commit_seq, message=memory.turn_index + 1,
                                  session=session))
        return output, dict(policy_hash=self.hash, activated=bool(positions), positions=positions,
                            added_chars=used, ranking_preserved=True, source_text_preserved=True,
                            temporal_validity_inferred=False)
