"""Frozen source-aware output policy for the packaged Competition runtime.

The source binding admits one verified import-only relocation. The frozen
quality capsule stays byte-identical; any subsequent renderer edit fails closed.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from mandol_competition.textual.core import context_render
from mandol_substrate.state.models import ProfileMismatch, digest


def renderer_hash():
    from mandol_substrate.provenance import qualified_source_hash
    return qualified_source_hash(Path(context_render.__file__).read_bytes(),
        original='eb7feb547069d4091c784614b79b85ce663388200de8b400ef17ad48fc812512',
        promoted='97cc802ad570f2255bcbd542f34ce246f3d8f25e51c11a994f18f28ed4571bbb')


@dataclass(frozen=True)
class ContextPolicy:
    storage_profile_hash: str
    renderer_sha256: str
    mode: str = "raw"
    selected_k: int = 10
    schema: str = "mandol-context-policy-v1"

    def __post_init__(self):
        if self.schema != "mandol-context-policy-v1" or self.mode not in {
            "raw",
            "source_mention",
            "source_mention_chronological",
        }:
            raise ValueError("Unknown explicit context policy")
        for value in (self.storage_profile_hash, self.renderer_sha256):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Context policy requires explicit hash bindings")
        if type(self.selected_k) is not int or not 1 <= self.selected_k <= 100:
            raise ValueError("Invalid selected context window")

    @property
    def hash(self):
        return digest(asdict(self))

    def validate(self, physical_profile):
        if self.storage_profile_hash != physical_profile.hash:
            raise ProfileMismatch("Context policy belongs to another physical profile")
        if self.renderer_sha256 != renderer_hash():
            raise ProfileMismatch("Context renderer changed since policy qualification")

    def render(self, scope, hits, canonical):
        # Caller must retain the very same publication lease used for retrieval
        # and reranking. No post-Search second hydration or network work here.
        result, _audit = context_render.render_context_items(
            scope,
            hits,
            canonical,
            mode=self.mode,
            final_k=self.selected_k,
        )
        return result

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))
