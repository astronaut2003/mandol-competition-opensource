"""Query-only evidence packaging, independent of the Add derivation recipe."""
from dataclasses import asdict, dataclass

from mandol_substrate.state.models import digest


@dataclass(frozen=True)
class EvidencePackagingPolicy:
    version: str = "textual-provenance-families-v1"
    selected_k: int = 100
    max_context_chars: int = 60000

    def __post_init__(self):
        if (self.version != "textual-provenance-families-v1"
                or type(self.selected_k) is not int or not 1 <= self.selected_k <= 100
                or type(self.max_context_chars) is not int or not 1000 <= self.max_context_chars <= 100000):
            raise ValueError("Invalid explicit evidence-packaging policy")

    @property
    def hash(self):
        return digest(asdict(self))
