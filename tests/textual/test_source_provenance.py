import hashlib
from pathlib import Path
from mandol_substrate.provenance import qualified_source_hash
from mandol_competition.textual.core import context_render
from mandol_competition.textual.core.context_policy import renderer_hash


def test_renderer_is_exact_import_only_relocation_and_changes_fail_closed():
    original = (Path(__file__).resolve().parents[1] / "oracles/context_render.py").read_bytes()
    assert hashlib.sha256(original).hexdigest() == "eb7feb547069d4091c784614b79b85ce663388200de8b400ef17ad48fc812512"
    promoted = Path(context_render.__file__).read_bytes()
    assert original.replace(b'from mandol.storage.state.models import', b'from mandol_substrate.state.models import') == promoted
    old_hash, new_hash = hashlib.sha256(original).hexdigest(), hashlib.sha256(promoted).hexdigest()
    assert renderer_hash() == old_hash
    assert qualified_source_hash(promoted + b'\n# edit', original=old_hash, promoted=new_hash) != old_hash
