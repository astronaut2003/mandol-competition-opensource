"""Load a pinned tokenizer from its local snapshot, without hub metadata probes."""

from pathlib import Path


def load_snapshot_tokenizer(model, revision, *, cache_dir=None):
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Immutable tokenizer revision required")
    snapshot = Path(snapshot_download(
        model, revision=revision, cache_dir=cache_dir, local_files_only=True,
    ))
    if snapshot.name != revision or snapshot.parent.name != "snapshots":
        raise ValueError("Tokenizer snapshot revision differs")
    # Transformers4.57's generic remote-name Mistral regex detection may contact
    # model_info even with local_files_only. A local path reads the same config
    # and tokenizer bytes; no monkeypatch or alternate tokenizer is involved.
    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
