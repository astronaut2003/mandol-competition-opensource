"""CPU-only contracts for explicitly loaded Stage A reranker references."""

import hashlib
import math
from pathlib import Path


def file_sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def pinned_snapshot(model_name_or_path, model_name, revision, cache_dir=None):
    """Resolve cached identity without network access or model construction."""
    if str(model_name_or_path) == model_name:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(
            model_name, "config.json", revision=revision, cache_dir=cache_dir
        )
        if not isinstance(config, str):
            raise ValueError("pinned model snapshot is not available locally")
        path = Path(config).parent
    else:
        path = Path(model_name_or_path)
    if (
        path.name != revision
        or path.parent.name != "snapshots"
        or path.parent.parent.name != "models--" + model_name.replace("/", "--")
        or not path.is_dir()
    ):
        raise ValueError("reference path must identify the exact pinned model snapshot")
    return path


def verify_files(snapshot, manifest):
    for name, expected in manifest.items():
        path = Path(snapshot) / name
        if not path.is_file():
            raise ValueError(f"qualified reference file is missing: {name}")
        if (
            expected.get("bytes") is not None
            and path.stat().st_size != expected["bytes"]
        ):
            raise ValueError(f"qualified reference file size differs: {name}")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"qualified reference file digest differs: {name}")


def validate_settings(
    dtype, device, max_pair_tokens, model_limit, batch_size, max_batch_tokens
):
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("unsupported native reference dtype")
    if not isinstance(device, str) or not device:
        raise ValueError("native reference device must be explicit")
    if type(max_pair_tokens) is not int or not 1 <= max_pair_tokens <= model_limit:
        raise ValueError("native pair budget exceeds the model's configured context")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("native reference batch size must be in [1, 128]")
    if type(max_batch_tokens) is not int or max_batch_tokens < max_pair_tokens:
        raise ValueError("native padded-token budget must fit one maximum pair")


def bounded_batches(lengths, batch_size, max_batch_tokens):
    """Keep candidate order while bounding dense padded-token allocation."""
    batches, current, longest = [], [], 0
    for index, length in enumerate(lengths):
        if type(length) is not int or not 0 < length <= max_batch_tokens:
            raise ValueError("invalid native reference pair length")
        next_longest = max(longest, length)
        if current and (
            len(current) == batch_size
            or next_longest * (len(current) + 1) > max_batch_tokens
        ):
            batches.append(current)
            current, longest = [], 0
        current.append(index)
        longest = max(longest, length)
    if current:
        batches.append(current)
    return batches


def finite_raw_scores(values, expected_count):
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError("native reference returned the wrong score count")
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise ValueError("native reference score is not numeric")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError("native reference score cannot be represented") from exc
        if not math.isfinite(value):
            raise ValueError("native reference returned a non-finite score")
        result.append(value)
    return result
