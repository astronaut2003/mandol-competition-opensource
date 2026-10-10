"""Competition-aligned, maximum-safe extraction with unchanged citation checks."""

from .api_context import canonical_json
from .episode_distillation import EngineeringEpisodeDistiller, INSTRUCTION


def distillation_messages(prepared):
    body = canonical_json({k: v for k, v in prepared.items() if k != "instruction"})
    return [{"role": "system", "content": prepared["instruction"]}, {"role": "user", "content": body}]


class MaximumSafeEpisodeDistiller(EngineeringEpisodeDistiller):
    def __init__(self, context):
        self.context = context

    def prepare(self, episode, sources):
        episode.validate_provenance(sources)
        # Task and commit intent precede diffs, so a large diff cannot evict
        # them. Inputs must already have passed the historical PR scope gate.
        priorities = {"statement": 0, "historical_commit_message": 1,
                      "test_output": 2, "issue": 3, "pull_request": 3,
                      "historical_commit_diff": 4}
        ordered = sorted(enumerate(episode.source_uids),
            key=lambda item: (priorities.get(sources[item[1]].raw_data.get("artifact_kind"), 3), item[0]))
        selected = [uid for _, uid in ordered[:64]]
        evidence = []
        for uid in selected:
            source = sources[uid].raw_data
            scope = {key: source[key] for key in ("artifact_kind", "artifact_scope", "scope") if key in source}
            text = str(source.get("text_content", ""))
            evidence.append({"source_uid": uid, "source_sha256": episode.source_hashes[uid],
                "start": 0, "end": 0, "source_characters": len(text), "text": "", **scope})
        prepared = {"instruction": INSTRUCTION, "evidence": evidence,
                    "omitted_source_count": len(episode.source_uids) - len(selected)}
        if self.context.count_prompt(distillation_messages(prepared), []) > self.context.budget.max_input_tokens:
            raise ValueError("Historical scope metadata alone exceeds the safe budget")
        for entry in evidence:
            raw = str(sources[entry["source_uid"]].raw_data.get("text_content", ""))
            entry.update(text=raw, end=len(raw))
            if self.context.count_prompt(distillation_messages(prepared), []) <= self.context.budget.max_input_tokens:
                continue
            low, high, best = 0, len(raw), 0
            while low <= high:
                end = (low + high) // 2
                entry.update(text=raw[:end], end=end)
                if self.context.count_prompt(distillation_messages(prepared), []) <= self.context.budget.max_input_tokens:
                    best = end
                    low = end + 1
                else:
                    high = end - 1
            entry.update(text=raw[:best], end=best)
        return prepared
