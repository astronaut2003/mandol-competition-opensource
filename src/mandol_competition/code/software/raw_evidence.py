"""Opt-in, exact-source chunk retrieval and diff-hunk excerpt selection.

No model generation, data discovery or eligibility changes. Callers supply only
already authorized episodes/sources. Existing runtime defaults are untouched.
"""
from dataclasses import dataclass
import hashlib
import math
import re

from .anchors import CodeAnchorExtractor
from .episodes import source_digest
from .experience_retrieval import EpisodeRetriever, EpisodeRetrievalConfig, EvidencePacker


@dataclass(frozen=True)
class SourceSpan:
    source_uid: str
    source_hash: str
    text_sha256: str
    start: int
    end: int
    kind: str

    @property
    def uid(self):
        value = f'{self.source_uid}:{self.source_hash}:{self.start}:{self.end}'
        return 'raw_chunk:' + hashlib.sha256(value.encode()).hexdigest()

    def text(self, source):
        text = str(source.raw_data.get('text_content', ''))
        if (source.uid != self.source_uid or source_digest(source) != self.source_hash
                or hashlib.sha256(text.encode()).hexdigest() != self.text_sha256
                or not 0 <= self.start <= self.end <= len(text)):
            raise ValueError('Source span provenance changed')
        return text[self.start:self.end]


def diff_hunks(text):
    """Character ranges in original unified diffs; no decoding or rewriting."""
    files = list(re.finditer(r'(?m)^diff --git [^\n]*(?:\n|$)', text))
    output = []
    for i, file in enumerate(files):
        stop = files[i+1].start() if i+1 < len(files) else len(text)
        hunks = list(re.finditer(r'(?m)^@@[^\n]*@@[^\n]*(?:\n|$)', text[file.end():stop]))
        for j, hunk in enumerate(hunks):
            start = file.end() + hunk.start()
            end = file.end() + hunks[j+1].start() if j+1 < len(hunks) else stop
            output.append((start, end, file.group(0).rstrip()))
    return output


def strong_query_anchors(query):
    kinds = {'path', 'symbol', 'error', 'test', 'config', 'constant', 'version'}
    return sorted({a.raw.split(':', 1)[0] if a.kind == 'error' else a.raw
                   for a in CodeAnchorExtractor().extract(query) if a.kind in kinds})


def exact_matches(text, anchors):
    # Preserve original offsets, including Unicode; casefolded-string offsets
    # are not interchangeable with original source character offsets.
    return [(m.start(), m.end(), anchor) for anchor in anchors
            for m in re.finditer(r'(?<![\w])' + re.escape(anchor) + r'(?![\w])', text, flags=re.IGNORECASE)]


def matched_changed_lines(text, query):
    anchors = strong_query_anchors(query)
    rows = []
    for start, end, path in diff_hunks(text):
        position = start
        for line in text[start:end].splitlines(keepends=True):
            if line.startswith(('+', '-')) and not line.startswith(('+++', '---')):
                matches = exact_matches(line[1:], anchors)
                if matches:
                    rows.append((position, position + len(line), start, end, path,
                                 sorted({a for _, _, a in matches})))
            position += len(line)
    return rows


def source_chunks(source, tokenizer, *, trigger_tokens=2048, chunk_tokens=1024, overlap_tokens=64):
    if not 0 <= overlap_tokens < chunk_tokens < trigger_tokens:
        raise ValueError('Invalid fixed token windows')
    text = str(source.raw_data.get('text_content', ''))
    def span(start, end, kind):
        return SourceSpan(source.uid, source_digest(source), hashlib.sha256(text.encode()).hexdigest(), start, end, kind)
    tokens = tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']
    kind = str(source.raw_data.get('artifact_kind', 'raw'))
    if len(tokens) <= trigger_tokens:
        return [span(0, len(text), kind)]
    boundaries = sorted({0, len(text)} | {i for a,b,_ in diff_hunks(text) for i in (a,b)}
                        | {m.start() for m in re.finditer(r'(?m)^diff --git ', text)})
    result = []
    for start, end in zip(boundaries, boundaries[1:]):
        pos = start
        while pos < end:
            offsets = tokenizer(text[pos:end], add_special_tokens=False, truncation=False,
                                return_offsets_mapping=True)['offset_mapping']
            # Reserve tokenizer specials and boundary re-tokenization allowance.
            window = min(chunk_tokens - 8, len(offsets))
            cut = pos + offsets[window-1][1]
            while len(tokenizer(text[pos:cut], add_special_tokens=True)['input_ids']) > chunk_tokens:
                window -= 1
                if window <= 0:
                    raise ValueError('Tokenizer cannot fit source character')
                cut = pos + offsets[window-1][1]
            if cut <= pos:
                raise ValueError('Tokenizer failed to advance')
            result.append(span(pos, cut, 'diff_hunk' if any(a <= pos < b for a,b,_ in diff_hunks(text)) else kind))
            if cut >= end:
                break
            overlap_start = pos + offsets[max(0, window-overlap_tokens)][0]
            pos = overlap_start if overlap_start > pos else cut
    return result


class SourceBackedChunkRetriever:
    """Dense chunk hits collapse to best score per raw source then episode.

    Duplicate chunk count never adds relevance. Evidence UIDs remain original
    source UIDs so retrieval-only A/B leaves the current packer unchanged.
    """
    def __init__(self, episodes, sources, *, user_id, encoder, tokenizer):
        scoped = [ep for ep in episodes if ep.user_id == user_id]
        allowed = set()
        for ep in scoped:
            ep.validate_provenance(sources)
            allowed.update(ep.source_uids)
        self.spans = {s.uid:s for uid in sorted(allowed) for s in source_chunks(sources[uid], tokenizer)}
        self.index = encoder.index({uid:s.text(sources[s.source_uid]) for uid,s in self.spans.items()})
        self.last_chunk_hits = []
        owner = self
        class Channel:
            def search(self, query, *, top_k):
                matches = owner.index.search(query, top_k=len(owner.spans))
                best = {}
                owner.last_chunk_hits = []
                for uid, score in matches:
                    if uid not in owner.spans or not math.isfinite(score):
                        raise ValueError('Invalid chunk hit')
                    s = owner.spans[uid]
                    owner.last_chunk_hits.append((uid, score))
                    best[s.source_uid] = max(best.get(s.source_uid, -math.inf), score)
                return sorted(best.items(), key=lambda pair:(-pair[1],pair[0]))[:top_k]
        self.retriever = EpisodeRetriever(scoped, sources, user_id=user_id, raw_channel=Channel(),
                                         config=EpisodeRetrievalConfig(mode='raw'))

    def search(self, query, top_k=8):
        return self.retriever.search(query, top_k=top_k)


EXCERPT = re.compile(r'\nEvidence (?P<uid>\S+) sha256:(?P<hash>[0-9a-f]{64})\n(?:Evidence scope: [^\n]*\n)?'
                     r'\[raw excerpt (?P<start>\d+):(?P<end>\d+)/(?P<total>\d+)\]\n')


class QueryAwareHunkPacker(EvidencePacker):
    """Only excerpt selection changes; ranking, source order and cards do not."""
    def pack(self, hits, sources, *, query, top_k):
        cards = super().pack(hits, sources, query=query, top_k=top_k)
        result = []
        for card in cards:
            content = card['content']
            replacements = []
            for match in EXCERPT.finditer(content):
                uid = match['uid']; source = sources[uid]
                text = str(source.raw_data.get('text_content',''))
                if source_digest(source) != match['hash']:
                    raise ValueError('Packed evidence source changed')
                old_start, old_end = int(match['start']), int(match['end'])
                length = old_end - old_start
                lines = matched_changed_lines(text, query)
                if not lines:
                    continue
                hunks = {}
                for row in lines:
                    hunks.setdefault((row[2],row[3]), []).append(row)
                (hs,he), selected = min(hunks.items(), key=lambda pair:(-len({a for r in pair[1] for a in r[5]}), pair[0][0]))
                start = max(hs, selected[0][0] - 100)
                end = min(he, start + length)
                # Same serialized character budget even when offsets gain digits.
                prefix = match.group(0).split('[raw excerpt ',1)[0]
                replacement = prefix + f'[raw excerpt {start}:{end}/{len(text)}]\n' + text[start:end]
                old_size = len(match.group(0)) + length
                while len(replacement) > old_size and end > start:
                    end -= len(replacement) - old_size
                    replacement = prefix + f'[raw excerpt {start}:{end}/{len(text)}]\n' + text[start:end]
                if end <= start or not any(start <= r[0] < end for r in selected):
                    continue
                replacements.append((match.start(),match.end()+length,replacement))
            for start,end,replacement in reversed(replacements):
                content = content[:start] + replacement + content[end:]
            if len(content) > len(card['content']):
                raise ValueError('Hunk selection exceeded original budget')
            result.append({**card,'content':content})
        return result
