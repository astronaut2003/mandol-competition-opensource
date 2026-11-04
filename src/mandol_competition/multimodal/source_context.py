"""Read-only, bounded source-neighbor hints; never inferred relationships.

The baseline pool is untouched. Supplemental originals must be reviewed as
useful before delivery, retain their own provenance, and consume no top_k slot.
No database/index changes or additional model/embedding calls are made here.
"""
import re

from mandol_substrate.representation import MemoryUnit
from .evidence import source_text
from .evidence_review import ContextSource

MAX_CONTEXT_SOURCES = 8
MAX_CONTEXT_TEXT_BYTES = 32 * 1024
MAX_SOURCE_BYTES = 12 * 1024
MAX_SEEDS = 3
MAX_ANCHORS = 2
MAX_ANCHOR_HITS = 12
RELATIONAL = re.compile(
    r'\b(before|after|during|earlier|later|previous|previously|latest|now|then|changed|'
    r'correction|update|same|that|this|those|these|he|she|they|his|her|their|it|its)\b'
    r'|之前|之后|以前|后来|上次|这次|现在|最新|更正|更新|同一个|那个|这个|他们|她|他|它', re.I)
GENERIC = {'person', 'people', 'man', 'woman', 'child', 'unknown', 'user', 'assistant',
           'image', 'photo', 'picture', 'object', 'building', 'car', 'dog', 'cat',
           'table', 'chair', 'shirt', 'park', 'city', 'someone', '人物', '男子', '女子',
           '男人', '女人', '用户', '未知', '照片', '图片', '建筑', '桌子', '椅子'}


def _contains(text, anchor):
    return re.search(r'(?<!\w)' + re.escape(anchor) + r'(?!\w)', text, re.I) is not None if anchor.isascii() else anchor.casefold() in text.casefold()


def _anchors(unit, query):
    names = []
    info = unit.raw_data.get('multimodal_info')
    images = info.get('images') if isinstance(info, dict) else None
    if not isinstance(images, list):
        return names
    for image in images:
        entities = image.get('entities') if isinstance(image, dict) else None
        if not isinstance(entities, list):
            continue
        for entity in entities:
            name = entity.get('name') if isinstance(entity, dict) else None
            if (isinstance(name, str) and 3 <= len(name.strip()) <= 80
                    and name.strip().casefold() not in GENERIC
                    and any(c.isalpha() for c in name)
                    and _contains(unit.raw_data.get('text_content', ''), name.strip())):
                if name.strip() not in names:
                    names.append(name.strip())
    return sorted(names, key=lambda name: (not _contains(query, name), names.index(name)))


def collect_context(backend, user_id, hits, query):
    if not hits or not RELATIONAL.search(query):
        return ()
    scope = backend.scope(user_id)
    baseline = {unit.uid for unit, _ in hits}
    selected, added, bytes_used = [], set(baseline), 0
    with backend.coordinator.search_view(scope) as view:
        seeds = backend.state.get_memories(scope, [unit.uid for unit, _ in hits[:MAX_SEEDS]])

        def eligible(memory):
            return (memory is not None and memory.active and memory.lifecycle != 'deleted'
                    and memory.commit_seq <= view.published_seq and 'aml' in memory.spaces
                    and memory.memory_type == 'observation' and memory.memory_level == 'L0_Observation'
                    and memory.metadata.get('user_id') == user_id)

        def admit(parent, memory, relation):
            nonlocal bytes_used
            if not eligible(memory) or memory.logical_uid in added or len(selected) >= MAX_CONTEXT_SOURCES:
                return
            unit = MemoryUnit(memory.logical_uid, memory.raw_data, memory.metadata)
            size = len((source_text(unit) + unit.raw_data.get('original_content', '') + '\n'
                        + unit.raw_data.get('text_content', '')).encode('utf-8')) + 512
            if size > MAX_SOURCE_BYTES or bytes_used + size > MAX_CONTEXT_TEXT_BYTES:
                return
            bytes_used += size
            added.add(unit.uid)
            selected.append(ContextSource(parent.uid, unit, relation))

        anchors = []
        for parent, _ in hits[:MAX_SEEDS]:
            seed = seeds.get(parent.uid)
            if not eligible(seed):
                continue
            if seed.session_id:
                rows = backend.state._rows(
                    'SELECT logical_uid FROM memories WHERE scope=? AND session_id=? '
                    "AND active=1 AND lifecycle<>'deleted' AND commit_seq<=? AND logical_uid<>? "
                    'ORDER BY CASE WHEN request_id=? THEN 0 ELSE 1 END, '
                    'abs(commit_seq-?),abs(turn_index-?),logical_uid LIMIT 2',
                    (scope.key, seed.session_id, view.published_seq, parent.uid,
                     seed.request_id, seed.commit_seq, seed.turn_index))
                neighbors = backend.state.get_memories(scope, [r['logical_uid'] for r in rows])
                for row in rows:
                    admit(parent, neighbors.get(row['logical_uid']), 'same-session-neighbor')
            for anchor in _anchors(parent, query):
                if anchor.casefold() not in {value.casefold() for _, value in anchors}:
                    anchors.append((parent, anchor))
        for parent, anchor in anchors[:MAX_ANCHORS]:
            results = backend.state.lexical.search(scope, anchor, limit=MAX_ANCHOR_HITS,
                                                  published_seq=view.published_seq)
            candidates = backend.state.get_memories(scope, [r.logical_uid for r in results])
            for result in results:
                memory = candidates.get(result.logical_uid)
                if memory is not None and _contains(memory.text_content, anchor):
                    admit(parent, memory, 'shared-explicit-text-anchor')
    return tuple(selected)
