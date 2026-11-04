"""Evaluation-only AML prompt and wire-content preparation for the next Q0.

This module never imports the official API configuration or calls a model. It
reads hash-pinned public prompt literals and preserves the public renderer's
semantics. It does not modify the already frozen Q-1M QA experiment.
"""

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .paired_quality import file_sha256

OFFICIAL_REVISION = '1b8142bfe0f20f1c5218d6b554aa0012de34e504'
OFFICIAL_SOURCE_HASHES = {
    'locomo-refined': '31a49fc09e381bc117b59063c7a4cea543a672b23e1264aedb03290239671c1a',
    'longmemeval-s': 'c7f17c363bd0803cfeedd2b793a0cb0a74ab119a70eeced2c7a835b2b2aecb46',
}


def memory_text(value):
    """Public AML renderer behavior, including literal list/dict handling."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(memory_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value or '')


@dataclass(frozen=True)
class OfficialPromptContract:
    dataset: str
    source_sha256: str
    answer_template: str
    accuracy_template: str

    @property
    def manifest(self):
        return {
            'source_revision': OFFICIAL_REVISION, 'source_sha256': self.source_sha256,
            'dataset': self.dataset,
            'answer_prompt_sha256': hashlib.sha256(self.answer_template.encode()).hexdigest(),
            'judge_prompt_sha256': hashlib.sha256(self.accuracy_template.encode()).hexdigest(),
            'semantics': 'pinned-public-aml-renderer-v1',
            'official_cycle2_bundle': 'UNKNOWN',
        }

    def answer(self, question, context, *, speaker_1_name='speaker 1', speaker_2_name='speaker 2'):
        values = {
            'speaker_1_name': speaker_1_name, 'speaker_1_memories': context,
            'speaker_2_name': speaker_2_name, 'speaker_2_memories': '', 'question': question,
        }
        return re.sub(
            r'\{\{(speaker_1_name|speaker_1_memories|speaker_2_name|speaker_2_memories|question)\}\}',
            lambda match: memory_text(values[match.group(1)]), self.answer_template,
        )

    def judge(self, question, gold_answer, generated_answer):
        values = {'question': question, 'gold_answer': gold_answer, 'generated_answer': generated_answer}
        return re.sub(
            r'\{(question|gold_answer|generated_answer)\}',
            lambda match: memory_text(values[match.group(1)]), self.accuracy_template,
        )

    @staticmethod
    def parse_judge_label(response):
        match = re.search(r'\{.*?\}', response, re.DOTALL)
        if not match:
            raise ValueError('judge response does not contain a JSON object')
        payload = json.loads(match.group(0))
        label = str(payload.get('label', '')).upper()
        if label not in {'CORRECT', 'WRONG'}:
            raise ValueError('judge label must be CORRECT or WRONG')
        return label


def load_official_contract(path, dataset):
    """Literal extraction only; never execute downloaded code or import keys."""
    if dataset not in OFFICIAL_SOURCE_HASHES:
        raise ValueError('Unsupported public prompt contract')
    digest = file_sha256(path)
    if digest != OFFICIAL_SOURCE_HASHES[dataset]:
        raise ValueError('Official prompt source checksum mismatch')
    names = {'OPEN_ENDED_ANSWER_TEMPLATE', 'ACCURACY_PROMPT'}
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in names:
                if name in found:
                    raise ValueError('Duplicate official prompt declaration')
                found[name] = ast.literal_eval(node.value)
    if set(found) != names or any(not isinstance(v, str) or not v for v in found.values()):
        raise ValueError('Missing literal official prompt')
    return OfficialPromptContract(dataset, digest, found['OPEN_ENDED_ANSWER_TEMPLATE'], found['ACCURACY_PROMPT'])


def validate_add_echo(payload, response):
    """An Add response is not success merely because its HTTP status is200."""
    if not isinstance(response, dict) or response.get('success') is not True:
        raise ValueError('Add did not acknowledge success')
    for key in ('request_id', 'user_id', 'session_id'):
        if not isinstance(response.get(key), str) or response[key] != payload[key]:
            raise ValueError('Add exact identifier echo mismatch: ' + key)


def wire_context(response, *, requested_top_k, context_k, max_chars, allowed_uids=None):
    """Pack only actual API content in returned order, recording partial pieces.

    Neither score sorting, metadata hydration, source snippets, gold labels nor
    ingestion timestamps may add text. A character budget can expose only part
    of a returned record; record that fact instead of claiming full evidence.
    allowed_uids is an optional evaluator-side scope/provenance guard.
    """
    if any(type(v) is not int or v <= 0 for v in (requested_top_k, context_k, max_chars)):
        raise ValueError('Invalid context budget')
    if not isinstance(response, dict) or not isinstance(response.get('data'), list):
        raise TypeError('Malformed Search response')
    hits = response['data']
    if len(hits) > requested_top_k:
        raise ValueError('Search exceeded requested top_k')
    ids = []
    for hit in hits:
        if not isinstance(hit, dict):
            raise TypeError('Search hit must be an object')
        for key in ('id', 'content'):
            if not isinstance(hit.get(key), str) or not hit[key].strip():
                raise ValueError('Search hit has empty/nontext ' + key)
        if hit['id'] in ids:
            raise ValueError('Duplicate Search logical UID')
        ids.append(hit['id'])
        if allowed_uids is not None and hit['id'] not in allowed_uids:
            raise ValueError('Search UID lacks same-scope provenance')
        score = hit.get('score')
        if score is not None and (type(score) not in (int, float) or not math.isfinite(score)):
            raise ValueError('Search score must be finite')
        created = hit.get('created_at')
        if created is not None and not isinstance(created, str):
            raise ValueError('Invalid created_at')
    context, pieces = '', []
    for rank, hit in enumerate(hits[:context_k], 1):
        separator = '\n' if context else ''
        available = max_chars - len(context) - len(separator)
        if available <= 0:
            break
        content = hit['content'][:available]
        context += separator + content
        pieces.append({
            'id': hit['id'], 'api_rank': rank, 'content_chars': len(hit['content']),
            'visible_chars': len(content), 'fully_visible': content == hit['content'],
            'visible_content_sha256': hashlib.sha256(content.encode()).hexdigest(),
        })
    return {
        'context': context, 'context_chars': len(context),
        'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
        'returned_ids': ids, 'returned_chars': sum(len(hit['content']) for hit in hits),
        'context_ids': [piece['id'] for piece in pieces], 'pieces': pieces,
        'context_k': context_k, 'max_chars': max_chars,
        'policy': 'api-content-only-returned-order-newline-v1',
        'source_side_text_added': False,
    }
