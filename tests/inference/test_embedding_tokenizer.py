"""Compatibility counts captured from owned text, not evaluation material."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from mandol_competition.inference import embedding_tokenizer as tokens
from mandol_competition.inference.embedding_query_views import plan_query
from mandol_competition.inference.errors import InferenceSchemaError, InferenceUnavailable


COUNTS=[
    ('hello world',2),
    ('Find the artificial note about the blue notebook on the kitchen shelf.',13),
    ('请查找人工测试笔记中蓝色笔记本放在哪个架子上。',13),
    ('青いノートを置いた棚に関する人工テストの記録を探してください。',18),
    ('파란색 공책을 놓은 선반에 관한 인공 테스트 기록을 찾아 주세요.',26),
    ('ابحث عن الملاحظة التجريبية حول الدفتر الأزرق على الرف.',20),
    ('नीली नोटबुक वाले कृत्रिम परीक्षण नोट को खोजें।',43),
    ('Найдите искусственную тестовую заметку о синей записной книжке.',20),
    ('Retrouvez la note artificielle sur le cahier bleu près de la fenêtre.',22),
    ('Synthetic Unicode: café cafe\u0301 naïve nai\u0308ve Å Å A\u030a.',14),
    ('Artificial location labels 🧑🏽‍💻 👨‍👩‍👧‍👦 🇨🇳 🏳️‍🌈 📘.',36),
    ('Artificial literal markers <|endoftext|> <|im_start|>user\n<|im_end|>.',12),
    ('{"artificial":true,"book":{"color":"blue","shelf":3},"values":[0,1.25,-2]}',27),
    ('def artificial_lookup(items, key):\n    return [x for x in items if x["key"] == key]\n',23),
    ('人工测试：函数 artificial_lookup(user_id: str) -> list[dict] 必须按用户隔离。',23),
    ('  Artificial\ttext\r\nwith multiple     spaces.\n\nEnd.  ',12),
    ('artificial_6d12e359_8c2e_42f3_a38e_deadbeef1234_0123456789abcdef',44),
    ('Artificial memory alpha beta gamma. '*700,4202),
    ('这是人工合成的检索证据，蓝色笔记本位于第三层架子。'*250,3500),
    (('def synthetic_lookup(user_id, note_id):\n'
      '    return synthetic_store.get((user_id, note_id))\n')*300,6300),
    ('Artificial memory alpha beta gamma. '*1000,6002),
    ('synthetic token budget check. '*1558,7792),
]


@pytest.mark.parametrize('text,expected',COUNTS,ids=[f'owned-case-{i}' for i in range(len(COUNTS))])
def test_pinned_counts_match_owned_provider_usage_minus_one(text,expected):
    assert tokens.count_tokens(text)==expected


def test_corrupt_asset_never_silently_uses_another_tokenizer(monkeypatch):
    monkeypatch.setattr(tokens,'_instance',None)
    monkeypatch.setattr(tokens,'ASSET_SHA256','0'*64)
    with pytest.raises(InferenceSchemaError,match='integrity'):
        tokens.count_tokens('owned text')


def test_missing_asset_fails_explicitly(monkeypatch):
    monkeypatch.setattr(tokens,'_instance',None)
    def missing(*_):
        raise OSError('Artificial unavailable asset')
    monkeypatch.setattr(tokens,'files',missing)
    with pytest.raises(InferenceUnavailable,match='tokenizer_unavailable'):
        tokens.count_tokens('owned text')


def test_sixteen_simultaneous_first_calls_load_one_shared_local_instance(monkeypatch):
    import tokenizers
    original=tokenizers.Tokenizer
    calls=[]
    def load(data):
        calls.append(len(data))
        return original.from_buffer(data)
    monkeypatch.setattr(tokens,'_instance',None)
    monkeypatch.setattr(tokenizers,'Tokenizer',SimpleNamespace(from_buffer=load))
    with ThreadPoolExecutor(max_workers=16) as workers:
        counts=list(workers.map(tokens.count_tokens,['hello world']*16))
    assert counts==[2]*16 and len(calls)==1


@pytest.mark.parametrize('seed',range(6))
def test_random_unicode_boundaries_are_lossless_and_recounted(seed):
    rng=random.Random(seed)
    alphabet=['a','b','中','文','你','🙂','\u0301','\r','\n','\t',' ',
        '👨‍👩‍👧‍👦','<|im_start|>','नमस्ते','بارد','é','e\u0301']
    text='owned\n'+''.join(rng.choice(alphabet) for _ in range(11000))+'\nTAIL'
    plan=plan_query(text)
    assert ''.join(plan.parts)==text
    assert all(tokens.count_tokens(part)<=8000 for part in plan.parts)
    assert all(part.encode('utf-8').decode('utf-8')==part for part in plan.parts)
