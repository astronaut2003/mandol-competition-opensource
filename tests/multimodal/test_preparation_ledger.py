import pytest
from mandol_substrate.state import Conflict
from mandol_competition.multimodal.vlm.base import Extraction
from .backend_support import request, services, query
from .test_sqlite_qdrant import image_part


class Vision:
    model_name = 'counting'
    def __init__(self):
        self.calls = 0
    def extract(self, **inputs):
        self.calls += 1
        return Extraction(description='red backpack', model=self.model_name, prompt_version='v1', schema_version='1')


def test_vlm_preparation_replayed_before_T1_without_model_calls(backend):
    vision = Vision()
    add, search = services(backend, vision)
    parts = [image_part(), {'type': 'text', 'text': 'between'}, image_part('blue'), {'type': 'text', 'text': 'after'}]
    body = request(content=parts)
    def fault(point):
        if point == 'before_T1':
            raise TimeoutError('unacknowledged preparation')
    backend.coordinator.fault = fault
    with pytest.raises(TimeoutError):
        add.handle(body)
    assert vision.calls == 2
    with pytest.raises(Conflict):
        add.handle(request(content='changed'))
    backend.coordinator.fault = lambda _: None
    assert add.handle(body)['success']
    assert vision.calls == 2
    assert add.handle(body)['success'] and vision.calls == 2
    result = search.handle(query())['data'][0]['content']
    assert result[:4] == parts
    assert 'red backpack' in result[-1]['text']


def test_unknown_vlm_preparation_uses_degraded_policy_without_repeat(backend):
    class Interrupted(Vision):
        def extract(self, **inputs):
            self.calls += 1
            raise SystemExit('simulate process loss after dispatch')
    vision = Interrupted(); add, _ = services(backend, vision)
    body = request(content=[image_part()])
    with pytest.raises(SystemExit):
        add.handle(body)
    assert add.handle(body)['success']
    assert vision.calls == 1
    memory = next(backend.state.iter_memories(backend.scope('alice')))[0]
    assert memory.metadata['vlm_degraded'] is True
