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


def test_explicit_quota_rejection_retries_only_rejected_image_without_degraded_success(backend):
    from mandol_competition.inference.errors import InferenceRateLimited
    class Limited(Vision):
        limited=True
        def extract(self, **inputs):
            if self.limited and self.calls==1:
                self.calls+=1
                raise InferenceRateLimited(retry_after=60)
            return super().extract(**inputs)
    vision=Limited();add,_=services(backend,vision)
    body=request(content=[image_part(),image_part('blue')])
    with pytest.raises(InferenceRateLimited):add.handle(body)
    assert vision.calls==2
    assert list(backend.state.iter_memories(backend.scope('alice')))==[]
    vision.limited=False
    assert add.handle(body)['success'] and vision.calls==3
    assert add.handle(body)['success'] and vision.calls==3
    memory=next(backend.state.iter_memories(backend.scope('alice')))[0]
    assert not memory.metadata.get('vlm_degraded', False)


def test_expired_before_vlm_dispatch_does_not_leave_an_unknown_or_degraded_image(backend):
    from mandol_competition.inference.errors import InferenceDeadlineExceeded
    from mandol_competition.inference.http import deadline_budget
    vision=Vision();add,_=services(backend,vision)
    body=request(content=[image_part()])
    with deadline_budget(0):
        with pytest.raises(InferenceDeadlineExceeded):add.handle(body)
    assert vision.calls==0
    assert list(backend.state.iter_memories(backend.scope('alice')))==[]
    assert add.handle(body)['success'] and vision.calls==1
    assert add.handle(body)['success'] and vision.calls==1
    memory=next(backend.state.iter_memories(backend.scope('alice')))[0]
    assert not memory.metadata.get('vlm_degraded', False)


@pytest.mark.parametrize('text', ['Literal syntax data:image/',
    'Source includes ![example](data:image/png;base64,AAAA) and must stay unchanged.'])
def test_text_only_data_uri_survives_preparation_retry_and_exactly_once_publication(backend, text):
    vision = Vision(); add, search = services(backend, vision)
    body = request(content=text)
    def interrupt(point):
        if point == 'before_T1':
            raise TimeoutError('artificial pre-publication interruption')
    backend.coordinator.fault = interrupt
    with pytest.raises(TimeoutError):
        add.handle(body)
    assert vision.calls == 0
    backend.coordinator.fault = lambda _: None
    assert add.handle(body)['success'] and add.handle(body)['success']
    memories = list(backend.state.iter_memories(backend.scope('alice')))
    assert len(memories) == 1 and memories[0][0].raw_data['original_content'] == text
    assert search.handle(query())['data'][0]['content'][0]['text'] == '[Dialogue] ' + text
