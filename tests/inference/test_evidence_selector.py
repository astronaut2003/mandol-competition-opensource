import json

import pytest

from mandol_competition.inference.evidence_selector import EvidenceSelector
from mandol_competition.inference.errors import InferenceUnavailable


class Client:
    def __init__(self, content='{"indices":[1,0]}', model='gpt-4o-mini-2024-07-18'):
        self.content, self.model = content, model
        self.calls = []
    def complete(self, messages, **options):
        self.calls.append((messages, options))
        return {"model": self.model, "choices": [{"finish_reason": "stop",
            "message": {"content": self.content}}], "usage": {"total_tokens": 10}}
    def close(self):
        pass


def select(client, **extra):
    return EvidenceSelector('unused', client=client).select(query='question', options=['A','B'],
        candidates=[{'text':'source zero'}, {'text':'source one'}], limit=2, **extra)


def test_only_existing_candidates_can_be_ranked_and_no_answer_channel():
    client = Client()
    result = select(client)
    assert result.indices == (1, 0) and result.status == 'selected'
    messages, kwargs = client.calls[0]
    assert 'gold' not in str(messages) and kwargs['temperature'] == 0
    assert kwargs['response_format']['type'] == 'json_object'


@pytest.mark.parametrize('content', ['{"indices":[2]}', '{"indices":[1,1]}',
    '{"indices":[true]}', '{"indices":[-1]}', '{"indices":[0],"answer":"bad"}',
    'answer', '{"indices":"0"}', '{"indices":[0,1,0]}'])
def test_invalid_selection_keeps_baseline(content):
    result = select(Client(content))
    assert result.indices == (0,1) and result.status == 'fallback'


def test_wrong_model_or_unknown_timeout_never_improves_score_silently():
    assert select(Client(model='gpt-4.1')).status == 'fallback'
    class Timeout(Client):
        def complete(self, messages, **kwargs):
            self.calls.append(messages)
            raise InferenceUnavailable('timeout', unknown_outcome=True)
    client = Timeout()
    assert select(client).status == 'fallback' and len(client.calls) == 1


def test_visual_candidates_use_image_blocks_not_base64_as_text():
    client = Client()
    uri = 'data:image/png;base64,AAAA'
    EvidenceSelector('unused', client=client).select(query='what is here?', options=[],
        candidates=[{'text':'description', 'images':[uri]}], limit=1, visual=True,
        query_images=[uri])
    blocks = client.calls[0][0][1]['content']
    assert len([b for b in blocks if b['type']=='image_url']) == 2
    assert all('AAAA' not in b['text'] for b in blocks if b['type']=='text')


def test_empty_evidence_supports_abstention_without_call():
    client = Client('{"indices":[]}')
    assert select(client).indices == ()
    result = EvidenceSelector('unused',client=client).select(query='q',options=[],candidates=[],limit=0)
    assert result.status=='empty' and len(client.calls)==1


def test_common_mini_client_rejects_silent_provider_model_substitution():
    from mandol_competition.inference.vlm import HTTPVLMClient
    from mandol_competition.inference.errors import InferenceIdentityError
    class Wire:
        def request(self, *args, **kwargs):
            return {'model':'gpt-4.1','choices':[{'message':{'content':'{}'}}]}
    client=HTTPVLMClient('unused','gpt-4o-mini',transport=Wire())
    with pytest.raises(InferenceIdentityError):
        client.complete([],request_id=None)
