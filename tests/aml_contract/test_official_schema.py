"""Current official wire examples, independently of historical adapter oracles."""
import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from mandol_competition.multimodal.protocol import SearchRequest as MMSearchRequest

CONTRACT = json.loads(Path(__file__).with_name('contract.json').read_text())


def schemas(track):
    module = ('mandol_competition.multimodal.protocol' if track == 'multimodal'
              else 'mandol_competition.' + track + '.api.schemas')
    return importlib.import_module(module)


def add_body():
    return {'request_id': 'contract-request', 'user_id': 'contract-user', 'session_id': 'session-two',
            'messages': [{'role': 'user', 'content': 'First source message.'},
                         {'role': 'assistant', 'content': 'Second source message.', 'timestamp': 1791302400000}]}


@pytest.mark.parametrize('track', CONTRACT['tracks'])
def test_minimal_add_echo_order_and_optional_timestamp(track):
    body = add_body()
    request = schemas(track).AddRequest.model_validate(body)
    assert [m.content for m in request.messages] == [m['content'] for m in body['messages']]
    assert request.messages[0].timestamp is None
    assert request.messages[1].timestamp == 1791302400000
    response = schemas(track).AddResponse(success=True, **{k: body[k] for k in CONTRACT['add_success']['echo']})
    assert response.model_dump() == {'success': True, **{k: body[k] for k in CONTRACT['add_success']['echo']}}


@pytest.mark.parametrize('track', CONTRACT['tracks'])
@pytest.mark.parametrize('missing', CONTRACT['add_required'])
def test_missing_add_field_is_invalid(track, missing):
    body = add_body()
    del body[missing]
    with pytest.raises(ValidationError):
        schemas(track).AddRequest.model_validate(body)


@pytest.mark.parametrize('track', CONTRACT['tracks'])
@pytest.mark.parametrize('timestamp', [True, 1791302400000.5, '1791302400000'])
def test_timestamp_cannot_be_bool_float_or_string(track, timestamp):
    body = add_body()
    body['messages'][0]['timestamp'] = timestamp
    with pytest.raises(ValidationError):
        schemas(track).AddRequest.model_validate(body)


@pytest.mark.parametrize('track', CONTRACT['tracks'])
def test_search_string_options_and_top100_require_no_extension_fields(track):
    request = schemas(track).SearchRequest.model_validate({'query': 'First source?',
        'options': ['A', 'B'], 'user_id': 'contract-user', 'top_k': CONTRACT['formal_top_k']})
    assert request.options == ['A', 'B'] and request.top_k == 100
    assert schemas(track).SearchResponse(data=[]).model_dump() == {'data': []}


@pytest.mark.parametrize('track', CONTRACT['tracks'])
@pytest.mark.parametrize('missing', CONTRACT['search_required'])
def test_missing_search_field_is_invalid(track, missing):
    body = {'query': 'First source?', 'user_id': 'contract-user', 'top_k': 100}
    del body[missing]
    with pytest.raises(ValidationError):
        schemas(track).SearchRequest.model_validate(body)


def test_mm_query_string_uses_existing_representation_and_preserves_order():
    plain = MMSearchRequest(query='Question\nwith original bytes.', user_id='u', top_k=100)
    parts = MMSearchRequest(query=[{'type': 'text', 'text': plain.query}], user_id='u', top_k=100)
    assert plain.parts() == parts.parts()
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}
    request = MMSearchRequest(query=[image, {'type': 'text', 'text': 'after'}], user_id='u', top_k=100)
    assert [p.type for p in request.parts()] == ['image_url', 'text']
