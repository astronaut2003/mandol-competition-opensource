"""Public startup script over real loopback HTTP and owned synthetic storage."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import io
import json
import multiprocessing
import os
from pathlib import Path
import runpy
import socket
import sys
import threading
import time
import uuid

import httpx
from PIL import Image

from tests.competition.test_native_frozen_process import native_fixture
from tests.support.scratch_qdrant import formal_stack_qdrant

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def visual_fixture():
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            schema = body.get('response_format', {}).get('json_schema', {}).get('name')
            calls.append(schema)
            if schema == 'visual_memory_ocr_lines_v2':
                value = dict(description='A synthetic red square on a white table', entities=[],
                    objects=[dict(name='square', count=1, attributes=['red'])], events=[],
                    relations=[], tags=['synthetic'], location_hint='', ocr_lines=[])
            elif schema == 'ocr_physical_text_localization_v1':
                value = dict(inspected_entire_image=True, assessment='no_text', regions=[])
            else:
                # Malformed optional evidence review must retain the base recall.
                value = {'fixture': 'unusable optional review'}
            content = json.dumps(value)
            usage = dict(prompt_tokens=30, completion_tokens=60, total_tokens=90)
            if body.get('stream'):
                rows = [dict(model='gpt-4o-mini', choices=[dict(index=0,
                    delta=dict(content=content), finish_reason=None)]),
                    dict(model='gpt-4o-mini', choices=[dict(index=0,delta={},finish_reason='stop')],usage=usage)]
                encoded = (''.join('data: '+json.dumps(r)+'\n\n' for r in rows)+'data: [DONE]\n\n').encode()
                kind = 'text/event-stream'
            else:
                encoded = json.dumps(dict(model='gpt-4o-mini', choices=[dict(
                    finish_reason='stop', message=dict(content=content))], usage=usage)).encode()
                kind = 'application/json'
            self.send_response(200); self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
    server = ThreadingHTTPServer(('127.0.0.1',0),Handler); server.daemon_threads=True
    thread = threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try: yield 'http://127.0.0.1:'+str(server.server_port)+'/v1',calls
    finally:server.shutdown();server.server_close();thread.join(timeout=3)


def launch(env, port, log, initialize=False):
    """Only the paid provider transport is replaced; the launcher/factory run."""
    from mandol_competition.inference.dashscope_rerank import DashScopeRerankerClient
    from mandol_competition.inference.http import HTTPTransport
    from mandol_competition.multimodal.config import get_config
    original = DashScopeRerankerClient.__init__
    def init(self, endpoint, **kwargs):
        def reply(request):
            body=json.loads(request.content)
            return httpx.Response(200,json={'output':{'results':[
                {'index':i,'relevance_score':1/(i+1)} for i in range(len(body['input']['documents']))]}})
        kwargs['transport']=HTTPTransport('https://unit.test',max_attempts=1,
            transport=httpx.MockTransport(reply))
        original(self,endpoint,**kwargs)
    DashScopeRerankerClient.__init__=init
    os.environ.clear();os.environ.update(env);get_config.cache_clear()
    with open(log,'w') as output:
        os.dup2(output.fileno(),1);os.dup2(output.fileno(),2)
        sys.argv=['start_api.py','multimodal','--port',str(port)]
        if initialize:sys.argv.append('--initialize')
        runpy.run_path(str(ROOT/'scripts/start_api.py'))['main']()


def test_public_script_image_add_replay_search_and_isolation(tmp_path):
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    token='synthetic-fixture-only';namespace='public-fixture-'+uuid.uuid4().hex
    with formal_stack_qdrant(tmp_path/'qdrant') as qdrant, native_fixture('v4') as (v4,_), visual_fixture() as (vlm,calls):
        env=dict(os.environ,MANDOL_RUNTIME_ROOT=str(tmp_path/'data'),MANDOL_MEMORY_SYSTEM_KEY=token,
            MANDOL_QDRANT_ENDPOINT=qdrant,MANDOL_MULTIMODAL_NAMESPACE=namespace,
            MANDOL_MULTIMODAL_EMBEDDING_ENDPOINT=v4,MANDOL_MULTIMODAL_SPARSE_ENDPOINT=v4,
            MANDOL_MULTIMODAL_VLM_ENDPOINT=vlm,CLOSEAI_API_KEY='synthetic-fixture-only',
            DASHSCOPE_API_KEY='synthetic-fixture-only')
        env.pop('MANDOL_CONFIG',None)
        process=multiprocessing.get_context('spawn').Process(target=launch,args=(env,port,str(tmp_path/'api.log'),True))
        process.start()
        url='http://127.0.0.1:'+str(port);headers={'X-Api-Key':token}
        try:
            with httpx.Client(base_url=url,timeout=30,trust_env=False) as client:
                until=time.monotonic()+40
                while time.monotonic()<until:
                    assert process.is_alive(),(tmp_path/'api.log').read_text()[-2500:]
                    try:
                        if client.get('/readyz',timeout=1).status_code==200:break
                    except httpx.RequestError:pass
                    time.sleep(.1)
                else:raise AssertionError('Public synthetic API did not become ready')
                assert client.get('/health').json()=={'status':'ok'}
                # The normal deployment restart uses audit=0 on the newly bound store.
                process.terminate();process.join(timeout=30);assert not process.is_alive()
                process=multiprocessing.get_context('spawn').Process(target=launch,args=(env,port,str(tmp_path/'restart.log')))
                process.start();until=time.monotonic()+40
                while time.monotonic()<until:
                    assert process.is_alive(),(tmp_path/'restart.log').read_text()[-2500:]
                    try:
                        if client.get('/readyz',timeout=1).status_code==200:break
                    except httpx.RequestError:pass
                    time.sleep(.1)
                else:raise AssertionError('Public fast restart did not become ready')
                stream=io.BytesIO();Image.new('RGB',(32,32),'red').save(stream,format='PNG')
                image={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}
                body=dict(request_id='synthetic-add',user_id='synthetic-owner',session_id='session',
                    messages=[dict(role='user',content=[dict(type='text',text='Synthetic red square memory'),image])])
                assert client.post('/add',headers=headers,json=body).status_code==200
                count=len(calls)
                assert client.post('/add',headers=headers,json=body).status_code==200
                assert len(calls)==count, 'Replay paid for visual extraction again'
                query=dict(query='red square',user_id='synthetic-owner',top_k=100)
                def search(options=None):
                    response=client.post('/search',headers=headers,json=query if options is None else query|{'options':options})
                    assert response.status_code==200,response.text
                    rows=response.json()['data'];assert rows and len(rows)<=100
                    assert all(isinstance(r['id'],str) and r['content'] for r in rows)
                    json.dumps(rows,allow_nan=False)
                    return [r['id'] for r in rows]
                expected=search();assert search(['red square','other'])==expected
                with ThreadPoolExecutor(16) as pool:
                    assert all(ids==expected for ids in pool.map(lambda _:search(),range(16)))
                assert client.post('/search',headers=headers,json=query|{'user_id':'synthetic-stranger'}).json()=={'data':[]}
        finally:
            process.terminate();process.join(timeout=30)
            if process.is_alive():process.kill();process.join(timeout=5)
