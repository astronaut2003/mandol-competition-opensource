"""Native-v4 API factories with real storage and synthetic HTTP inference.

This checks storage, HTTP contracts and lifecycle behavior with synthetic models.
"""
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

import httpx
from PIL import Image

from mandol_competition.inference.dashscope import official_embedding_recipes
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes
from tests.support.http_inference import fake_inference

ROOT = Path(__file__).resolve().parents[2]

@contextmanager
def native_fixture(channel):
    dense, sparse = official_embedding_recipes()
    profile = load_profile(MultimodalConfig(retrieval_profile_path=ROOT/'configs/profiles/aml-official-eligible/multimodal-v1.json', model_device='external-http'))
    active = (dense, sparse) if channel == 'v4' else (recipes(profile)[2],)
    calls = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *_): pass
        def reply(self, status, value):
            encoded=json.dumps(value).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(encoded)));self.end_headers();self.wfile.write(encoded)
        def do_GET(self):
            self.rfile.read(int(self.headers.get('Content-Length','0')))
            if self.path in ('/health','/ready'):
                return self.reply(200,{'status':'ok'})
            if self.path == '/v1/recipe':
                return self.reply(200,{'recipes':[{'recipe':asdict(r),'hash':r.hash} for r in active]})
            self.reply(404,{})
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append({'path':self.path,'mode':body.get('input_type'),
                'input_hashes':[hashlib.sha256(t.encode()).hexdigest() for t in body.get('input',[])]})
            if self.path == '/v1/encode':
                assert body['model']==dense.served_alias
                assert body['dense_recipe_hash']==dense.hash and body['sparse_recipe_hash']==sparse.hash
                return self.reply(200,{'model':dense.served_alias,'data':[{'index':i,'pair':{
                    'embedding':[1.]+[0.]*1023,'sparse':{'51765':.4}}} for i in range(len(body['input']))]})
            if self.path == '/v1/embeddings':
                # Code legitimately requests only Dense; it does not use the
                # paired Dense/native-Sparse route used by Textual and MM.
                assert channel == 'v4' and body['model'] == dense.served_alias
                assert body['recipe_hash'] == dense.hash
                return self.reply(200,{'model':dense.served_alias,'data':[
                    {'index':i,'embedding':[1.]+[0.]*1023}
                    for i in range(len(body['input']))]})
            if self.path == '/v1/score':
                assert body['recipe_hash']==active[0].hash and body['model']==active[0].served_alias
                return self.reply(200,{'model':body['model'],'data':[{'index':i,'score':.7} for i in range(len(body['documents']))]})
            self.reply(404,{})
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.daemon_threads=True
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:yield f'http://127.0.0.1:{server.server_port}',calls
    finally:server.shutdown();server.server_close();thread.join()


def free_port():
    with socket.socket() as s:s.bind(('127.0.0.1',0));return s.getsockname()[1]


def test_three_formal_frozen_factories_replay_restart_and_isolation(tmp_path):
    qdrant=os.environ['MANDOL_TEST_QDRANT']
    suffix=uuid.uuid4().hex;key='synthetic-'+suffix
    processes={};logs=[];addresses={};namespaces={};created=[]
    marker={'textual':'TEXTUAL_ONLY_WEST_LAKE','code':'CODE_ONLY_JWT_FIX','multimodal':'MULTIMODAL_ONLY_RED_BAG'}
    with ExitStack() as stack, httpx.Client(timeout=40,trust_env=False) as http:
        v4,v4_calls=stack.enter_context(native_fixture('v4'))
        bge,_=stack.enter_context(native_fixture('bge'))
        querit,_=stack.enter_context(fake_inference('reranker'))
        vlm,vlm_calls=stack.enter_context(fake_inference('vlm'))
        env=dict(os.environ, MANDOL_RUNTIME_ROOT=str(tmp_path/'data'),MANDOL_MEMORY_SYSTEM_KEY=key,
            MANDOL_QDRANT_ENDPOINT=qdrant,MANDOL_TEXTUAL_INFERENCE_CONTRACT='aml-official-eligible-v1',
            MANDOL_CODE_INFERENCE_CONTRACT='aml-official-eligible-v1',MANDOL_MULTIMODAL_INFERENCE_CONTRACT='aml-official-eligible-v1',
            MANDOL_TEXTUAL_EMBEDDING_ENDPOINT=v4,MANDOL_TEXTUAL_SPARSE_ENDPOINT=v4,
            MANDOL_CODE_EMBEDDING_ENDPOINT=v4,MANDOL_TEXTUAL_RERANKER_ENDPOINT=querit,
            MANDOL_MULTIMODAL_EMBEDDING_ENDPOINT=v4,MANDOL_MULTIMODAL_SPARSE_ENDPOINT=v4,
            MANDOL_MULTIMODAL_RERANKER_ENDPOINT=bge,MANDOL_MM_MODEL_DEVICE='external-http',
            MANDOL_MM_RETRIEVAL_PROFILE=str(ROOT/'configs/profiles/aml-official-eligible/multimodal-v1.json'),
            MANDOL_MM_VLM_BASE_URL=vlm+'/v1',MANDOL_MM_VLM_API_KEY_ENV='OPENAI_API_KEY',
            OPENAI_API_KEY='synthetic-fixture',MANDOL_MM_VLM_MODEL='gpt-4o-mini-closeai',
            MANDOL_MM_VLM_REQUEST_MODEL='gpt-4o-mini')
        env.pop('MANDOL_CONFIG',None);env.pop('PYTHONPATH',None)
        for track in marker:
            namespaces[track]='native-test-'+suffix+'-'+track
            addresses[track]='http://127.0.0.1:'+str(free_port())
            assert http.get(qdrant+'/collections/'+track+'_aml_v4_v1').status_code==404
        def start(track):
            log=(tmp_path/(track+'-'+str(len(logs))+'.log')).open('w');logs.append(log)
            p=subprocess.Popen([sys.executable,'-m','uvicorn','mandol_competition.runtime.apps:'+track,
                '--factory','--host','127.0.0.1','--port',addresses[track].rsplit(':',1)[1],
                '--workers','1','--no-access-log'],cwd=ROOT,env=env|{'MANDOL_'+track.upper()+'_NAMESPACE':namespaces[track]},
                stdout=log,stderr=subprocess.STDOUT)
            processes[track]=p;end=time.monotonic()+30
            while time.monotonic()<end:
                assert p.poll() is None,(tmp_path/log.name).read_text()[-2500:]
                try:
                    if http.get(addresses[track]+'/readyz',timeout=1).status_code==200:return
                except httpx.RequestError:pass
                time.sleep(.1)
            raise AssertionError('synthetic native factory not ready: '+track)
        def stop(track):
            p=processes.pop(track);p.terminate();p.wait(timeout=30)
        def search(track,top_k=100):
            value=http.post(addresses[track]+'/search',headers={'X-Api-Key':key},json={
                'query':marker[track],'options':[marker[track],'unrelated'],
                'user_id':'same-user','top_k':top_k})
            assert value.status_code==200,value.text
            rows=value.json()['data'];assert 0<len(rows)<=top_k
            assert marker[track] in json.dumps(rows)
            assert all(marker[other] not in json.dumps(rows) for other in marker if other!=track)
            return rows
        try:
            for track in marker:start(track)
            stream=io.BytesIO();Image.new('RGB',(4,4),'red').save(stream,'PNG')
            image={'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()}}
            bodies={}
            for track in marker:
                content=('  '+marker[track]+'\n') if track!='multimodal' else [
                    {'type':'text','text':marker[track]},image,{'type':'text','text':'after'}]
                body={'request_id':'request-'+track,'user_id':'same-user','session_id':'session-1',
                    'messages':[{'role':'user','content':content,'timestamp':1791504000000}]}
                bodies[track]=body;url=addresses[track]
                for header in ({'X-Api-Key':key},{'Authorization':'Token '+key},{'Authorization':'Bearer '+key}):
                    r=http.post(url+'/add',headers=header,json=body);assert r.status_code==200,r.text
                    assert r.json()=={'success':True,**{k:body[k] for k in ('request_id','user_id','session_id')}}
                assert http.post(url+'/add',headers={'X-Api-Key':key},json=body|{'session_id':'changed'}).status_code==409
                assert http.get(url+'/health').json()=={'status':'ok'}
                assert http.get(url+'/livez').status_code==200
                search(track);search(track,1)
                empty=http.post(url+'/search',headers={'X-Api-Key':key},json={'query':marker[track],'user_id':'other-user','top_k':100})
                assert empty.json()=={'data':[]}
                created.append(track+'_aml_v4_v1')
            assert vlm_calls.count('/v1/chat/completions')==1
            assert len(list((tmp_path/'data').rglob('state.sqlite3')))==3
            assert list((tmp_path/'data/multimodal/media').rglob('*.png')) or any((tmp_path/'data/multimodal/media').rglob('*'))
            # Hash-only tracking separates admission probes from corpus encoding.
            source_hashes={hashlib.sha256(text.encode()).hexdigest() for text in
                (marker['textual'], '  '+marker['code']+'\n')}
            def encoded_sources():return sum(h in source_hashes for c in v4_calls if c['mode']=='document' for h in c['input_hashes'])
            before=encoded_sources();before_vlm=len(vlm_calls)
            original={t:[r['id'] for r in search(t)] for t in marker}
            for track in list(marker):stop(track)
            for track in marker:start(track)
            for track in marker:
                assert [r['id'] for r in search(track)]==original[track]
                assert http.post(addresses[track]+'/add',headers={'X-Api-Key':key},json=bodies[track]).status_code==200
            assert len(vlm_calls)==before_vlm,'completed preparation repeated VLM'
            assert encoded_sources()==before,'restart/replay re-encoded raw corpus'
            info=http.get(qdrant+'/collections').json()['result']['collections']
            assert {r['name'] for r in info} >= set(created)
        finally:
            for track in list(processes):stop(track)
            for log in logs:log.close()
            for track in marker:
                collection=track+'_aml_v4_v1'
                r=http.get(qdrant+'/collections/'+collection)
                if r.status_code==200 and r.json()['result']['config'].get('metadata',{}).get('namespace')==namespaces[track]:
                    http.delete(qdrant+'/collections/'+collection).raise_for_status()
