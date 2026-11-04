"""Real TCP with synthetic model outputs, never real-model qualification."""
from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import threading
import time

from mandol_competition.inference.dashscope import official_embedding_recipes
from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes


class Control:
    def __init__(self):
        self.down = self.wrong = self.partial = False
        self.delay = 0.
        self.calls = []
        self.lock = threading.Lock()


@contextmanager
def frozen_http(channel):
    control = Control()
    rr = recipes(load_profile(MultimodalConfig(inference_contract='aml-official-eligible-v1', model_device='external-http')))[2]
    advertised = [rr] if channel == 'bge' else list(official_embedding_recipes())
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def setup(self):
            super().setup()
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        def respond(self, status, obj):
            encoded = json.dumps(obj).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded))); self.end_headers()
            self.wfile.write(encoded)
        def do_GET(self):
            if control.down: return self.respond(503, {})
            if self.path in ('/health', '/ready'): return self.respond(200, {'status':'ok'})
            if self.path == '/version': return self.respond(200, {'version':'0.25.1' if channel == 'bge' else '0.29.0'})
            if self.path == '/v1/models':
                return self.respond(200, {'data':[{'id':'querit-4b', 'root':'wrong' if control.wrong else 'Querit/Querit-4B', 'max_model_len':40960}]})
            if self.path == '/openapi.json': return self.respond(200, {'paths':{'/classify':{}}})
            if self.path == '/v1/recipe':
                return self.respond(200, {'recipes':[{'recipe':asdict(r), 'hash':'wrong' if control.wrong else r.hash} for r in advertised]})
            self.respond(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            docs = body.get('documents', body.get('input', []))
            with control.lock:
                control.calls.append({'path':self.path, 'count':len(docs),
                    'input_digest':hashlib.sha256(json.dumps(docs, sort_keys=True).encode()).hexdigest()})
            if control.delay: time.sleep(control.delay)
            if control.down: return self.respond(503, {})
            alias = body.get('model', '')
            def batch(field, values):
                values = values[:-1] if control.partial else values
                self.respond(200, {'model':alias, 'data':[{'index':i, field:v} for i,v in enumerate(values)]})
            if self.path == '/v1/encode':
                return batch('pair', [{'embedding':[1.]+[0.]*1023, 'sparse':{'51765':.75}} for _ in body['input']])
            if self.path == '/v1/embeddings': return batch('embedding', [[1.]+[0.]*1023 for _ in body['input']])
            if self.path == '/v1/sparse': return batch('sparse', [{'51765':.75} for _ in body['input']])
            if self.path == '/v1/score': return batch('score', [.5 for _ in body['documents']])
            if self.path == '/tokenize':
                ids = {'[CLS]':[151665], '<|endoftext|>':[151643]}.get(body['prompt'], [7])
                return self.respond(200, {'tokens':ids, 'count':len(ids)})
            if self.path == '/detokenize': return self.respond(200, {'prompt':'[CLS]' if body['tokens']==[151665] else '<|endoftext|>'})
            if self.path == '/classify':
                return self.respond(200, {'model':alias, 'data':[{'index':i, 'num_classes':2, 'probs':[.25,.75]} for i,_ in enumerate(body['input'])]})
            if self.path == '/v1/chat/completions':
                extraction = dict(description='a red square', entities=[], objects=[], ocr_text='', events=[], relations=[], tags=[], location_hint='')
                return self.respond(200, {'model':'gpt-4o-mini', 'choices':[{'message':{'content':json.dumps(extraction)}}]})
            self.respond(404, {})
        def log_message(self, *_): pass
    class FixtureServer(ThreadingHTTPServer):
        # Fixture networking must not cap the C16 client below the API/pool's
        # configured capacity (the stdlib listen backlog otherwise defaults5).
        request_queue_size = 128
    server = FixtureServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True; server.handle_error = lambda *_: None
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', control
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=3)
