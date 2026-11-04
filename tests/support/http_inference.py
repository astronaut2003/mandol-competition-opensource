"""Real TCP, synthetic outputs. This is contract testing, not model qualification."""
from contextlib import contextmanager
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from mandol_competition.multimodal.config import MultimodalConfig
from mandol_competition.multimodal.models import load_profile
from mandol_competition.multimodal.inference import recipes
from mandol_competition.textual.core.remote_inference import RemoteQueritClient


@contextmanager
def fake_inference(channel):
    profile = load_profile(MultimodalConfig())
    recipe_set = recipes(profile)
    identity = {'dense': ('qwen3-embedding-0.6b', 'Qwen/Qwen3-Embedding-0.6B', '/v1/embeddings', 4096),
                'sparse': ('splade-v3', 'naver/splade-v3', '/pooling', 512),
                'reranker': ('querit-4b', RemoteQueritClient.root, '/classify', 40960)}
    calls = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def respond(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path == '/health': return self.respond(200, {'status': 'ok'})
            if self.path == '/version': return self.respond(200, {'version': '0.29.0'})
            if self.path == '/v1/recipe': return self.respond(200, {'recipes': [{'recipe': asdict(r), 'hash': r.hash} for r in recipe_set]})
            alias, root, route, length = identity.get(channel, identity['dense'])
            if self.path == '/v1/models': return self.respond(200, {'data': [{'id': alias, 'root': root, 'max_model_len': length}]})
            if self.path == '/openapi.json': return self.respond(200, {'paths': {route: {}}})
            self.respond(404, {})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(self.path)  # No payload logging.
            alias = body.get('model', '')
            def batch(field, values):
                return self.respond(200, {'model': alias, 'data': [{'index': i, field: v} for i, v in enumerate(values)]})
            if self.path == '/v1/embeddings':
                return batch('embedding', [[1.] + [0.] * 1023 for _ in body['input']])
            if self.path == '/pooling':
                value = [0.] * 30522; value[7] = 1.
                return batch('data', [value for _ in body['input']])
            if self.path == '/v1/sparse': return batch('sparse', [{'7': 1., '9': 1e-8} for _ in body['input']])
            if self.path == '/v1/score': return batch('score', [.5 for _ in body['documents']])
            if self.path == '/tokenize':
                ids = {'[CLS]': [151665], '<|endoftext|>': [151643]}.get(body['prompt'], [7])
                return self.respond(200, {'tokens': ids, 'count': len(ids)})
            if self.path == '/detokenize':
                return self.respond(200, {'prompt': '[CLS]' if body['tokens'] == [151665] else '<|endoftext|>'})
            if self.path == '/classify':
                return self.respond(200, {'model': alias, 'data': [{'index': i, 'num_classes': 2, 'probs': [.2, .8]} for i, _ in enumerate(body['input'])]})
            if self.path in ('/chat/completions', '/v1/chat/completions'):
                return self.respond(200, {'model': alias, 'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'description': 'red backpack', 'entities': [], 'objects': [], 'ocr_text': '', 'events': [], 'relations': [], 'tags': [], 'location_hint': ''})}}]})
            self.respond(404, {})
        def log_message(self, *_): pass
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    server.handle_error = lambda *_: None  # clients may close keepalive connections at shutdown
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown(); server.server_close(); thread.join()
