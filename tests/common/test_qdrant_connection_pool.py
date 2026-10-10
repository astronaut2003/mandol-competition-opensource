"""Real SDK/HTTP transport, fake root endpoint; no vector/model qualification."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading

from mandol_substrate.qdrant.backend import QdrantProjectionBackend
from mandol_substrate.state import RetrievalProfile, SQLiteStateStore
from mandol_substrate.text import MandolTextProcessor


def test_localhost_reuses_connections_and_reconnects_after_server_close(tmp_path):
    ports = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def log_message(self, *args):
            pass
        def do_GET(self):
            ports.append(self.client_address[1])
            body = json.dumps({'title': 'qdrant', 'version': '1.19.1', 'commit': 'fixture'}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            if len(ports) == 11:
                self.send_header('Connection', 'close')
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    profile = RetrievalProfile('test/dense', 'fixed', 3, 'fixed',
        'test/sparse', 'fixed', MandolTextProcessor().fingerprint)
    state = SQLiteStateStore(tmp_path/'state.sqlite3', profile)
    backend = None
    try:
        backend = QdrantProjectionBackend(f'http://127.0.0.1:{server.server_port}',
            profile, state=state, storage_id=state.db_uuid)
        for _ in range(20):
            assert backend.health()
        assert len(ports) == 21
        assert len(set(ports[:11])) == 1
        assert len(set(ports[11:])) == 1
        assert ports[0] != ports[-1]
    finally:
        if backend is not None:
            backend.close()
        state.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
