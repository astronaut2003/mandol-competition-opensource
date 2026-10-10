import json
from mandol_competition.inference.http import HTTPTransport


def test_status_only_health_consumes_body_and_reuses_real_connection(tmp_path,monkeypatch):
    from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
    import threading
    ports=[]
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args):pass
        def do_GET(self):
            # HTTPTransport includes a JSON-null body even on GET; consume it.
            self.rfile.read(int(self.headers.get('Content-Length','0')))
            ports.append(self.client_address[1]);body=b'{"status":"ok"}'
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    monkeypatch.setenv('MANDOL_INFERENCE_TRACE',str(tmp_path/'trace.jsonl'))
    client=HTTPTransport(f'http://127.0.0.1:{server.server_port}')
    try:
        for _ in range(30):assert client.health()
        assert len(set(ports))==1
        events=[json.loads(x) for x in (tmp_path/'trace.jsonl').read_text().splitlines()]
        assert sum(x['new_connections'] for x in events)==1
        assert all(x['status']==200 for x in events)
    finally:client.close();server.shutdown();server.server_close();thread.join()

