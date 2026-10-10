"""Operator inspection must not corrupt a running soak's evidence or kill it."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import threading


def helper(monkeypatch, name):
    directory = Path(__file__).resolve().parents[2] / 'deployment' / 'cluster'
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location('evidence_' + name, directory / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_readers_only_see_complete_json_during_concurrent_writes(tmp_path, monkeypatch):
    module = helper(monkeypatch, 'qualification')
    target = tmp_path / 'status.json'
    payloads = [{'writer': i, 'padding': str(i) * 4096} for i in range(8)]
    module.save(target, payloads[0])
    done = threading.Event()
    observations = []

    def read():
        while not done.is_set():
            value = json.loads(target.read_text())
            assert value in payloads
            observations.append(value['writer'])

    def write(payload):
        for _ in range(30):
            module.save(target, payload)

    with ThreadPoolExecutor(max_workers=9) as pool:
        reader = pool.submit(read)
        try:
            futures = [pool.submit(write, payload) for payload in payloads]
            for future in futures:
                future.result()
        finally:
            done.set()
        reader.result()
    assert observations
    assert (target.stat().st_mode & 0o777) == 0o600
    assert list(tmp_path.iterdir()) == [target]


def test_running_report_and_inspector_have_separate_temporary_files(tmp_path, monkeypatch):
    module = helper(monkeypatch, 'soak')
    summary = {'status': 'SOAK_RUNNING', 'start': 'start', 'target_end': 'end',
               'pid': 123, 'tmux': 'test', 'implementation_commit': 'test',
               'checks': 3, 'errors': 0, 'external_requests': {}, 'component_requests': {}}
    audit = {'component_distribution': {}, 'api_distribution': {}, 'violations': []}
    barrier = threading.Barrier(8)

    def writer(index):
        barrier.wait(timeout=5)
        for _ in range(30):
            module.report(tmp_path, summary, audit if index % 2 else None)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(writer, range(8)))
    report = tmp_path / 'Report.md'
    assert report.read_text().startswith('# Local soak results\n')
    assert 'Status: **SOAK_RUNNING**' in report.read_text()
    assert 'An incomplete run is never PASS.' in report.read_text()
    assert (report.stat().st_mode & 0o777) == 0o600
    assert list(tmp_path.iterdir()) == [report]


def test_inspector_defers_a_sample_that_is_still_being_appended(tmp_path, monkeypatch):
    module = helper(monkeypatch, 'soak')
    summary = {'status': 'SOAK_RUNNING', 'start': 'start', 'target_end': 'end',
               'pid': 123, 'tmux': 'test', 'implementation_commit': 'test',
               'checks': 3, 'errors': 0, 'external_requests': {}, 'component_requests': {}}
    (tmp_path / 'status.json').write_text(json.dumps(summary))
    sample = {'application': {}, 'nodes': [], 'metrics': {}, 'elapsed_seconds': 0}
    (tmp_path / 'metrics.jsonl').write_text(json.dumps(sample) + '\n' + '{"application":')
    observed, audit = module.summarize(tmp_path)
    assert observed['status'] == 'SOAK_RUNNING'
    assert audit['samples'] == 1
    assert audit['violations'] == []
