import ast
from dataclasses import asdict

from mandol_competition.code.software.builder.parser.python import PythonParser, _SourceText


def test_cached_segments_match_stdlib_for_unicode_multiline_crlf_and_control_chars():
    source = 'def 函数(值):\r\n    变量 = "é\f\v"; result = pkg.call(\r\n        值, nested("中文"))\r\n    return result\r\n'
    cached = _SourceText(source)
    for node in ast.walk(ast.parse(source)):
        if hasattr(node, "lineno"):
            assert cached.segment(node) == ast.get_source_segment(source, node)


def test_parser_facts_are_identical_with_stdlib_segment_extraction(monkeypatch):
    source = '''class Example:
    def run(self, value):
        if value:
            self.result = parse(value, mode="é")
        def nested():
            return self.run(value)
        return nested()
'''
    current = asdict(PythonParser().parse_source(source, "example.py"))
    monkeypatch.setattr(_SourceText, "segment", lambda self, node: ast.get_source_segment(self, node))
    legacy = asdict(PythonParser().parse_source(source, "example.py"))
    assert current == legacy
