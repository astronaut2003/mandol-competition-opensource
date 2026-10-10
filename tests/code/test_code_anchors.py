import pytest

from mandol_competition.code.software.anchors import CodeAnchorExtractor


def test_anchor_exact_source_and_components():
    text = 'QwenRerankerManager getUserProfile MAX_RETRY_COUNT Foo::bar pkg.module.func src/auth/token_manager.py'
    anchors = CodeAnchorExtractor().extract(text)
    assert all(text[a.start:a.end] == a.raw for a in anchors)
    manager = next(a for a in anchors if a.raw == "QwenRerankerManager")
    assert manager.normalized == "qwenrerankermanager"
    assert manager.components == ("qwen", "reranker", "manager")
    assert any(a.raw == "src/auth/token_manager.py" and a.kind == "path" for a in anchors)
    assert any(a.raw == "Foo::bar" and a.kind == "symbol" for a in anchors)


def test_diagnostic_anchors_preserve_punctuation_and_case():
    text = 'TypeError: expected int, got None\npytest tests/test_tokens.py::test_refresh_token_expired -q\nPython 3.12 foo==2.4.1 v1.2.3\nauth.enabled=false max_retries=3 timeout: 10'
    anchors = CodeAnchorExtractor().extract(text)
    pairs = {(a.kind, a.raw) for a in anchors}
    assert ("error", "TypeError: expected int, got None") in pairs
    assert ("command", "pytest tests/test_tokens.py::test_refresh_token_expired -q") in pairs
    assert ("test", "test_refresh_token_expired") in pairs
    assert ("version", "Python 3.12") in pairs
    assert ("version", "foo==2.4.1") in pairs
    assert ("config", "auth.enabled") in pairs


def test_empty_and_bad_source():
    extractor = CodeAnchorExtractor()
    assert extractor.extract("") == []
    with pytest.raises(TypeError):
        extractor.extract(None)


def test_code_token_channel_keeps_whole_and_split_identifiers():
    tokens = CodeAnchorExtractor().tokens("getUserProfile MAX_RETRY_COUNT src/auth.py")
    assert {"getuserprofile", "get", "user", "profile", "max_retry_count", "max", "retry", "count", "src/auth.py"} <= set(tokens)


def test_optional_git_paths_are_exact_canonical_spans_not_global_prefix_rewrites():
    from mandol_competition.code.software.diff_paths import canonical_git_paths
    extractor = CodeAnchorExtractor()
    text = '--- a/src/token.py\n+++ b/src/token.py\nrename from a/real directory.py\nrename to b/real directory.py\nordinary b/src/token.py\n+++ "b/escaped\\040name.py"\n'
    anchors = canonical_git_paths(text)
    assert [a.raw for a in anchors] == ['src/token.py', 'src/token.py', 'a/real directory.py', 'b/real directory.py']
    assert all(text[a.start:a.end] == a.raw for a in anchors)
    assert extractor.normalize('b/src/token.py') == 'b/src/token.py'
