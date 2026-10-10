"""Optional exact Git path spans; never enabled implicitly by migration."""
import re

def canonical_git_paths(text):
    """Experimental exact spans; never globally rewrite real a/ or b/ paths."""
    from mandol_competition.code.software.anchors import CodeAnchor, CodeAnchorExtractor

    pattern = re.compile(r'(?m)^(?:--- a/|\+\+\+ b/|rename from |rename to )([^\t\r\n"]+)')
    result = []
    for match in pattern.finditer(text):
        start, end = match.span(1)
        raw = text[start:end]
        result.append(CodeAnchor("path", raw, CodeAnchorExtractor.normalize(raw), CodeAnchorExtractor.split(raw), start, end))
    return result

