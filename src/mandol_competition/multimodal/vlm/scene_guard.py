"""Conservative reread signals for scene strings; never alter visual facts."""
import json
import re

from .ocr_recovery import OCRRecoveryError


class SceneOutputNeedsReview(OCRRecoveryError):
    """Abort a paid scene stream, preserving its charge for bounded recovery."""
    vision_paid_output_failure = True

    def __init__(self, reason='scene_text_repetition'):
        self.reason = reason
        super().__init__(reason)


_STRING = re.compile(r'"((?:[^"\\]|\\.)*)(?:"|$)')


def _repeats(text):
    # Literal repetitions in OCR are intentionally outside this check. A
    # reread signal does not delete duplicates or promote a partial response.
    if re.search(r'(?:[ \t]*\r?\n){64}', text):
        return True
    words = re.findall(r'\w+|[^\w\s]', text[-4096:], re.UNICODE)
    for size in range(1, min(32, len(words) // 12) + 1):
        tail = words[-size * 12:]
        if len(' '.join(tail)) >= 160 and tail == tail[:size] * 12:
            return True
    tail = text[-2048:]
    for size in range(1, min(32, len(tail) // 32) + 1):
        repeated = tail[-size * 32:]
        if len(repeated) >= 256 and repeated == repeated[:size] * 32:
            return True
    return False


def scene_stream_observer(*, primary=False):
    def inspect(content):
        if not isinstance(content, str):
            return
        # The primary schema places OCR last. Its literal strings and genuine
        # repetitions belong to the existing physical OCR recovery, not here.
        if primary:
            marker = re.search(r'"ocr_lines"\s*:', content)
            if marker is not None:
                content = content[:marker.start()]
        for match in _STRING.finditer(content):
            wire = match.group(1)
            try:
                value = json.loads('"' + wire + '"')
            except (ValueError, TypeError):
                # A half escape/unicode sequence is not evidence of repetition.
                continue
            if _repeats(value):
                raise SceneOutputNeedsReview()
    return inspect


def validate_scene_strings(value):
    if isinstance(value, str):
        if _repeats(value):
            raise SceneOutputNeedsReview()
    elif isinstance(value, dict):
        for key, child in value.items():
            if key not in {'ocr_lines', 'ocr_text'}:
                validate_scene_strings(child)
    elif isinstance(value, list):
        for child in value:
            validate_scene_strings(child)
