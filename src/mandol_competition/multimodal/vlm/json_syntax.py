"""Repair punctuation around complete JSON values without inventing evidence.

This deliberately excludes completion of missing strings, containers or values.
Callers must first verify a non-truncated provider finish and then validate the
returned object against their exact output schema.
"""
from __future__ import annotations

import json
import re

POLICY = 'complete-json-punctuation-v1'
MAX_CONTENT_CHARS = 1_048_576


class JSONSyntaxError(ValueError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise JSONSyntaxError('duplicate JSON field')
        result[key] = value
    return result


def _constant(_):
    raise JSONSyntaxError('non-finite JSON value')


def complete_object(content):
    """Return (object, optional repair receipt), preserving every scalar token."""
    if not isinstance(content, str) or len(content) > MAX_CONTENT_CHARS:
        raise JSONSyntaxError('bounded string response required')
    decoder = json.JSONDecoder(object_pairs_hook=_unique, parse_constant=_constant)
    try:
        value = decoder.decode(content)
    except json.JSONDecodeError:
        pass
    except RecursionError:
        raise JSONSyntaxError('JSON nesting too deep') from None
    else:
        if not isinstance(value, dict):
            raise JSONSyntaxError('one JSON object required')
        return value, None

    text = content.strip()
    changes = dict(fence_removed=0, commas_inserted=0, trailing_commas_removed=0)
    fence = re.fullmatch(r'```(?:json)?[ \t]*\r?\n([\s\S]*)\r?\n```', text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
        changes['fence_removed'] = 1
    position = 0

    def whitespace():
        nonlocal position
        while position < len(text) and text[position] in ' \r\n\t':
            position += 1

    def peek():
        whitespace()
        return text[position] if position < len(text) else ''

    def consume(expected):
        nonlocal position
        if peek() != expected:
            raise JSONSyntaxError('missing or incomplete JSON structure')
        position += 1

    def scalar(*, key=False):
        nonlocal position
        whitespace()
        start = position
        try:
            value, position = decoder.raw_decode(text, position)
        except (json.JSONDecodeError, RecursionError):
            raise JSONSyntaxError('incomplete or invalid JSON scalar') from None
        if isinstance(value, (list, dict)) or key and not isinstance(value, str):
            raise JSONSyntaxError('explicit scalar required')
        if position < len(text) and text[position].isalnum():
            raise JSONSyntaxError('invalid scalar boundary')
        return text[start:position]

    def item(depth=0):
        if depth > 64:
            raise JSONSyntaxError('JSON nesting too deep')
        char = peek()
        if char not in ('{', '['):
            return scalar()
        object_mode = char == '{'
        closer = '}' if object_mode else ']'
        consume(char)
        parts = [char]
        if peek() == closer:
            consume(closer)
            return char + closer
        while True:
            if object_mode:
                parts.append(scalar(key=True))
                consume(':')
                parts.append(':')
            parts.append(item(depth + 1))
            next_char = peek()
            if next_char == closer:
                consume(closer)
                parts.append(closer)
                return ''.join(parts)
            if next_char == ',':
                consume(',')
                if peek() == closer:
                    changes['trailing_commas_removed'] += 1
                    consume(closer)
                    parts.append(closer)
                    return ''.join(parts)
            elif next_char and (not object_mode or next_char == '"'):
                changes['commas_inserted'] += 1
            else:
                raise JSONSyntaxError('incomplete JSON container')
            parts.append(',')

    repaired = item()
    if peek() or not any(changes.values()):
        raise JSONSyntaxError('one complete JSON object required')
    try:
        value = decoder.decode(repaired)
    except (json.JSONDecodeError, RecursionError):
        raise JSONSyntaxError('JSON syntax repair rejected') from None
    if not isinstance(value, dict):
        raise JSONSyntaxError('one JSON object required')
    return value, dict(policy=POLICY, **changes)
