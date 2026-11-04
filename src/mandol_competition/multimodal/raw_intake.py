"""Immutable original Add inputs, saved before any paid visual extraction.

Media retain their validated original bytes. The private manifest preserves the
entire validated request, including the exact image URI representation, using
media references rather than duplicating inline images in the state ledger.
This store never publishes memories or declares an Add successful.
"""
from __future__ import annotations

import base64
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from mandol_substrate.state.models import Conflict, Scope, StateUnavailable, canonical_json, digest
from .media_store import UserMediaStore, compute_media_id
from .protocol import AddRequest, ImagePart

FORMAT = 'mm-raw-intake-v1'
RECEIPT_FORMAT = 'mm-raw-intake-receipt-v1'
_HEX = re.compile(r'[0-9a-f]{64}')
_RECEIPT_FIELDS = frozenset({'format', 'manifest_relpath', 'manifest_sha256',
    'scope_key', 'request_id_sha256', 'body_hash'})
_MEDIA_FIELDS = frozenset({'media_id', 'mime_type', 'byte_size', 'sha256', 'uri_header',
    'base64_prefix_length', 'base64_tail', 'uri_sha256'})


class RawIntakeError(StateUnavailable):
    """Original intake could not be durably saved or independently verified."""


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _need(condition):
    if not condition:
        raise RawIntakeError('original intake verification failed')


class RawIntakeStore:
    def __init__(self, config, user_id):
        self.config, self.user_id = config, user_id
        self.root = Path(config.user_root(user_id)).absolute()
        self.media = UserMediaStore(config, user_id)
        # The HTTP body already has a configured bound. Media references may
        # be larger than a tiny source image, so permit bounded metadata growth.
        self.maximum_manifest_bytes = 4 * config.max_request_bytes + 65536

    def _path(self, relative):
        path = self.root / relative
        _need(path.absolute().is_relative_to(self.root))
        _need('..' not in Path(relative).parts and not Path(relative).is_absolute())
        # Reject an existing symlink at any component before touching source
        # content. The leaf is also opened with O_NOFOLLOW below.
        for part in (path, *path.parents):
            _need(not part.is_symlink())
        return path

    def _binding(self, scope_key, request_id, body_hash):
        _need(type(scope_key) is str and _HEX.fullmatch(scope_key) is not None)
        _need(type(body_hash) is str and _HEX.fullmatch(body_hash) is not None)
        _need(type(request_id) is str and bool(request_id))
        _need(scope_key == Scope(self.config.namespace, self.user_id).key)
        return dict(scope_key=scope_key, user_id_sha256=_hash(self.user_id),
            request_id_sha256=_hash(request_id), body_hash=body_hash)

    def _read(self, path, limit):
        self._path(path.relative_to(self.root))
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, 'rb') as stream:
            info = os.fstat(stream.fileno())
            _need(stat.S_ISREG(info.st_mode) and 0 <= info.st_size <= limit)
            data = stream.read(limit + 1)
        _need(len(data) <= limit)
        return data

    def _fsync_directories(self, directory):
        # Include newly-created namespace/user parents in the durability fence.
        for part in (directory, *directory.parents):
            descriptor = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def _persist_once(self, path, data):
        self._path(path.relative_to(self.root))
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._path(path.relative_to(self.root))
        _need(stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0)
        descriptor, temporary = tempfile.mkstemp(prefix='.intake-', suffix='.tmp', dir=path.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # An atomic no-clobber publication also protects callers that lack
            # the normal enclosing scope write lease. A competing body wins
            # neither by overwriting nor by selecting a different body-hash path.
            try:
                os.link(temporary, path)
            except FileExistsError:
                pass
            self._fsync_directories(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _durability_fence(self, manifest, envelope):
        files = {manifest}
        for message in envelope['request']['messages']:
            if isinstance(message['content'], list):
                for part in message['content']:
                    if part.get('type') == 'image_url':
                        ref = part['image_url']['url']
                        sha = ref['sha256']
                        relative = Path('media') / sha[:2] / sha[2:4] / (sha + '.bin')
                        files.update((self._path(relative), self._path(relative.with_suffix('.mime'))))
        directories = set()
        for path in files:
            self._path(path.relative_to(self.root))
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                _need(stat.S_ISREG(os.fstat(descriptor).st_mode))
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directories.add(path.parent)
        # Even an identical-file retry must repeat this fence: a preceding
        # attempt could have linked/replaced a file and then failed dir fsync.
        for directory in directories:
            self._fsync_directories(directory)

    def _media_reference(self, part, image):
        header, separator, encoded = part.image_url.url.partition(',')
        _need(separator == ',' and header.lower() == 'data:' + image.mime_type + ';base64')
        _need(base64.b64decode(encoded, validate=True) == image.data)
        media_id = compute_media_id(image.data)
        digest_value = media_id[4:]
        relative = Path('media') / digest_value[:2] / digest_value[2:4] / (digest_value + '.bin')
        self._path(relative)
        self._path(relative.with_suffix('.mime'))
        record = self.media.put(image)
        _need(record.media_id == media_id)
        canonical = base64.b64encode(image.data).decode('ascii')
        prefix_length = max(0, len(canonical) - 4)
        _need(encoded.startswith(canonical[:prefix_length]))
        # Only the last quartet/padding can vary while decoding to the same
        # validated bytes. Retain it and the MIME casing for exact body replay.
        return dict(media_id=media_id, mime_type=image.mime_type, byte_size=len(image.data),
            sha256=digest_value, uri_header=header, base64_prefix_length=prefix_length,
            base64_tail=encoded[prefix_length:], uri_sha256=_hash(part.image_url.url))

    def _restore_image(self, reference):
        _need(type(reference) is dict and set(reference) == _MEDIA_FIELDS)
        sha = reference['sha256']
        _need(type(sha) is str and _HEX.fullmatch(sha) is not None)
        _need(reference['media_id'] == 'img_' + sha)
        size = reference['byte_size']
        _need(type(size) is int and 0 < size <= self.config.max_image_bytes)
        mime = reference['mime_type']
        _need(mime in {'image/jpeg', 'image/png', 'image/webp'})
        relative = Path('media') / sha[:2] / sha[2:4] / (sha + '.bin')
        image = self._read(self._path(relative), self.config.max_image_bytes)
        actual_mime = self._read(self._path(relative.with_suffix('.mime')), 64).decode('ascii')
        _need(len(image) == size and compute_media_id(image) == reference['media_id'] and actual_mime == mime)
        canonical = base64.b64encode(image).decode('ascii')
        prefix, tail, header = (reference['base64_prefix_length'], reference['base64_tail'], reference['uri_header'])
        _need(type(prefix) is int and prefix == max(0, len(canonical) - 4))
        _need(type(tail) is str and type(header) is str and header.lower() == 'data:' + mime + ';base64')
        encoded = canonical[:prefix] + tail
        _need(len(encoded) <= 4 * ((self.config.max_image_bytes + 2) // 3))
        _need(base64.b64decode(encoded, validate=True) == image)
        uri = header + ',' + encoded
        _need(_hash(uri) == reference['uri_sha256'])
        return uri

    def save(self, request, decoded_messages, *, scope_key, body_hash):
        try:
            return self._save(request, decoded_messages, scope_key=scope_key, body_hash=body_hash)
        except (RawIntakeError, Conflict):
            raise
        except Exception as exc:
            raise RawIntakeError('original intake could not be saved') from exc

    def _save(self, request, decoded_messages, *, scope_key, body_hash):
        binding = self._binding(scope_key, request.request_id, body_hash)
        _need(request.user_id == self.user_id and digest(request.model_dump()) == body_hash)
        relative = 'raw-intake/' + binding['request_id_sha256'] + '.json'
        path = self._path(relative)
        if path.exists():
            data = self._read(path, self.maximum_manifest_bytes)
            existing = json.loads(data)
            _need(type(existing) is dict and existing.get('format') == FORMAT)
            old_binding = existing.get('binding')
            _need(type(old_binding) is dict)
            if old_binding.get('body_hash') != body_hash:
                raise Conflict('same request_id with different payload')
            _need(old_binding == binding)
        else:
            payload = deepcopy(request.model_dump())
            _need(len(decoded_messages) == len(request.messages))
            for index, (message, parts, images) in enumerate(decoded_messages):
                _need(message.model_dump() == request.messages[index].model_dump())
                _need(parts == message.parts())
                expected_images = {i for i, part in enumerate(parts) if isinstance(part, ImagePart)}
                _need(set(images) == expected_images)
                for part_index, image in images.items():
                    payload['messages'][index]['content'][part_index]['image_url']['url'] = self._media_reference(parts[part_index], image)
            envelope = dict(format=FORMAT, binding=binding, request=payload)
            data = canonical_json(envelope).encode('utf-8')
            _need(len(data) <= self.maximum_manifest_bytes)
            self._persist_once(path, data)
            # A competing atomic creation must be verified, never overwritten.
            stored = self._read(path, self.maximum_manifest_bytes)
            if stored != data:
                existing = json.loads(stored)
                if isinstance(existing, dict) and isinstance(existing.get('binding'), dict) and existing['binding'].get('body_hash') != body_hash:
                    raise Conflict('same request_id with different payload')
                raise RawIntakeError('original intake manifest differs')
        receipt = dict(format=RECEIPT_FORMAT, manifest_relpath=relative,
            manifest_sha256=hashlib.sha256(data).hexdigest(), scope_key=scope_key,
            request_id_sha256=binding['request_id_sha256'], body_hash=body_hash)
        _need(self.load(receipt, scope_key=scope_key, request_id=request.request_id, body_hash=body_hash) == request.model_dump())
        self._durability_fence(path, json.loads(data))
        return receipt

    def load(self, receipt, *, scope_key, request_id, body_hash):
        try:
            return self._load(receipt, scope_key=scope_key, request_id=request_id, body_hash=body_hash)
        except RawIntakeError:
            raise
        except Exception as exc:
            raise RawIntakeError('original intake could not be restored') from exc

    def _load(self, receipt, *, scope_key, request_id, body_hash):
        binding = self._binding(scope_key, request_id, body_hash)
        _need(type(receipt) is dict and set(receipt) == _RECEIPT_FIELDS)
        expected = dict(format=RECEIPT_FORMAT, manifest_relpath='raw-intake/' + binding['request_id_sha256'] + '.json',
            scope_key=scope_key, request_id_sha256=binding['request_id_sha256'], body_hash=body_hash)
        _need(all(receipt[key] == value for key, value in expected.items()))
        _need(type(receipt['manifest_sha256']) is str and _HEX.fullmatch(receipt['manifest_sha256']) is not None)
        data = self._read(self._path(receipt['manifest_relpath']), self.maximum_manifest_bytes)
        _need(hashlib.sha256(data).hexdigest() == receipt['manifest_sha256'])
        envelope = json.loads(data)
        _need(type(envelope) is dict and set(envelope) == {'format', 'binding', 'request'})
        _need(envelope['format'] == FORMAT and envelope['binding'] == binding)
        payload = deepcopy(envelope['request'])
        _need(type(payload) is dict and payload.get('user_id') == self.user_id and payload.get('request_id') == request_id)
        for message in payload.get('messages', []):
            if isinstance(message.get('content'), list):
                for part in message['content']:
                    if part.get('type') == 'image_url':
                        part['image_url']['url'] = self._restore_image(part['image_url']['url'])
        restored = AddRequest.model_validate(payload).model_dump()
        _need(restored == payload and digest(restored) == body_hash)
        return restored
