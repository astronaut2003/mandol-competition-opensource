import hashlib
import json
import struct

import pytest

from scripts.qualification.prepare_querit_vllm import rename_head


def test_querit_conversion_preserves_bias_and_exact_tensor_bytes(tmp_path):
    source, target = tmp_path / 'original', tmp_path / 'converted'
    header = {'head.weight': {'dtype': 'BF16', 'shape': [2, 2], 'data_offsets': [0, 8]},
              'head.bias': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [8, 12]},
              'model.embed_tokens.weight': {'dtype': 'BF16', 'shape': [1, 2], 'data_offsets': [12, 16]}}
    encoded = json.dumps(header).encode(); payload = bytes(range(16))
    source.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert rename_head(source, target) == hashlib.sha256(payload).hexdigest()
    data = target.read_bytes(); size = struct.unpack('<Q', data[:8])[0]
    actual = json.loads(data[8:8 + size])
    assert actual['score.weight'] == header['head.weight']
    assert actual['score.bias'] == header['head.bias']
    assert actual['model.embed_tokens.weight'] == header['model.embed_tokens.weight']
    assert 'head.weight' not in actual and data[8 + size:] == payload
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    with pytest.raises(FileExistsError):
        rename_head(source, target)
