"""Lossless naming adaptation of the pinned Querit head for native vLLM.

No model execution or tensor arithmetic. Source files are verified and retained;
only safetensors header keys head.{weight,bias} become score.{weight,bias}.
The official complete prompt/final CLS remains the application's responsibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

from mandol_competition.textual.core.querit_reference import FILE_MANIFEST, MODEL_REVISION


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rename_head(source, destination):
    with source.open('rb') as src, destination.open('xb') as dst:
        size = struct.unpack('<Q', src.read(8))[0]
        header = json.loads(src.read(size))
        converted = {('score.' + k[5:] if k.startswith('head.') else k): v for k, v in header.items()}
        data = json.dumps(converted, separators=(',', ':')).encode()
        data += b' ' * (-len(data) % 8)
        dst.write(struct.pack('<Q', len(data))); dst.write(data)
        digest = hashlib.sha256()
        while chunk := src.read(1024 * 1024):
            dst.write(chunk); digest.update(chunk)
    with destination.open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]; stream.seek(size, 1)
        if hashlib.file_digest(stream, 'sha256').hexdigest() != digest.hexdigest():
            raise ValueError('tensor payload changed during naming adaptation')
    return digest.hexdigest()


def prepare(source, destination):
    if source.name != MODEL_REVISION or destination.exists():
        raise ValueError('pinned source and a new output directory required')
    for name, expected in FILE_MANIFEST.items():
        if sha(source / name) != expected['sha256']:
            raise ValueError('source artifact checksum mismatch: ' + name)
    destination.mkdir(parents=True)
    records = {}
    for name in FILE_MANIFEST:
        if name.endswith('.safetensors'):
            payload = rename_head(source / name, destination / name)
            records[name] = {'source_sha256': FILE_MANIFEST[name]['sha256'],
                             'output_sha256': sha(destination / name), 'tensor_payload_sha256': payload}
        elif name not in ('config.json', 'model.safetensors.index.json', 'modeling_querit_4b.py'):
            shutil.copyfile(source / name, destination / name)
    config = json.loads((source / 'config.json').read_text())
    config.update(architectures=['Qwen3ForSequenceClassification'], num_labels=2,
                  id2label={'0': 'irrelevant', '1': 'relevant'},
                  label2id={'irrelevant': 0, 'relevant': 1}, head_dtype='model',
                  is_original_qwen3_reranker=False, classifier_from_token=None)
    (destination / 'config.json').write_text(json.dumps(config, indent=2))
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    index['weight_map'] = {('score.' + k[5:] if k.startswith('head.') else k): v
                           for k, v in index['weight_map'].items()}
    (destination / 'model.safetensors.index.json').write_text(json.dumps(index, indent=2))
    record = {'schema': 'querit-native-vllm-head-naming-v1', 'source_revision': MODEL_REVISION,
              'source_manifest': FILE_MANIFEST, 'output_shards': records,
              'output_config_sha256': sha(destination / 'config.json'),
              'semantics': 'causal complete pair; LAST=final CLS; unchanged binary head including bias; BF16 head; softmax p1-p0'}
    (destination / 'conversion.json').write_text(json.dumps(record, indent=2))
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); prepare(args.source.resolve(), args.output.resolve())
    print('Pinned source verified; tensor payloads preserved; conversion manifest written.')
