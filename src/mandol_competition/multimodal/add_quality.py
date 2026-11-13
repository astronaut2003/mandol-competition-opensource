"""Durable per-image publication quality and an explicit read-only aggregate audit.

Canonical metadata travels in the same atomic publication as its memory. Rates
count visible images, never HTTP attempts, replays, or unclassified old records.
The audit is an operator action, not a full-table scan on each request/metrics.
"""
from collections import Counter
from pathlib import Path
import json
import sqlite3
import time

POLICY='mm-add-quality-v1'
KEY='visual_add_quality'
STATUSES=('complete','incomplete_ocr','source_only','unverified')

def annotate(unit,statuses):
    if any(s not in STATUSES for s in statuses):raise ValueError('invalid image publication quality')
    unit.metadata[KEY]=dict(policy=POLICY,images=[dict(image_position=i,status=s) for i,s in enumerate(statuses)])

def aggregate(metadata):
    counts=Counter({s:0 for s in STATUSES});classified=legacy=0
    for item in metadata:
        value=item.get(KEY)
        if value is None:legacy+=1;continue
        if (not isinstance(value,dict) or value.get('policy')!=POLICY or not isinstance(value.get('images'),list)):
            raise ValueError('quality metadata is inconsistent')
        for i,image in enumerate(value['images']):
            if (not isinstance(image,dict) or set(image)!={'image_position','status'}
                    or type(image['image_position']) is not int or image['image_position']!=i
                    or image['status'] not in STATUSES):
                raise ValueError('quality image metadata is inconsistent')
            counts[image['status']]+=1
        classified+=1
    total=sum(counts.values())
    ratio=lambda n:n/total if total else None
    return dict(policy=POLICY,population='visible_active_memories',classified_memories=classified,
        unclassified_legacy_memories=legacy,published_images=total,counts=dict(counts),
        source_only_rate=ratio(counts['source_only']),incomplete_ocr_rate=ratio(counts['incomplete_ocr']),
        degradation_rate=ratio(counts['source_only']+counts['incomplete_ocr']),
        unverified_rate=ratio(counts['unverified']),retry_attempts_counted=False)

def audit(database,*,since_ms=0,timeout=30):
    path=Path(database).resolve(strict=True)
    if type(since_ms) is not int or since_ms<0 or not 0<timeout<=300:
        raise ValueError('invalid quality audit bounds')
    db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=2)
    expiry=time.monotonic()+timeout
    try:
        db.execute('PRAGMA query_only=ON')
        db.set_progress_handler(lambda:int(time.monotonic()>expiry),10000)
        rows=db.execute('SELECT m.metadata FROM memories m JOIN scopes s ON s.scope=m.scope '
            'WHERE m.active=1 AND m.lifecycle=\'active\' AND m.commit_seq<=s.published_seq AND m.created_at_ms>=?',(since_ms,))
        result=aggregate(json.loads(row[0]) for row in rows)
        return dict(result,since_ms=since_ms,read_only=True)
    finally:db.close()

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Read-only published-image degradation rates; no source content output.')
    parser.add_argument('--database',required=True)
    parser.add_argument('--since-ms',type=int,default=0)
    parser.add_argument('--timeout',type=float,default=30)
    args=parser.parse_args()
    print(json.dumps(audit(args.database,since_ms=args.since_ms,timeout=args.timeout),allow_nan=False))
