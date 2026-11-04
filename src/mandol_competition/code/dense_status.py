"""Read-only aggregate diagnostics, without inferring failed jobs from lag.

An absent durable attempt is not proof that no process ever began a build.
In-progress status cannot be recovered reliably from frontier rows alone.
This reader does not open canonical memory content or call inference/Qdrant.
"""
from collections import Counter
from pathlib import Path
import sqlite3


def aggregate(path):
    path=Path(path).resolve()
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5) as db:
        db.row_factory=sqlite3.Row
        db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
        rows=db.execute("SELECT s.scope,s.canonical_seq,s.published_seq,l.lexical_seq,"
            "f.published_seq AS dense_seq,f.status,f.error_class,"
            "(SELECT count(*) FROM runtime_events e WHERE e.scope=s.scope AND e.channel='dense' AND e.operation='projection') AS projection_event_classes "
            "FROM scopes s LEFT JOIN lexical_corpus_stats l ON l.scope=s.scope "
            "LEFT JOIN projection_frontiers f ON f.scope=s.scope AND f.channel='dense'").fetchall()
        counts=Counter();errors=Counter();lag=0;lag_scopes=[]
        for row in rows:
            if not row['canonical_seq']==row['published_seq']==row['lexical_seq']:category='CANONICAL_OR_LEXICAL_NOT_ACKNOWLEDGED'
            elif row['status']=='ready' and row['dense_seq']==row['published_seq']:category='READY_COMPLETE'
            elif row['status']=='ready':category='STALE_COMPLETE_FRONTIER_AFTER_NEW_PUBLICATION'
            elif row['status']=='degraded' and row['error_class']=='ProjectionNotBuilt':category='INVALIDATED_REQUIRES_LAZY_REBUILD'
            elif row['status']=='degraded':category='RECORDED_PROJECTION_FAILURE'
            elif row['status'] is None and not row['projection_event_classes']:category='NO_DURABLE_PROJECTION_ATTEMPT_RECORD'
            else:category='UNKNOWN_FRONTIER_STATE'
            counts[category]+=1
            if row['error_class']:errors[row['error_class']]+=1
            delta=max(0,row['published_seq']-(row['dense_seq'] or 0));lag+=delta
            if delta:lag_scopes.append(row['scope'])
        # Aggregate active row count: no raw_data, text_content or user IDs leave
        # SQLite, even when scope count exceeds SQLite's parameter limit.
        active_lag=db.execute("SELECT count(*) FROM memories m JOIN scopes s ON s.scope=m.scope "
            "LEFT JOIN projection_frontiers f ON f.scope=s.scope AND f.channel='dense' "
            "WHERE s.published_seq>coalesce(f.published_seq,0) AND m.active=1 AND m.lifecycle!='deleted'").fetchone()[0]
        historical=dict(db.execute("SELECT error_class,sum(count) FROM runtime_events "
            "WHERE channel='dense' AND operation='projection' GROUP BY error_class"))
        return dict(schema='code-optional-dense-aggregate-v1',scopes=len(rows),classes=dict(counts),
            dense_error_classes=dict(errors),historical_projection_error_events=historical,
            lagging_scope_count=len(lag_scopes),publication_sequence_delta=lag,
            active_raw_rows_in_lagging_scopes=active_lag,
            current_build_in_progress='UNKNOWN: no durable in-progress state in this schema',
            lag_is_not_vector_or_failure_count=True,readonly=True,model_calls=0)


if __name__=='__main__':
    import argparse
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(aggregate(args.database),sort_keys=True))
