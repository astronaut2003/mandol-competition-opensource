"""Summarize streamed local soak metrics and write a measurement report."""
import json
from pathlib import Path
from qualification import utc, atomic_text, save


def report(output,summary,audit=None):
    text=['# Local soak results','',f"Status: **{summary['status']}**",
        f"Start (UTC): {summary['start']}",f"Target end (UTC): {summary['target_end']}",
        f"PID: {summary['pid']}; tmux: `{summary['tmux']}`",
        f"Implementation: `{summary['implementation_commit']}`",
        f"Observed seconds: {summary.get('elapsed_seconds',0):.1f}; checks: {summary['checks']}; errors: {summary['errors']}",
        '', 'Real provider request counts: `'+json.dumps(summary['external_requests'])+'`',
        'Real component Search counts: `'+json.dumps(summary['component_requests'])+'`',
        '', 'Full streamed metrics: `metrics.jsonl`; canaries: `canaries.jsonl`; events: `events.jsonl`.',
        'Call `summarize(output)` with the measurement directory to update this report.',
        'An incomplete run is never PASS. This is synthetic participant traffic, not official Smoke/Full.']
    if audit:
        text+=['','Resource audit: `resource-audit.json`',
            'Component A/B/C distribution: `'+json.dumps(audit['component_distribution'])+'`',
            'API A/B/C distribution: `'+json.dumps(audit['api_distribution'])+'`',
            'Post-warmup alerts: `'+json.dumps(audit['violations'])+'`']
    atomic_text(output/'Report.md','\n\n'.join(text)+'\n')

def summarize(output):
    summary=json.loads((output/'status.json').read_text())
    # Read streamed metrics without retaining full 12h/24h records in memory.
    first={};last={};high={};warm_end={};rows=0;max_queue={};violations=[]
    api_first={};api_last={};inflight_high={}
    metrics_path=output/'metrics.jsonl'
    if not metrics_path.exists():metrics_path.touch(mode=0o600)
    with metrics_path.open() as stream:
        for line in stream:
            # An inspector may reach an append in progress. Review its complete
            # newline-delimited sample on the next pass; never accept half JSON.
            if not line.endswith('\n'):
                break
            row=json.loads(line);rows+=1
            values={}
            for name,value in row['application'].items():
                if name!='system':
                    for field in ('rss_kib','fd_count'):values[f'app/{name}/{field}']=value[field]
            for node in row['nodes']:
                if 'gpu' in node:values[f'gpu/{node["host"]}/memory_mib']=node['gpu']['memory_mib']
                for name in ('querit','bge'):
                    if 'parent' in node.get(name,{}):
                        for p in [node[name]['parent'],*node[name]['children']]:
                            for field in ('rss_kib','fd_count'):
                                values[f'node/{node["host"]}/{name}/{p["pid"]}/{field}']=p[field]
            for key,value in values.items():
                first.setdefault(key,value);last[key]=value;high[key]=max(high.get(key,value),value)
                if row['elapsed_seconds']<=3600:warm_end[key]=value
            for track,m in row['metrics'].items():
                for key,value in m.items():
                    if 'queue_depth{' in key:max_queue[key]=max(max_queue.get(key,0),value)
                    if key.startswith('reranker_replica_requests{'):
                        api_first.setdefault(key,value);api_last[key]=value
                    if key.startswith('reranker_replica_inflight{'):
                        inflight_high[key]=max(inflight_high.get(key,0),value)
            for name,peaks in row.get('component_inflight_peak',{}).items():
                for replica,value in peaks.items():
                    key=f'component/{name}/{replica}'
                    inflight_high[key]=max(inflight_high.get(key,0),value)
            for name,value in row.get('component_queue_peak',{}).items():
                key=f'component/{name}/queue'
                max_queue[key]=max(max_queue.get(key,0),value)
    growth={key:last[key]-warm_end.get(key,first[key]) for key in last}
    for key,value in growth.items():
        bound=128 if key.endswith('/fd_count') else 512 if key.endswith('/memory_mib') else 512*1024
        if value>bound:violations.append({'metric':key,'post_warmup_growth':value,'audit_threshold':bound})
    balance={}
    for name,pool in summary.get('component_pool',{}).items():
        counts=[r['requests'] for r in pool['replicas']]
        ratio=max(counts)/min(counts) if min(counts,default=0)>0 else None
        balance[name]={'requests':counts,'max_min_ratio':ratio,'ready_replicas':pool['ready_replicas']}
        if ratio is None or ratio>1.25 or pool['ready_replicas']!=3:
            violations.append({'metric':'component/'+name+'/balance','value':balance[name]})
    api_balance={}
    for track in ('textual','multimodal'):
        counts=[api_last.get(f'reranker_replica_requests{{track="{track}",replica="{r}"}}',0)
            -api_first.get(f'reranker_replica_requests{{track="{track}",replica="{r}"}}',0) for r in ('A','B','C')]
        ratio=max(counts)/min(counts) if min(counts,default=0)>0 else None
        api_balance[track]={'requests_since_first_sample':counts,'max_min_ratio':ratio,
            'gate':'at least 30 low-frequency API canaries before balance qualification'}
        if sum(counts)>=30 and (ratio is None or ratio>1.25):
            violations.append({'metric':'api/'+track+'/balance','value':api_balance[track]})
    audit={'timestamp':utc(),'samples':rows,'first':first,'last':last,'high':high,
        'post_warmup_growth':growth,'warmup_seconds':3600,'max_queue_depth':max_queue,
        'component_distribution':balance,'api_distribution':api_balance,
        'inflight_high':inflight_high,'violations':violations,
        'limit':'fixed-load growth thresholds are audit gates, not proof against power-loss or untested workloads'}
    if summary['status']=='COMPLETED_AWAITING_RESOURCE_AUDIT':
        summary['status']='PASS' if summary['errors']==0 and not violations and summary['elapsed_seconds']>=43200 else 'FAIL'
        save(output/'status.json',summary)
    save(output/'resource-audit.json',audit)
    report(output,summary,audit)
    return summary,audit
