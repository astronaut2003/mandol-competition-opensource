"""Explicit qualification harness; results go to a fresh operator-selected dir.

Never run against a serving database. All published rows here use a synthetic
vector-free fixture: these numbers qualify lexical storage only.
"""

import argparse
import json
import resource
import time
from dataclasses import replace
from pathlib import Path

from mandol_substrate.state import (
    CanonicalMemory,
    Mutation,
    RetrievalProfile,
    Scope,
    SQLiteStateStore,
)
from .text import MandolTextProcessor


def percentiles(values):
    ordered = sorted(values)
    return {
        f"p{p}_ms": ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p / 100))]
        * 1000
        for p in (50, 95, 99)
    }


def lexical_scale(output, scales=(1000, 10000, 50000, 100000), repeats=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    processor = MandolTextProcessor()
    profile = RetrievalProfile(
        "synthetic/dense",
        "fixture-1",
        3,
        "fixture-1",
        "synthetic/sparse",
        "fixture-1",
        processor.fingerprint,
    )
    results = []
    for scale in scales:
        store = SQLiteStateStore(output / f"lexical-{scale}.sqlite", profile)
        scope = Scope("lexical-benchmark", str(scale))
        began = time.monotonic()
        write_times = []
        for start in range(0, scale, 1000):
            docs = tuple(
                CanonicalMemory(
                    f"d{i:08}",
                    f"common topic{i % 100} rare{i} alpha " + "beta " * (i % 5),
                    spaces=("selected",) if i % 10 == 0 else (),
                )
                for i in range(start, min(scale, start + 1000))
            )
            t = time.monotonic()
            row = store.prepare(
                scope,
                f"r{start}",
                f"r{start}",
                Mutation(memories=docs, spaces=("selected",)),
                {},
            )
            store.publish(scope, row["mutation_seq"])
            write_times.append(time.monotonic() - t)
        queries = {}
        for kind, query, spaces in [
            ("rare", "rare33", None),
            ("common", "common", None),
            ("medium_df", "topic33", None),
            ("multi", "alpha beta topic22", None),
            ("repeated", "alpha alpha beta", None),
            ("space", "common", ["selected"]),
        ]:
            times = []
            for _ in range(repeats):
                t = time.monotonic()
                store.lexical.search(scope, query, limit=100, spaces=spaces)
                times.append(time.monotonic() - t)
            queries[kind] = {
                **percentiles(times),
                "query_plan": store.lexical.explain(scope, query, spaces=spaces),
                "posting_upper_bound": sum(
                    store.lexical.inspect_stats(
                        scope, terms=processor.query(query).split()
                    )["df"].values()
                ),
            }
        mutation = {}
        for kind, count in [
            ("append", 1),
            ("20_message_add", 20),
            ("batch_append", 128),
        ]:
            docs = tuple(
                CanonicalMemory(f"{kind}{i}", "common added alpha")
                for i in range(count)
            )
            t = time.monotonic()
            row = store.prepare(scope, kind, kind, Mutation(memories=docs), {})
            store.publish(scope, row["mutation_seq"])
            mutation[kind + "_ms"] = (time.monotonic() - t) * 1000
        m = store.get_memories(scope, ["d00000000"])["d00000000"]
        for kind, new in [
            ("update", replace(m, version=2, text_content="common changed beta")),
            ("identical", replace(m, version=3, text_content="common changed beta")),
        ]:
            t = time.monotonic()
            row = store.prepare(scope, kind, kind, Mutation(memories=(new,)), {})
            store.publish(scope, row["mutation_seq"])
            mutation[kind + "_ms"] = (time.monotonic() - t) * 1000
        t = time.monotonic()
        row = store.prepare(
            scope, "delete", "delete", Mutation(delete_uids=(m.logical_uid,)), {}
        )
        store.publish(scope, row["mutation_seq"])
        mutation["delete_ms"] = (time.monotonic() - t) * 1000
        store.lexical.verify_scope(scope)
        entry = {
            "n": scale,
            "elapsed_seconds": time.monotonic() - began,
            "batch_add": percentiles(write_times),
            "queries": queries,
            "mutations": mutation,
            "db_bytes": store.path.stat().st_size,
            "wal_bytes": Path(str(store.path) + "-wal").stat().st_size,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        store.close()
        results.append(entry)
        (output / "results.json").write_text(json.dumps(results, indent=2))
        print(
            json.dumps(
                {
                    "n": scale,
                    "seconds": entry["elapsed_seconds"],
                    "common": queries["common"]["p95_ms"],
                    "rss_kib": entry["peak_rss_kib"],
                }
            ),
            flush=True,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--scales", default="1000,10000,50000,100000")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    lexical_scale(args.output, tuple(map(int, args.scales.split(","))), args.repeats)
