"""Objective offline benchmark for the grafeo-memory storage + search layer.

No LLM, no API keys, no network. Measures the layer that every memory
benchmark (LoCoMo, LongMemEval, BEAM) ultimately depends on:

  1. Insert throughput: N memory nodes with a dims-dim cosine vector index
  2. HNSW recall@k vs brute-force ground truth, swept over ef
  3. Query latency p50/p90/p99 + QPS, unfiltered and user_id-filtered
  4. On-disk footprint of the single .grafeo file (+ peak RSS)
  5. End-to-end MemoryManager.add(infer=False) / search(rerank=False) overhead
     at a smaller N, using the deterministic mock LLM/embedder from tests/

Run with the project venv:

    python -m grafeo_memory.bench.vector_search_bench
    python -m grafeo_memory.bench.vector_search_bench --n 100000 --dims 1536

Prints a JSON report to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

try:
    import resource
except ImportError:  # Windows
    resource = None

import numpy as np

from grafeo_memory.bench.common import REPO

sys.path.insert(0, str(REPO / "tests"))

import grafeo


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, round((p / 100.0) * (len(xs) - 1))))
    return xs[i]


def peak_rss_mb() -> float | None:
    """Peak RSS in MiB: ``ru_maxrss`` is KiB on Linux, bytes on macOS; None on Windows."""
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024, 1)


def dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def bench_raw_vector_layer(args) -> dict:
    rng = np.random.default_rng(42)
    n, d, k, nq = args.n, args.dims, args.k, args.queries
    x = rng.standard_normal((n, d), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = rng.standard_normal((nq, d), dtype=np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    sims = q @ x.T  # unit vectors: dot == cosine similarity
    gt = np.argsort(-sims, axis=1)[:, :k]
    users = [f"u{i % args.users}" for i in range(n)]
    gt_filtered = []
    for qi in range(nq):
        mask = np.array([users[i] == users[qi % args.users] for i in range(n)])
        sims_f = sims[qi].copy()
        sims_f[~mask] = -np.inf
        gt_filtered.append(np.argsort(-sims_f)[:k])

    tmpdir = tempfile.TemporaryDirectory(prefix="grafeo-bench-", ignore_cleanup_errors=True)
    tmp = tmpdir.name
    db_path = os.path.join(tmp, "bench.grafeo")
    db = grafeo.GrafeoDB(db_path)

    t0 = time.perf_counter()
    db.create_vector_index("Memory", "embedding", dimensions=d, metric="cosine")
    t_index = time.perf_counter() - t0

    t0 = time.perf_counter()
    for start in range(0, n, args.batch):
        stop = min(start + args.batch, n)
        props = [{"text": f"mem-{i}", "user_id": users[i], "embedding": x[i].tolist()} for i in range(start, stop)]
        db.batch_create_nodes_with_props("Memory", props)
    t_insert = time.perf_counter() - t0

    id2idx = {}
    for row in db.execute("MATCH (m:Memory) RETURN id(m) AS nid, m.text AS text"):
        id2idx[row["nid"]] = int(row["text"].split("-")[1])

    def run(ef: int | None, filtered: bool, queries: int) -> dict:
        lat: list[float] = []
        recalls: list[float] = []
        for qi in range(queries):
            qv = q[qi].tolist()
            filt = {"user_id": users[qi % args.users]} if filtered else None
            kwargs = {"filters": filt}
            if ef is not None:
                kwargs["ef"] = ef
            t1 = time.perf_counter()
            hits = db.vector_search("Memory", "embedding", qv, k, **kwargs)
            lat.append(time.perf_counter() - t1)
            got = {id2idx[h[0]] for h in hits}
            truth = set((gt_filtered[qi] if filtered else gt[qi]).tolist())
            recalls.append(len(got & truth) / k)
        total = sum(lat)
        return {
            "ef": ef,
            "filtered": filtered,
            "recall_at_k": round(sum(recalls) / len(recalls), 4),
            "p50_ms": round(percentile(lat, 50) * 1e3, 3),
            "p90_ms": round(percentile(lat, 90) * 1e3, 3),
            "p99_ms": round(percentile(lat, 99) * 1e3, 3),
            "qps": round(len(lat) / total, 1) if total > 0 else 0.0,
        }

    sweep = [
        run(None, False, nq),
        run(40, False, nq),
        run(100, False, nq),
        run(400, False, nq),
        run(None, True, nq),
        run(100, True, nq),
    ]

    db.close()
    size = dir_size(tmp)
    tmpdir.cleanup()
    return {
        "config": {"n": n, "dims": d, "k": k, "queries": nq, "users": args.users, "batch": args.batch},
        "index_create_s": round(t_index, 2),
        "insert_s": round(t_insert, 2),
        "insert_per_s": round(n / t_insert, 1),
        "db_bytes": size,
        "bytes_per_node": round(size / n, 1),
        "peak_rss_mb": peak_rss_mb(),
        "sweep": sweep,
    }


def bench_manager_e2e(args) -> dict:
    """Full MemoryManager path with mock LLM + mock embedder (no API calls)."""
    from mock_llm import MockEmbedder, make_test_model

    from grafeo_memory import MemoryConfig, MemoryManager

    n, d, k = args.e2e_n, args.dims, args.k
    texts = [f"fact number {i} about topic {i % 50}" for i in range(n)]
    model = make_test_model([{"entities": [], "relations": []}] * (args.queries + 10))
    embedder = MockEmbedder(d)
    tmpdir = tempfile.TemporaryDirectory(prefix="grafeo-e2e-", ignore_cleanup_errors=True)
    tmp = tmpdir.name
    config = MemoryConfig(db_path=os.path.join(tmp, "e2e.grafeo"), user_id="bench", embedding_dimensions=d)

    t0 = time.perf_counter()
    with MemoryManager(model, config, embedder=embedder) as mem:
        t1 = time.perf_counter()
        mem.add_batch(texts, infer=False)
        t_add = time.perf_counter() - t1

        add_lat: list[float] = []
        for i in range(100):
            t2 = time.perf_counter()
            mem.add(f"extra fact {i}", infer=False)
            add_lat.append(time.perf_counter() - t2)

        search_lat: list[float] = []
        for qi in range(args.queries):
            t2 = time.perf_counter()
            res = mem.search(f"topic {qi % 50}", k=k, rerank=False)
            search_lat.append(time.perf_counter() - t2)
            assert len(res) > 0
        t_total = time.perf_counter() - t0

    size = dir_size(tmp)
    tmpdir.cleanup()
    return {
        "config": {"n": n, "dims": d, "k": k, "queries": args.queries},
        "add_batch_s": round(t_add, 2),
        "add_batch_per_s": round(n / t_add, 1),
        "add_single_p50_ms": round(percentile(add_lat, 50) * 1e3, 3),
        "add_single_p99_ms": round(percentile(add_lat, 99) * 1e3, 3),
        "search_p50_ms": round(percentile(search_lat, 50) * 1e3, 3),
        "search_p99_ms": round(percentile(search_lat, 99) * 1e3, 3),
        "search_qps": round(len(search_lat) / sum(search_lat), 1),
        "db_bytes": size,
        "total_s": round(t_total, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--dims", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument("--e2e-n", type=int, default=5_000)
    ap.add_argument("--skip-e2e", action="store_true")
    args = ap.parse_args()

    report = {
        "grafeo": grafeo.build_info() if hasattr(grafeo, "build_info") else {},
        "vector_layer": bench_raw_vector_layer(args),
    }
    if not args.skip_e2e:
        report["manager_e2e"] = bench_manager_e2e(args)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
