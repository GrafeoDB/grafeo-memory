"""BEAM (Beyond a Million Tokens) evaluation harness for grafeo-memory.

Dataset: Mohammadta/BEAM on HuggingFace (splits 100K / 500K / 1M; default 100K,
20 conversations). Each row has:
  chat, list of sessions; each session is a list of messages
                      {role, content, time_anchor, question_type...}
  probing_questions, string repr of {question_type: [{question, ideal_response...}]}

Protocol:
  ingest : each conversation gets a fresh db_<conversation_id>.grafeo; every chat
           session is added via add(infer=True) with the time anchor as a system
           message and session_id "S<i>".
  qa     : every probing question is answered from the top-K retrieved memories
           (answerer LLM), then scored 0 / 0.5 / 1 by a judge LLM against the
           question's ideal_response (BEAM's response-score metric; abstention
           questions score 1 when the model correctly withholds).
  score  : mean score per question type + overall.

Shares locomo_eval.py helpers; all LLM/decision calls go through the shared cache.

Usage:
  python -m grafeo_memory.bench.beam_eval all --run beam_r0 --split 100K --convs 0,1,2
  python -m grafeo_memory.bench.beam_eval all --run beam_r1 --split 100K --convs all --decision-model jev --memories-engine decision
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import time
from pathlib import Path

from grafeo_memory.bench.common import (
    add_set_argument,
    apply_overrides,
    data_dir,
    default_cache,
    ingest_complete,
    load_env,
    remove_db,
    require_bench_deps,
    suite_runs,
)

load_env()
require_bench_deps()

from mistralai.client import Mistral

from grafeo_memory import (
    AsyncMemoryManager,
    CacheConfig,
    Engine,
    MemoryConfig,
    MistralEmbedder,
)
from grafeo_memory.bench.common import DIRECT_TOKENS, Cache, UsageTracker, call_mistral, with_retry
from grafeo_memory.bench.locomo_eval import BETTER_EXTRACTION_PROMPT, CountingDecisionModel, build_pipeline
from grafeo_memory.common.cache import MemoryCache

CACHE_PATH = str(default_cache("locomo"))

BEAM_DIR = data_dir("beam_data")
BEAM_RUNS = suite_runs("beam")
SPLIT_FILES = {"100K": BEAM_DIR / "100K.parquet", "500K": BEAM_DIR / "500K.parquet", "1M": BEAM_DIR / "1M.parquet"}

ANSWER_PROMPT = """\
You are answering a question about a user, using memories extracted from their past conversations.
The question is asked at time anchor: {anchor}

Memories (most relevant first; each tagged with its session id and date):
{mems}

QUESTION: {q}

Instructions: answer the QUESTION using only the memories above. Be concise but complete — include the specific facts, dates, names and details the question asks for. If the memories genuinely do not contain the answer, reply exactly: "no information available".
ANSWER:"""

JUDGE_PROMPT = """\
You are scoring a memory system's answer to a probing question.

QUESTION: {q}
IDEAL RESPONSE: {gold}
MODEL RESPONSE: {pred}

Assign a score:
- 1.0 if the MODEL RESPONSE fully and correctly answers the QUESTION (equivalent to the IDEAL RESPONSE; all key facts present).
- 0.5 if it is partially correct (some key facts right, some missing or wrong).
- 0.0 if it is incorrect or irrelevant.
For abstention questions (the IDEAL RESPONSE says there is no information): 1.0 if the MODEL RESPONSE correctly states the information is not available, else 0.0.

Reply with just the number: 1.0, 0.5, or 0.0
SCORE:"""


def load_split(split: str) -> list[dict]:
    import pyarrow.parquet as pq

    path = SPLIT_FILES[split]
    if not path.exists():
        url = f"https://huggingface.co/datasets/Mohammadta/BEAM/resolve/main/data/{split}-00000-of-00001.parquet"
        import urllib.request

        print(f"downloading {url} -> {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return pq.read_table(path).to_pylist()


def last_anchor(row: dict) -> str:
    """The latest session time anchor: the "now" at which the probing questions are asked."""
    anchor = ""
    for session in row["chat"]:
        for m in session:
            if m.get("time_anchor"):
                anchor = m["time_anchor"]
    return anchor


def probing_items(row: dict) -> list[dict]:
    pq_raw = row["probing_questions"]
    data = ast.literal_eval(pq_raw) if isinstance(pq_raw, str) else pq_raw
    out = []
    for qtype, qs in data.items():
        for q in qs:
            out.append(
                {"question_type": qtype, "question": q["question"], "ideal_response": q.get("ideal_response", "")}
            )
    return out


async def ingest_conv(args, client, row, run_dir, tracker) -> dict:
    cid = row["conversation_id"]
    db_path = run_dir / f"db_{cid}.grafeo"
    log_path = run_dir / f"ingest_{cid}.json"
    if ingest_complete(log_path):  # resumable: re-adding would duplicate memories
        print(f"[ingest] {cid}: already ingested, skipping", flush=True)
        return json.loads(log_path.read_text())
    remove_db(db_path)  # a partial or errored ingest restarts from an empty db
    config = apply_overrides(
        MemoryConfig(
            db_path=str(db_path),
            user_id=cid,
            embedding_dimensions=1024,
            reconciliation_threshold=args.recon_threshold,
            enable_bitemporal=args.bitemporal,
            chronological_sort=False,
            hybrid_rrf_k=args.rrf_k,
            custom_fact_prompt=BETTER_EXTRACTION_PROMPT,
            cache=CacheConfig(
                persist_path=args.cache_path,
                max_entries=2_000_000,  # the byte cap bounds memory; never evict replayable calls by count
                max_memory_bytes=512 * 1024 * 1024,
                max_storage_bytes=4 * 1024 * 1024 * 1024,
            ),
            pipeline=build_pipeline(args),
            decision_model=getattr(args, "_dm", None),
            usage_callback=tracker,
        ),
        getattr(args, "set", None),
    )
    embedder = MistralEmbedder(client)
    log = {"conversation_id": cid, "sessions": 0, "errors": []}
    t0 = time.perf_counter()
    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:
        for i, session in enumerate(row["chat"]):
            anchor = ""
            for m in session:
                if m.get("time_anchor"):
                    anchor = m["time_anchor"]
                    break
            messages = [{"role": "system", "content": f"This session took place at: {anchor}"}]
            messages += [{"role": m["role"], "content": m["content"]} for m in session]
            try:
                res = await mem.add(messages, user_id=cid, session_id=f"S{i}", infer=True)
                log["sessions"] += 1
            except Exception as e:
                log["errors"].append({"session": i, "error": repr(e)[:200]})
        if getattr(args, "compile", False):  # inside the manager: the db closes on exit
            cres = await mem.acompile()
            log["compile"] = json.loads(json.dumps(cres, default=lambda o: getattr(o, "__dict__", str(o))))
    log["ingest_s"] = round(time.perf_counter() - t0, 1)
    with open(log_path, "w") as f:
        json.dump(log, f)
    print(f"[ingest] {cid}: {log['sessions']} sessions, errors={len(log['errors'])}, {log['ingest_s']}s", flush=True)
    return log


async def qa_conv(args, client, row, run_dir, ans_cache: Cache, judge_cache: Cache, cache) -> list[dict]:
    """Answer and judge one conversation's probing questions, --workers at a time (per conversation)."""
    cid = row["conversation_id"]
    db_path = run_dir / f"db_{cid}.grafeo"
    if not db_path.exists():
        raise RuntimeError(f"run ingest first for {cid}")
    config = apply_overrides(
        MemoryConfig(
            db_path=str(db_path),
            user_id=cid,
            embedding_dimensions=1024,
            reconciliation_threshold=args.recon_threshold,
            chronological_sort=False,
            hybrid_rrf_k=args.rrf_k,
            cache=CacheConfig(
                persist_path=args.cache_path,
                max_entries=2_000_000,  # the byte cap bounds memory; never evict replayable calls by count
                max_memory_bytes=512 * 1024 * 1024,
                max_storage_bytes=4 * 1024 * 1024 * 1024,
            ),
            pipeline=build_pipeline(args),
            decision_model=getattr(args, "_dm", None),
        ),
        getattr(args, "set", None),
    )
    embedder = MistralEmbedder(client)
    sem = asyncio.Semaphore(args.workers)
    items = probing_items(row)
    anchor = last_anchor(row)
    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:

        async def one(it: dict) -> dict:
            async with sem:
                q = it["question"]
                res = await with_retry(lambda: mem.search(q, k=args.retrieval_k, user_id=cid, rerank=False))
                mems = "\n".join(f"- [{r.session_id}] {r.text}" for r in res[: args.k])
                prompt = ANSWER_PROMPT.format(anchor=anchor, q=q, mems=mems)
                akey = hashlib.sha1((args.answer_model + prompt).encode()).hexdigest()
                pred = ans_cache.get(akey)
                if pred is None:
                    pred = await asyncio.to_thread(call_mistral, client, args.answer_model, prompt, 400, cache=cache)
                    ans_cache.put(akey, pred)
                jprompt = JUDGE_PROMPT.format(q=q, gold=it["ideal_response"], pred=pred)
                jkey = hashlib.sha1((args.judge_model + jprompt).encode()).hexdigest()
                verdict = judge_cache.get(jkey)
                if verdict is None:
                    verdict = await asyncio.to_thread(call_mistral, client, args.judge_model, jprompt, 100, cache=cache)
                    judge_cache.put(jkey, verdict)
                try:
                    score = float(verdict.strip().rstrip("."))
                    score = min(max(score, 0.0), 1.0)
                except ValueError:
                    score = (
                        1.0 if verdict.strip().startswith("1") else (0.5 if verdict.strip().startswith("0.5") else 0.0)
                    )
                return {**it, "conversation_id": cid, "prediction": pred, "judge": verdict, "score": score}

        t0 = time.perf_counter()
        recs = await asyncio.gather(*[one(it) for it in items])
        print(f"[qa] {cid}: {len(recs)} probing questions in {time.perf_counter() - t0:.0f}s", flush=True)
    return list(recs)


def score_run(args, run_dir: Path, ids: list[str], failed: set[str] | frozenset[str] = frozenset()) -> dict:
    """Score the selected conversations only; ones without a fresh qa file are listed as missing."""
    recs, missing = [], []
    for cid in ids:
        p = run_dir / f"qa_{cid}.json"
        if cid in failed or not p.exists():
            missing.append(cid)
            continue
        recs.extend(json.load(open(p)))
    per_type: dict[str, list[float]] = {}
    for r in recs:
        per_type.setdefault(r["question_type"], []).append(r["score"])
    summary = {
        "run": run_dir.name,
        "split": args.split,
        "n": len(recs),
        "convs_scored": len(ids) - len(missing),
        "convs_missing": missing,
        "overall_score": round(sum(r["score"] for r in recs) / len(recs), 4) if recs else None,
        "per_type": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_type.items())},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


async def run_all(args) -> None:
    rows = load_split(args.split)
    conv_ids = [int(c) for c in args.convs.split(",")] if args.convs != "all" else list(range(len(rows)))
    selected = [rows[i] for i in conv_ids]
    run_dir = BEAM_RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir / "args.json", "w"), indent=1, default=str)
    client = Mistral()

    cache = MemoryCache(
        CacheConfig(
            persist_path=args.cache_path,
            max_entries=2_000_000,  # the byte cap bounds memory; never evict replayable calls by count
            max_memory_bytes=512 * 1024 * 1024,
            max_storage_bytes=4 * 1024 * 1024 * 1024,
        ),
        persist_path=args.cache_path,
    )
    tracker = UsageTracker()
    args._tracker = tracker
    args._dm = None
    if args.decision_model:
        if args.decision_model == "jev":
            from grafeo_memory.providers.decision.jev import JevDecisionModel

            inner = JevDecisionModel(cache=cache)
        else:
            from grafeo_memory.providers.decision import get_decision_model

            inner = get_decision_model(args.decision_model)
        args._dm = CountingDecisionModel(inner)
    non_llm = any(getattr(build_pipeline(args), s).engine != Engine.LLM for s in ("memories", "relations", "search"))
    if non_llm and args._dm is None:
        raise SystemExit("non-LLM pipeline engine requires --decision-model")

    ids = [str(r["conversation_id"]) for r in selected]
    failed: set[str] = set()
    if args.phase in ("ingest", "all"):
        sem = asyncio.Semaphore(args.workers)

        async def one(row):
            async with sem:
                await ingest_conv(args, client, row, run_dir, tracker)

        await asyncio.gather(*[one(r) for r in selected])
    if args.phase in ("qa", "all"):
        ans_cache = Cache(run_dir / "answers_cache.json")
        judge_cache = Cache(run_dir / "judge_cache.json")
        sem = asyncio.Semaphore(args.workers)

        async def one(row):
            async with sem:
                try:
                    recs = await qa_conv(args, client, row, run_dir, ans_cache, judge_cache, cache)
                    with open(run_dir / f"qa_{row['conversation_id']}.json", "w") as f:
                        json.dump(recs, f)
                except Exception as e:
                    failed.add(str(row["conversation_id"]))
                    print(f"[qa] {row['conversation_id']} FAILED: {e!r}", flush=True)

        await asyncio.gather(*[one(r) for r in selected])
    if args.phase in ("score", "all"):
        score_run(args, run_dir, ids, failed)

    cs = cache.stats()
    print(f"[cache] hits={cs['hits']} misses={cs['misses']} hit_rate={cs['hit_rate']} entries={cs['entries']}")
    usage_out = {"llm": tracker.stats(), "mistral_direct": DIRECT_TOKENS}
    if args._dm is not None:
        usage_out["decision_model"] = args._dm.stats()
    with open(run_dir / "usage.json", "w") as f:
        json.dump(usage_out, f, indent=1)
    print(f"[usage] {json.dumps(usage_out)[:600]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ingest", "qa", "score", "all"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="100K", choices=["100K", "500K", "1M"])
    ap.add_argument("--convs", default="all")
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--retrieval-k", type=int, default=100)
    ap.add_argument("--recon-threshold", type=float, default=0.3)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--bitemporal", action="store_true")
    ap.add_argument("--ingest-model", default="mistral:mistral-small-latest")
    ap.add_argument("--answer-model", default="mistral-small-latest")
    ap.add_argument("--judge-model", default="mistral-small-latest")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="conversations in parallel, and questions in parallel within each (up to workers**2 calls)",
    )
    ap.add_argument("--cache-path", default=CACHE_PATH)
    ap.add_argument("--decision-model", default="")
    ap.add_argument("--memories-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--relations-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--search-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--granularity", default="per_fact", choices=["per_fact", "pairwise"])
    ap.add_argument("--conf-threshold", type=float, default=0.75)
    ap.add_argument("--escalation", default="confidence", choices=["confidence", "prob_gap", "combined"])
    ap.add_argument("--question-pack", default="", choices=["", "per_fact", "pairwise"])
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument(
        "--temporal-precheck",
        action="store_true",
        help="pre-model temporal rules: disjoint-validity facts skip the engine",
    )
    ap.add_argument(
        "--ontology", action="store_true", help="entity normalization rungs 1-2 + controlled relation-type validation"
    )
    ap.add_argument("--compile", action="store_true", help="Layer E: run memory.compile() after ingest")
    ap.add_argument("--taxonomy", action="store_true", help="Layer B: faceted-taxonomy scoping")
    ap.add_argument("--policy", action="store_true", help="Layer F: policy cascade rule packs")
    ap.add_argument("--bucket", action="store_true", help="Layer G: open-decisions bucket")
    add_set_argument(ap)
    args = ap.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
