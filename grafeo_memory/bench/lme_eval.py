"""LongMemEval (S, cleaned) evaluation harness for grafeo-memory.

Protocol (matching the official LongMemEval QA evaluation):
  ingest : for each of the N questions, a fresh database (db_<question_id>.grafeo);
           every haystack session is added via AsyncMemoryManager.add(infer=True)
           with the session date injected as a system message and
           session_id=haystack_session_ids[i], user_id=question_id.
  qa     : each question is answered from the top-K retrieved memories
           (mem.search, user_id=question_id), answered by a strong LLM, and judged
           with the OFFICIAL per-category LongMemEval judge prompt (yes/no),
           vendored in lme_data/lme_judge_prompts.py.
  score  : accuracy per question_type + overall.

Pipeline-engine args (--decision-model, --memories-engine, ...) are identical to
locomo_eval.py and share its helpers; every LLM/decision call goes through the
shared cache (.benchmarks/locomo_runs/llm_cache.jsonl), so reruns and config
sweeps only pay for genuinely new calls.

Usage:
  python -m grafeo_memory.bench.lme_eval all --run lme_r0 --n 100
  python -m grafeo_memory.bench.lme_eval all --run lme_r1 --n 100 --decision-model jev --memories-engine decision
  python -m grafeo_memory.bench.lme_eval score --run lme_r0

Requires MISTRAL_API_KEY and (for decision engines) TYPESAFE_API_KEY in .env.
Results land in .benchmarks/lme_runs/<run>/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util as _ilu
import json
import re
import time
from pathlib import Path

from grafeo_memory.bench.common import data_dir, default_cache, load_env, suite_runs

load_env()

from mistralai.client import Mistral

from grafeo_memory import (
    AsyncMemoryManager,
    CacheConfig,
    MemoryConfig,
    MistralEmbedder,
)
from grafeo_memory.bench.common import DIRECT_TOKENS, Cache, UsageTracker, call_mistral, with_retry
from grafeo_memory.bench.locomo_eval import BETTER_EXTRACTION_PROMPT, CountingDecisionModel, build_pipeline

CACHE_PATH = str(default_cache("locomo"))
RUNS_DIR = suite_runs("locomo")

LME_DATA = data_dir("lme_data") / "longmemeval_s_cleaned.json"
LME_RUNS = suite_runs("lme")

# LongMemEval's judge prompts, vendored next to the dataset (not shipped). Missing
# file: the harness still imports (for --help and the bench tests) and the judge
# step raises a clear error.
_LME_JUDGE_PATH = data_dir("lme_data") / "lme_judge_prompts.py"
_spec = _ilu.spec_from_file_location("lme_judge", _LME_JUDGE_PATH)
lme_judge = _ilu.module_from_spec(_spec) if _LME_JUDGE_PATH.exists() else None
if lme_judge is not None:
    _spec.loader.exec_module(lme_judge)

ANSWER_PROMPT = """\
You are answering a question about a user, using memories extracted from their past conversations.
The question was asked on: {qdate}

Memories (most relevant first; each tagged with its session id and the date of that session):
{mems}

QUESTION: {q}

Instructions: answer the QUESTION using only the memories above. Use the session dates to resolve relative times. Reply with just the answer — a short phrase. No explanation. Only if the memories truly do not contain the answer at all, reply exactly: "no information available".
ANSWER:"""


def load_questions() -> list[dict]:
    return json.load(open(LME_DATA, encoding="utf-8"))


def parse_date(s: str):
    from datetime import datetime

    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


async def ingest_question(args, client, item, run_dir, cache, tracker) -> dict:
    qid = item["question_id"]
    db_path = run_dir / f"db_{qid}.grafeo"
    config = MemoryConfig(
        db_path=str(db_path),
        user_id=qid,
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
    )
    embedder = MistralEmbedder(client)
    log: dict = {"question_id": qid, "sessions": 0, "errors": [], "events": 0}
    t0 = time.perf_counter()
    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:
        sessions = item["haystack_sessions"]
        dates = item.get("haystack_dates", [])
        sids = item.get("haystack_session_ids", [])
        for i, turns in enumerate(sessions):
            date = dates[i] if i < len(dates) else ""
            sid = sids[i] if i < len(sids) else f"S{i}"
            messages = [{"role": "system", "content": f"This session took place at: {date}"}]
            messages += [{"role": t["role"], "content": t["content"]} for t in turns]
            try:
                res = await mem.add(messages, user_id=qid, session_id=sid, infer=True)
                log["sessions"] += 1
                log["events"] += len(res)
            except Exception as e:
                log["errors"].append({"session": sid, "error": repr(e)[:200]})
    if getattr(args, "compile", False):
        cres = mem.compile()
        if asyncio.iscoroutine(cres):
            cres = await cres
        log["compile"] = cres.__dict__ if hasattr(cres, "__dict__") else str(cres)
    log["ingest_s"] = round(time.perf_counter() - t0, 1)
    with open(run_dir / f"ingest_{qid}.json", "w") as f:
        json.dump(log, f)
    return log


async def qa_question(args, client, item, run_dir, ans_cache: Cache, judge_cache: Cache, cache) -> dict:
    qid = item["question_id"]
    db_path = run_dir / f"db_{qid}.grafeo"
    if not db_path.exists():
        raise RuntimeError(f"run ingest first for {qid}")
    config = MemoryConfig(
        db_path=str(db_path),
        user_id=qid,
        embedding_dimensions=1024,
        reconciliation_threshold=args.recon_threshold,
        search_min_score=args.min_score,
        graph_search_depth=args.graph_depth,
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
    )
    embedder = MistralEmbedder(client)
    q = item["question"]
    qdate = item.get("question_date", "")
    dates = item.get("haystack_dates", [])
    sids = item.get("haystack_session_ids", [])
    sess_iso = {
        sids[i]: (parse_date(dates[i]).strftime("%Y-%m-%d") if i < len(dates) and parse_date(dates[i]) else "")
        for i in range(len(sids))
    }

    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:
        res = await with_retry(
            lambda: mem.search(q, k=args.retrieval_k, user_id=qid, rerank=False, diverse=args.diverse)
        )
        retrieval = [
            {"memory_id": r.memory_id, "text": r.text, "score": round(r.score, 4), "session_id": r.session_id}
            for r in res
        ]
        seen, mems = set(), []
        for r in retrieval[: args.k]:
            if r["text"] in seen:
                continue
            seen.add(r["text"])
            sid = r.get("session_id") or ""
            mems.append(f"- [{sess_iso.get(sid, '')} | {sid}] {r['text']}")
        prompt = ANSWER_PROMPT.format(qdate=qdate, q=q, mems="\n".join(mems))
        akey = hashlib.sha1((args.answer_model + prompt).encode()).hexdigest()
        pred = ans_cache.get(akey)
        if pred is None:
            pred = await asyncio.to_thread(call_mistral, client, args.answer_model, prompt, 300, cache=cache)
            ans_cache.put(akey, pred)

    if lme_judge is None:
        raise FileNotFoundError(f"LongMemEval judge prompts not found at {_LME_JUDGE_PATH}")
    jprompt = lme_judge.get_anscheck_prompt(item["question_type"], q, str(item["answer"]), pred)
    jkey = hashlib.sha1((args.judge_model + jprompt).encode()).hexdigest()
    verdict = judge_cache.get(jkey)
    if verdict is None:
        verdict = await asyncio.to_thread(call_mistral, client, args.judge_model, jprompt, 100, cache=cache)
        judge_cache.put(jkey, verdict)
    label = "yes" if re.search(r"\byes\b", verdict.lower()) else "no"
    rec = {
        "question_id": qid,
        "question_type": item["question_type"],
        "question": q,
        "answer": item["answer"],
        "prediction": pred,
        "judge": verdict,
        "judge_score": 1.0 if label == "yes" else 0.0,
        "evidence": item.get("answer_session_ids", []),
        "retrieval": retrieval,
    }
    ev_sessions = set(rec["evidence"])
    got = {r["session_id"] for r in retrieval if r.get("session_id")}
    rec["evidence_recall_at_k"] = (len(ev_sessions & got) / len(ev_sessions)) if ev_sessions else None
    return rec


def score_run(args, run_dir: Path) -> dict:
    items = {x["question_id"]: x for x in load_questions()}
    recs = []
    for p in sorted(run_dir.glob("qa_*.json")):
        recs.append(json.load(open(p)))
    per_cat: dict[str, list[float]] = {}
    for r in recs:
        per_cat.setdefault(r["question_type"], []).append(r["judge_score"])
    summary = {
        "run": run_dir.name,
        "n": len(recs),
        "overall_accuracy": round(sum(r["judge_score"] for r in recs) / len(recs), 4) if recs else None,
        "per_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(per_cat.items())},
        "evidence_recall": round(sum(r["evidence_recall_at_k"] or 0.0 for r in recs) / len(recs), 4) if recs else None,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


async def run_all(args) -> None:
    items = load_questions()
    if args.n:
        items = items[: args.n]
    run_dir = LME_RUNS / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir / "args.json", "w"), indent=1, default=str)
    client = Mistral()

    from grafeo_memory import Engine
    from grafeo_memory.common.cache import MemoryCache

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

    if args.phase in ("ingest", "all"):
        sem = asyncio.Semaphore(args.workers)
        t0 = time.perf_counter()

        async def one(item):
            async with sem:
                try:
                    log = await ingest_question(args, client, item, run_dir, cache, tracker)
                    if log["sessions"] % 10 == 0 or log["errors"]:
                        print(
                            f"[ingest] {item['question_id']}: {log['sessions']} sessions, {log['events']} events, "
                            f"errors={len(log['errors'])}, {log['ingest_s']}s",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[ingest] {item['question_id']} FAILED: {e!r}", flush=True)

        await asyncio.gather(*[one(it) for it in items])
        print(f"[ingest] done: {len(items)} questions in {time.perf_counter() - t0:.0f}s", flush=True)
    if args.phase in ("qa", "all"):
        ans_cache = Cache(run_dir / "answers_cache.json")
        judge_cache = Cache(run_dir / "judge_cache.json")
        sem = asyncio.Semaphore(args.workers)
        t0 = time.perf_counter()

        async def one(item):
            async with sem:
                try:
                    rec = await qa_question(args, client, item, run_dir, ans_cache, judge_cache, cache)
                    with open(run_dir / f"qa_{item['question_id']}.json", "w") as f:
                        json.dump(rec, f)
                except Exception as e:
                    print(f"[qa] {item['question_id']} FAILED: {e!r}", flush=True)

        await asyncio.gather(*[one(it) for it in items])
        print(f"[qa] done: {len(items)} questions in {time.perf_counter() - t0:.0f}s", flush=True)
    if args.phase in ("score", "all"):
        score_run(args, run_dir)

    cs = cache.stats()
    print(
        f"[cache] hits={cs['hits']} misses={cs['misses']} hit_rate={cs['hit_rate']} entries={cs['entries']} per_kind={cs['per_kind']}"
    )
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
    ap.add_argument("--n", type=int, default=0, help="first N questions (0 = all 500)")
    ap.add_argument("--k", type=int, default=30, help="memories fed to the answerer")
    ap.add_argument("--retrieval-k", type=int, default=100)
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--diverse", action="store_true")
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--bitemporal", action="store_true")
    ap.add_argument("--recon-threshold", type=float, default=0.3)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--ingest-model", default="mistral:mistral-small-latest")
    ap.add_argument("--answer-model", default="mistral-small-latest")
    ap.add_argument("--judge-model", default="mistral-small-latest")
    ap.add_argument("--workers", type=int, default=16)
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
    ap.add_argument("--taxonomy", action="store_true", help="Layer B: faceted-taxonomy scoping (not implemented yet)")
    ap.add_argument("--policy", action="store_true", help="Layer F: policy cascade rule packs (not implemented yet)")
    ap.add_argument("--bucket", action="store_true", help="Layer G: open-decisions bucket (not implemented yet)")
    args = ap.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
