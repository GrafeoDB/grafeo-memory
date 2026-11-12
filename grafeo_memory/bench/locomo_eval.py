"""LOCoMo evaluation harness for grafeo-memory.

Pipeline per conversation (sample) of snap-research/locomo data/locomo10.json:

  ingest : each session is added via AsyncMemoryManager.add(infer=True) with the
           session date injected as a system message and session_id="D<n>"
           (matching the evidence ids "D<n>:<m>" in the QA annotations).
  qa     : each question is answered from the top-K retrieved memories
           (memory.search), answered by a strong LLM, and scored with the
           official LoCoMo F1 metric plus an optional binary LLM judge.
  score  : aggregate per-category and overall (categories 1-4 = the comparable
           set of 1,540 questions; category 5 adversarial reported separately).

Also reports evidence-session recall: the fraction of the question's evidence
sessions ("D<n>") that appear in the session provenance of all --retrieval-k
retrieved memories (retrieval depth, not the --k memories the answerer sees).

Usage:
  python -m grafeo_memory.bench.locomo_eval all --run pilot --convs 0,1 --limit 30
  python -m grafeo_memory.bench.locomo_eval ingest --run full
  python -m grafeo_memory.bench.locomo_eval qa --run full --k 60
  python -m grafeo_memory.bench.locomo_eval score --run full

Requires MISTRAL_API_KEY in .env (ingest/extraction, answer, judge, embeddings).
Results land in .benchmarks/locomo_runs/<run>/.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import string
import time
import urllib.request
from collections import Counter
from pathlib import Path

from grafeo_memory.bench.common import (
    DIRECT_TOKENS,
    Cache,
    UsageTracker,
    add_set_argument,
    apply_overrides,
    call_mistral,
    data_dir,
    default_cache,
    ingest_complete,
    load_env,
    remove_db,
    require_bench_deps,
    suite_runs,
    with_retry,
)

load_env()  # .env wins over any ambient env var (e.g. a stale API key)
require_bench_deps()

from mistralai.client import Mistral
from nltk.stem.porter import PorterStemmer

from grafeo_memory import (
    AsyncMemoryManager,
    CacheConfig,
    Engine,
    EscalationSignal,
    Granularity,
    MemoryCache,
    MemoryConfig,
    MistralEmbedder,
    PipelineConfig,
    StageConfig,
)

DATA_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DATA_PATH = data_dir("locomo_data") / "locomo10.json"
MEM0_PROMPTS_PATH = data_dir("locomo_data") / "mem0_prompts.py"
RUNS_DIR = suite_runs("locomo")

CACHE_PATH = str(default_cache("locomo"))
# Mem0's LoCoMo answer/judge prompts, vendored verbatim, so our judge score is
# protocol-comparable with their published 92.5 (gpt-5 answerer+judge, top_200).
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("mem0_locomo_prompts", MEM0_PROMPTS_PATH)
mem0p = _ilu.module_from_spec(_spec) if MEM0_PROMPTS_PATH.exists() else None
if mem0p is not None:
    _spec.loader.exec_module(mem0p)

CAT_NAMES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}

DATE_FORMATS = ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %B %Y", "%d %B, %Y", "%d %B %Y", "%Y-%m-%d")


def parse_session_date(s: str):
    from datetime import datetime

    s = (s or "").strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


ps = PorterStemmer()

# ---------------------------------------------------------------------------
# Official LoCoMo metrics (copied from snap-research/locomo task_eval/evaluation.py)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    s = s.replace(",", "")

    def remove_articles(text):
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def f1_multi(prediction: str, ground_truth: str) -> float:
    import numpy as np

    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(np.mean([max(f1_score(p, g) for p in predictions) for g in ground_truths]))


def locomo_score(category: int, prediction: str, answer) -> float:
    if isinstance(answer, list):
        answer = "; ".join(str(x) for x in answer)
    answer = str(answer)
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category in (2, 3, 4):
        return f1_score(prediction, answer)
    if category == 1:
        return f1_multi(prediction, answer)
    if category == 5:
        low = prediction.lower()
        return 1.0 if ("no information available" in low or "not mentioned" in low) else 0.0
    raise ValueError(category)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BETTER_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Extract facts, entities, and relationships from conversation text.

Extract three things:
1. FACTS: discrete, self-contained factual statements.
   - Each fact is one or two sentences; group closely related details about the same topic.
   - Replace pronouns with the actual names.
   - Prefer third-person statements ("alice likes hiking", not "I like hiking").
   - PRESERVE exact dates, times, numbers, names, places and quotes verbatim — never round, approximate or drop them. If the text says "yesterday" or "last week", keep that wording (the session date is stated in the text).
   - Extract preferences, biographical details, relationships, events, plans and opinions.
   - Do NOT extract greetings, pleasantries, filler, or the user_id field.
   - If the text contains nothing memorable, return an empty list.
2. ENTITIES: the named entities appearing in the facts (people, organizations, places, creative works, animals, ...).
3. RELATIONSHIPS: source -> relation_type -> target edges between the entities.
"""

ANSWER_PROMPT = """Below is a conversation between {a} and {b}. The conversation took place over multiple days.
You have access to memories extracted from this conversation, listed below (most relevant first). Each memory is tagged with the session id and the date of that session.

{mems}

QUESTION: {q}

Instructions: answer the QUESTION using only the memories above. Use the session dates to resolve relative times ("yesterday", "last week", "previous year" -> concrete dates). If the question asks "when", always give the concrete date. Reply with just the answer — a short phrase (a date, a name, a number, a short list). No explanation. Only if the memories truly do not contain the answer at all, reply exactly: "no information available".
ANSWER:"""

JUDGE_PROMPT = """You are an impartial judge evaluating a question-answering system.
Given the QUESTION, the GOLD answer, and the MODEL answer, decide whether the MODEL answer is correct: the MODEL answer must contain the same information as the GOLD answer. Exact wording is not required; if the GOLD answer is a short phrase (a date, a name, a list of items), a MODEL answer that contains that phrase or those items is CORRECT.
Reply with exactly one word: CORRECT or INCORRECT.

QUESTION: {q}
GOLD ANSWER: {gold}
MODEL ANSWER: {pred}
VERDICT:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_data() -> list[dict]:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading LoCoMo dataset -> {DATA_PATH}")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return json.load(open(DATA_PATH))


def session_keys(conv: dict) -> list[tuple[int, str, str]]:
    out = []
    for k in conv:
        m = re.fullmatch(r"session_(\d+)", k)
        if m:
            n = int(m.group(1))
            out.append((n, k, conv.get(f"session_{n}_date_time", "")))
    return sorted(out)


# ---------------------------------------------------------------------------
# Pipeline engine config + usage accounting (decision-model matrix rounds)
# ---------------------------------------------------------------------------


def build_pipeline(args) -> PipelineConfig:
    """PipelineConfig from CLI args. Defaults are LLM everywhere (baseline)."""
    return PipelineConfig(
        memories=StageConfig(
            engine=Engine(args.memories_engine),
            granularity=Granularity(args.granularity),
            confidence_threshold=args.conf_threshold,
            escalation_signal=EscalationSignal(args.escalation),
        ),
        relations=StageConfig(engine=Engine(args.relations_engine)),
        search=StageConfig(engine=Engine(args.search_engine), rerank_blend_weight=args.blend),
        question_pack=args.question_pack or None,
        temporal_precheck=getattr(args, "temporal_precheck", False),
        ontology=getattr(args, "ontology", False),
        bitemporal=getattr(args, "bitemporal", False),
        episodes=getattr(args, "episodes", False),
        compile=getattr(args, "compile", False),
        taxonomy=getattr(args, "taxonomy", False),
        policy=getattr(args, "policy", False),
        bucket=getattr(args, "bucket", False),
    )


class CountingDecisionModel:
    """Wraps a DecisionModel to count calls/questions/tokens for the run's usage.json."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = 0
        self.questions = 0
        self.input_tokens = 0
        self.output_tokens = 0

    async def decide_async(self, state, questions):
        self.calls += 1
        self.questions += len(questions)
        decisions = await self.inner.decide_async(state, questions)
        lu = getattr(self.inner, "last_usage", None)
        if isinstance(lu, dict):
            self.input_tokens += int(lu.get("prompt_tokens", 0) or 0)
            self.output_tokens += int(lu.get("completion_tokens", 0) or 0)
        return decisions

    def decide(self, state, questions):
        from grafeo_memory.common.compat import run_sync

        return run_sync(self.decide_async(state, questions))

    def stats(self) -> dict:
        return {
            "provider": type(self.inner).__name__,
            "calls": self.calls,
            "questions": self.questions,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


# Phases
# ---------------------------------------------------------------------------


async def ingest_conversation(args, client: Mistral, sample: dict, run_dir: Path, cache: MemoryCache) -> dict:
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    a, b = conv["speaker_a"], conv["speaker_b"]
    db_path = run_dir / f"db_{sample_id}.grafeo"
    remove_db(db_path)  # a partial or errored ingest restarts from an empty db
    config = apply_overrides(
        MemoryConfig(
            db_path=str(db_path),
            user_id=sample_id,
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
            usage_callback=getattr(args, "_tracker", None),
        ),
        getattr(args, "set", None),
    )
    embedder = MistralEmbedder(client)
    log: dict = {"sample_id": sample_id, "sessions": [], "errors": []}
    t0 = time.perf_counter()
    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:
        for n, key, date in session_keys(conv):
            turns = conv[key]
            messages = [{"role": "system", "content": f"This session took place at: {date}"}]
            for t in turns:
                role = "user" if t["speaker"] == a else "assistant"
                messages.append({"role": role, "name": t["speaker"], "content": t["text"]})
            ts = time.perf_counter()
            try:
                res = await mem.add(messages, session_id=f"D{n}", infer=True)
                usage = res.usage
                log["sessions"].append(
                    {
                        "session": f"D{n}",
                        "date": date,
                        "turns": len(turns),
                        "events": [
                            {
                                "action": e.action.value,
                                "text": e.text[:200],
                                "engine": e.engine,
                                "confidence": e.confidence,
                                "rung": e.rung,
                                "open_decision": e.open_decision_id,
                            }
                            for e in res
                        ],
                        "input_tokens": getattr(usage, "input_tokens", None),
                        "output_tokens": getattr(usage, "output_tokens", None),
                        "add_s": round(time.perf_counter() - ts, 2),
                    }
                )
            except Exception as e:
                log["errors"].append({"session": f"D{n}", "error": repr(e)})
        if getattr(args, "compile", False):
            cres = mem.acompile() if hasattr(mem, "acompile") else mem.compile()
            if asyncio.iscoroutine(cres):
                cres = await cres
            log["compile"] = json.loads(json.dumps(cres, default=lambda o: getattr(o, "__dict__", str(o))))
        stats = mem.stats()
        if asyncio.iscoroutine(stats):
            stats = await stats
        log["stats"] = stats.__dict__ if hasattr(stats, "__dict__") else str(stats)
    log["ingest_s"] = round(time.perf_counter() - t0, 1)
    with open(run_dir / f"ingest_{sample_id}.json", "w") as f:
        json.dump(log, f, indent=1)
    print(
        f"[ingest] {sample_id}: {len(log['sessions'])} sessions, stats={log['stats']}, "
        f"errors={len(log['errors'])}, {log['ingest_s']}s"
    )
    return log


async def qa_conversation(
    args, client: Mistral, sample: dict, run_dir: Path, ans_cache: Cache, judge_cache: Cache, cache: MemoryCache
) -> list[dict]:
    sample_id = sample["sample_id"]
    conv = sample["conversation"]
    a, b = conv["speaker_a"], conv["speaker_b"]
    db_path = run_dir / f"db_{sample_id}.grafeo"
    if not db_path.exists():
        raise RuntimeError(f"run ingest first for {sample_id}")
    config = apply_overrides(
        MemoryConfig(
            db_path=str(db_path),
            user_id=sample_id,
            embedding_dimensions=1024,
            reconciliation_threshold=args.recon_threshold,
            enable_bitemporal=args.bitemporal,
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
            usage_callback=getattr(args, "_tracker", None),
        ),
        getattr(args, "set", None),
    )
    embedder = MistralEmbedder(client)
    sem = asyncio.Semaphore(args.workers)
    records: list[dict] = []
    qas = sample["qa"]
    if args.limit:
        qas = qas[: args.limit]
    sess = session_keys(conv)
    sess_dates = {f"D{n}": date for n, _key, date in sess}
    sess_iso = {}
    sess_num = {}
    parsed = []
    for n, _key, date in sess:
        sess_num[f"D{n}"] = n
        dt = parse_session_date(date)
        if dt is not None:
            sess_iso[f"D{n}"] = dt.strftime("%Y-%m-%d")
            parsed.append(dt)
    ref_date = max(parsed).strftime("%A, %B %d, %Y") if parsed else "2023"

    async with AsyncMemoryManager(args.ingest_model, config, embedder=embedder) as mem:

        async def one(qa: dict) -> dict:
            async with sem:
                q = qa["question"]
                res = await with_retry(lambda: mem.search(q, k=args.retrieval_k, rerank=False, diverse=args.diverse))
                retrieval = [
                    {
                        "memory_id": r.memory_id,
                        "text": r.text,
                        "score": round(r.score, 4),
                        "session_id": r.session_id,
                        "source": r.source,
                    }
                    for r in res
                ]
                ev_sessions = {ev.split(":")[0] for ev in qa.get("evidence", [])}
                got_sessions = {r["session_id"] for r in retrieval if r["session_id"]}
                ev_recall = (len(ev_sessions & got_sessions) / len(ev_sessions)) if ev_sessions else None

                seen_texts: set[str] = set()
                search_results = []
                for r in retrieval[: args.k]:
                    if r["text"] in seen_texts:
                        continue
                    seen_texts.add(r["text"])
                    sid = r.get("session_id") or ""
                    search_results.append(
                        {
                            "memory": r["text"],
                            "created_at": sess_iso.get(sid, ""),
                            "_sess": sess_num.get(sid, 10**9),
                        }
                    )
                search_results.sort(key=lambda x: x["_sess"])  # chronological, oldest first
                if mem0p is not None and args.mem0_protocol:
                    prompt = mem0p.get_answer_generation_prompt(q, search_results, reference_date=ref_date)
                else:
                    mems_text = "\n".join(f"{i + 1}. {x['memory']}" for i, x in enumerate(search_results))
                    prompt = ANSWER_PROMPT.format(a=a, b=b, mems=mems_text, q=q)
                akey = hashlib.sha1((args.answer_model + prompt).encode()).hexdigest()
                pred = ans_cache.get(akey)
                if pred is None:
                    pred = await asyncio.to_thread(call_mistral, client, args.answer_model, prompt, 300, cache=cache)
                    ans_cache.put(akey, pred)
                return {
                    "question": q,
                    "answer": qa["answer"] if "answer" in qa else "no information available",
                    "adversarial_answer": qa.get("adversarial_answer"),
                    "category": qa["category"],
                    "evidence": qa.get("evidence", []),
                    "retrieval": retrieval,
                    "evidence_recall_at_k": ev_recall,
                    "prediction": pred,
                }

        t0 = time.perf_counter()
        raw = await asyncio.gather(*[one(qa) for qa in qas], return_exceptions=True)
        records = []
        for qa, r in zip(qas, raw):
            if isinstance(r, Exception):
                print(f"[qa] {sample_id}: retrying failed question after {type(r).__name__}")
                try:
                    r = await with_retry(lambda: one(qa))
                except Exception as e:  # keep the conversation's other answers
                    print(f"[qa] {sample_id}: skipping question after retry failed: {e!r}", flush=True)
                    continue
            records.append(r)
        print(f"[qa] {sample_id}: {len(records)} questions in {time.perf_counter() - t0:.0f}s")

    # scoring (f1 + optional llm judge)
    for rec in records:
        rec["f1"] = round(locomo_score(rec["category"], rec["prediction"], rec["answer"]), 4)

    async def judge(rec: dict) -> None:
        if args.judge != "llm":
            return
        if mem0p is not None and args.mem0_protocol:
            jprompt = mem0p.get_judge_prompt(rec["category"], rec["question"], str(rec["answer"]), rec["prediction"])
        else:
            jprompt = JUDGE_PROMPT.format(q=rec["question"], gold=rec["answer"], pred=rec["prediction"])
        jkey = hashlib.sha1((args.judge_model + jprompt).encode()).hexdigest()
        verdict = judge_cache.get(jkey)
        if verdict is None:
            async with sem:
                verdict = await asyncio.to_thread(call_mistral, client, args.judge_model, jprompt, 200, cache=cache)
            judge_cache.put(jkey, verdict)
        rec["judge"] = verdict
        m = re.search(r'"label"\s*:\s*"(CORRECT|WRONG)"', verdict, re.IGNORECASE)
        label = (m.group(1) if m else ("CORRECT" if verdict.upper().startswith("CORRECT") else "WRONG")).upper()
        rec["judge_label"] = label
        rec["judge_score"] = 1.0 if label == "CORRECT" else 0.0

    await asyncio.gather(*[judge(rec) for rec in records])

    with open(run_dir / f"qa_{sample_id}.json", "w") as f:
        json.dump(records, f, indent=1)
    return records


def score_run(args, run_dir: Path) -> dict:
    samples = ensure_data()
    per_conv = {}
    all_recs = []
    for sample in samples:
        sid = sample["sample_id"]
        p = run_dir / f"qa_{sid}.json"
        if p.exists():
            recs = json.load(open(p))
            all_recs.extend(recs)
            per_conv[sid] = len(recs)

    def agg(recs: list[dict], key: str) -> dict:
        cats = {}
        for c in (1, 2, 3, 4, 5):
            sub = [r[key] for r in recs if r["category"] == c and key in r]
            if sub:
                cats[f"{c} ({CAT_NAMES[c]})"] = round(sum(sub) / len(sub), 4)
        main = [r[key] for r in recs if r["category"] in (1, 2, 3, 4) and key in r]
        out = {
            "per_category": cats,
            "overall_cats_1_4": round(sum(main) / len(main), 4) if main else None,
            "n": len(main),
        }
        allr = [r[key] for r in recs if key in r]
        out["overall_all_cats"] = round(sum(allr) / len(allr), 4) if allr else None
        return out

    summary = {
        "run": run_dir.name,
        "k": args.k,
        "retrieval_k": args.retrieval_k,  # evidence recall is measured at this depth
        "per_conversation_n": per_conv,
        "f1": agg(all_recs, "f1"),
        "evidence_recall": agg(
            [{**r, "evidence_recall_at_k": r["evidence_recall_at_k"] or 0.0} for r in all_recs], "evidence_recall_at_k"
        ),
    }
    if args.judge == "llm":
        summary["llm_judge"] = agg(all_recs, "judge_score")
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


# ---------------------------------------------------------------------------


async def run_all(args) -> None:
    samples = ensure_data()
    run_dir = RUNS_DIR / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(run_dir / "args.json", "w"), indent=1, default=str)
    client = Mistral()
    conv_ids = [int(c) for c in args.convs.split(",")] if args.convs != "all" else list(range(len(samples)))
    selected = [samples[i] for i in conv_ids]
    if args.phase in ("qa", "all") and args.mem0_protocol and mem0p is None:
        raise SystemExit(
            f"mem0 protocol prompts not found at {MEM0_PROMPTS_PATH}; vendor them there or pass --no-mem0-protocol"
        )

    cache = MemoryCache(
        CacheConfig(
            persist_path=args.cache_path,
            max_entries=2_000_000,  # the byte cap bounds memory; never evict replayable calls by count
            max_memory_bytes=512 * 1024 * 1024,
            max_storage_bytes=4 * 1024 * 1024 * 1024,
        ),
        persist_path=args.cache_path,
    )

    # Decision model for non-LLM pipeline stages (matrix rounds). A live instance is
    # built once per run and shared by every conversation; it carries the shared
    # cache so decision calls persist to llm_cache.jsonl (kind "decision").
    args._tracker = UsageTracker()
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
        raise SystemExit("non-LLM pipeline engine requires --decision-model (e.g. jev)")

    if args.phase in ("ingest", "all"):
        # skip conversations whose ingest finished without errors (resumable retries);
        # the rest restart from an empty db, so re-adding never duplicates memories
        todo = [s for s in selected if not ingest_complete(run_dir / f"ingest_{s['sample_id']}.json")]
        if len(todo) < len(selected):
            print(f"[ingest] skipping {len(selected) - len(todo)} already-ingested conversations", flush=True)
        # conversations are independent (separate db files) -> ingest in parallel
        results = await asyncio.gather(
            *[ingest_conversation(args, client, s, run_dir, cache) for s in todo], return_exceptions=True
        )
        for s, r in zip(todo, results):
            if isinstance(r, Exception):
                print(f"[ingest] {s['sample_id']} FAILED: {r!r}", flush=True)
    if args.phase in ("qa", "all"):
        ans_cache = Cache(run_dir / "answers_cache.json")
        judge_cache = Cache(run_dir / "judge_cache.json")
        results = await asyncio.gather(
            *[qa_conversation(args, client, s, run_dir, ans_cache, judge_cache, cache) for s in selected],
            return_exceptions=True,
        )
        for s, r in zip(selected, results):
            if isinstance(r, Exception):
                print(f"[qa] {s['sample_id']} FAILED: {r!r}", flush=True)
    if args.phase in ("score", "all"):
        score_run(args, run_dir)
    cs = cache.stats()
    print(
        f"[cache] hits={cs['hits']} misses={cs['misses']} hit_rate={cs['hit_rate']} entries={cs['entries']} "
        f"bytes={cs['bytes']} storage_bytes={cs['storage_bytes']} per_kind={cs['per_kind']}"
    )
    with open(run_dir / "cache_stats.json", "w") as f:
        json.dump(cs, f, indent=1)
    usage_out = {"llm": args._tracker.stats(), "mistral_direct": DIRECT_TOKENS}
    if args._dm is not None:
        usage_out["decision_model"] = args._dm.stats()
    with open(run_dir / "usage.json", "w") as f:
        json.dump(usage_out, f, indent=1)
    print(f"[usage] {json.dumps(usage_out)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["ingest", "qa", "score", "all"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--convs", default="all", help="comma-separated indices or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="cap questions per conversation (0 = all)")
    ap.add_argument("--k", type=int, default=60, help="memories fed to the answerer")
    ap.add_argument("--retrieval-k", type=int, default=150, help="memories retrieved and stored")
    ap.add_argument("--min-score", type=float, default=0.0)
    ap.add_argument("--diverse", action="store_true", help="MMR diversity in search")
    ap.add_argument("--graph-depth", type=int, default=1)
    ap.add_argument("--bitemporal", action="store_true")
    ap.add_argument("--recon-threshold", type=float, default=0.3)
    ap.add_argument("--rrf-k", type=int, default=60, help="RRF constant for hybrid BM25+vector search")
    ap.add_argument(
        "--mem0-protocol", action="store_true", default=True, help="use mem0's LoCoMo answer/judge prompts (default on)"
    )
    ap.add_argument("--no-mem0-protocol", dest="mem0_protocol", action="store_false")
    ap.add_argument("--ingest-model", default="mistral:mistral-small-latest")
    ap.add_argument("--answer-model", default="mistral-small-latest")
    ap.add_argument("--judge", choices=["f1", "llm"], default="llm")
    ap.add_argument("--judge-model", default="mistral-small-latest")
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="questions in parallel per conversation; conversations run in parallel too",
    )
    ap.add_argument("--cache-path", default=CACHE_PATH)
    # --- pipeline engine matrix (decision-model rounds) ---
    ap.add_argument("--decision-model", default="", help="registered decision model, e.g. jev / fake")
    ap.add_argument("--memories-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--relations-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--search-engine", default="llm", choices=["llm", "decision", "hybrid"])
    ap.add_argument("--granularity", default="per_fact", choices=["per_fact", "pairwise"])
    ap.add_argument("--conf-threshold", type=float, default=0.75)
    ap.add_argument("--escalation", default="confidence", choices=["confidence", "prob_gap", "combined"])
    ap.add_argument("--question-pack", default="", choices=["", "per_fact", "pairwise"])
    ap.add_argument("--blend", type=float, default=0.5, help="search-stage decision/rerank blend weight")
    # --- Layer D pre-model switches (pure code, zero extra model calls) ---
    ap.add_argument(
        "--temporal-precheck",
        action="store_true",
        help="pre-model temporal rules: disjoint-validity facts skip the engine",
    )
    ap.add_argument(
        "--ontology", action="store_true", help="entity normalization rungs 1-2 + controlled relation-type validation"
    )
    # --- Composable layer toggles (map 1:1 onto PipelineConfig) ---
    ap.add_argument("--compile", action="store_true", help="Layer E: run memory.compile() after ingest")
    ap.add_argument("--taxonomy", action="store_true", help="Layer B: faceted-taxonomy scoping")
    ap.add_argument("--policy", action="store_true", help="Layer F: policy cascade rule packs")
    ap.add_argument("--bucket", action="store_true", help="Layer G: open-decisions bucket")
    add_set_argument(ap)
    args = ap.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
