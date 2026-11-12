"""Config-matrix runner for the decision-model and layer-toggle benchmark rounds.

Each matrix cell is a full LOCOMO run (`python -m grafeo_memory.bench.locomo_eval all --run mx_<cell>`)
with a different pipeline config. Because every LLM/decision call goes through the
shared cache (.benchmarks/locomo_runs/llm_cache.jsonl), re-running a cell or a whole
round costs zero new calls; only genuinely new (config, prompt, state) calls hit the
APIs.

Rounds (ablation order, vary one axis at a time):
  0  baseline            LLM everywhere
  1  granularity         DECISION per_fact vs pairwise        (fixed: confidence, 0.75)
  2  escalation signal   HYBRID confidence / prob_gap / combined
  3  threshold           HYBRID 0.6 / 0.75 / 0.9
  4  engine headline     LLM vs DECISION vs HYBRID (winners fixed), all 10 conversations

Usage:
  python -m grafeo_memory.bench.matrix --round 1 --convs 0,1,2
  python -m grafeo_memory.bench.matrix --round 4 --convs all
  python -m grafeo_memory.bench.matrix --round 1 --convs 0,1,2 --dry-run
  python -m grafeo_memory.bench.matrix --report            # re-aggregate existing runs
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time

from grafeo_memory.bench.common import PY, REPO, suite_runs

RUNS = suite_runs("locomo")

# $/M tokens (assumptions for the cost column; edit DEFAULT_PRICES to change them)
DEFAULT_PRICES = {
    "mistral-small-latest": (0.2, 0.6),
    "mistral-large-latest": (2.0, 6.0),
    "jev": (0.042, 0.0),
}


def cells_for_round(round_id: int) -> list[dict]:
    base = {
        "memories_engine": "llm",
        "relations_engine": "llm",
        "search_engine": "llm",
        "granularity": "per_fact",
        "escalation": "confidence",
        "conf_threshold": 0.75,
        "decision_model": "",
        "blend": 0.5,
        "question_pack": "",
    }

    def cell(name, **kw):
        c = {**base, **kw}
        c["name"] = name
        return c

    if round_id == 0:
        return [cell("r0_baseline_llm")]
    if round_id == 1:
        return [
            cell(
                "r1_decision_perfact",
                decision_model="jev",
                memories_engine="decision",
                relations_engine="decision",
                granularity="per_fact",
            ),
            cell(
                "r1_decision_pairwise",
                decision_model="jev",
                memories_engine="decision",
                relations_engine="decision",
                granularity="pairwise",
            ),
        ]
    if round_id == 2:
        return [
            cell(
                "r2_hybrid_conf",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                escalation="confidence",
            ),
            cell(
                "r2_hybrid_probgap",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                escalation="prob_gap",
            ),
            cell(
                "r2_hybrid_combined",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                escalation="combined",
            ),
        ]
    if round_id == 3:
        return [
            cell(
                f"r3_hybrid_t{t}",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=t,
            )
            for t in (0.6, 0.75, 0.9)
        ]
    if round_id == 4:
        return [
            cell(
                "r4_llm",
            ),
            cell("r4_decision", decision_model="jev", memories_engine="decision", relations_engine="decision"),
            cell("r4_hybrid", decision_model="jev", memories_engine="hybrid", relations_engine="hybrid"),
        ]
    # --- Layer D/E/F benchmark rounds (flags wired in as layers land; off by default) ---
    # Rounds 5 and 6 are confounded: the off cell is LLM, the on cell is hybrid jev, so
    # the delta mixes engine and flag. Names are kept so existing results stay valid;
    # for the flag alone compare the on cell against r10_hyb (same engines, no flag).
    if round_id == 5:  # Layer D temporal pre-model rules
        return [
            cell("r5_temporal_off"),
            cell(
                "r5_temporal_on",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                extra_flags=["--temporal-precheck"],
            ),
        ]
    if round_id == 6:  # Layer D ontology: normalization ladder + controlled types
        return [
            cell("r6_onto_off"),
            cell(
                "r6_onto_on",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                extra_flags=["--ontology"],
            ),
        ]
    if round_id == 7:  # Layer D combined + threshold sweep
        return [
            cell(
                f"r7_d_t{t}",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=t,
                extra_flags=["--temporal-precheck", "--ontology"],
            )
            for t in (0.6, 0.75, 0.9)
        ]
    if round_id == 8:  # Layer E compile/certification on top of D
        return [
            cell(
                "r8_e_off",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                extra_flags=["--temporal-precheck", "--ontology"],
            ),
            cell(
                "r8_e_on",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                extra_flags=["--temporal-precheck", "--ontology", "--compile"],
            ),
        ]
    if round_id == 9:  # headline: full stack on all convs
        return [
            cell(
                "r9_full_t0.75",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=0.75,
                extra_flags=["--temporal-precheck", "--ontology", "--compile"],
            ),
            cell(
                "r9_full_t0.6",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=0.6,
                extra_flags=["--temporal-precheck", "--ontology", "--compile"],
            ),
        ]
    # --- OFAT sweep rounds: one factor per round,
    # off/on cells on a fixed base (hybrid jev conf 0.75 per_fact unless noted) ---
    if round_id == 10:  # pure temporal-precheck ablation, crossed with engine
        return [
            cell("r10_llm"),
            cell("r10_llm_pre", extra_flags=["--temporal-precheck"]),
            cell("r10_hyb", decision_model="jev", memories_engine="hybrid", relations_engine="hybrid"),
            cell(
                "r10_hyb_pre",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                extra_flags=["--temporal-precheck"],
            ),
        ]
    if round_id == 11:  # threshold x granularity on hybrid at full n
        return [
            cell(
                "r11_t0.6_pf",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=0.6,
            ),
            cell(
                "r11_t0.9_pf",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=0.9,
            ),
            cell(
                "r11_t0.75_pw",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                granularity="pairwise",
            ),
            cell(
                "r11_t0.6_pw",
                decision_model="jev",
                memories_engine="hybrid",
                relations_engine="hybrid",
                conf_threshold=0.6,
                granularity="pairwise",
            ),
        ]
    if round_id == 12:  # retrieval knobs on the hybrid base (r10_hyb is the reference)
        base_kw = dict(decision_model="jev", memories_engine="hybrid", relations_engine="hybrid")
        return [
            cell("r12_k100", extra_flags=["--k", "100"], **base_kw),
            cell("r12_rk300", extra_flags=["--retrieval-k", "300"], **base_kw),
            cell("r12_depth2", extra_flags=["--graph-depth", "2"], **base_kw),
            cell("r12_diverse", extra_flags=["--diverse"], **base_kw),
            cell("r12_rrf30", extra_flags=["--rrf-k", "30"], **base_kw),
        ]
    if round_id == 13:  # bitemporal storage off/on (adds temporal-annotation LLM calls)
        base_kw = dict(decision_model="jev", memories_engine="hybrid", relations_engine="hybrid")
        return [
            cell("r13_bitemp_off", **base_kw),
            cell("r13_bitemp_on", extra_flags=["--bitemporal"], **base_kw),
        ]
    # --- 0.3.0 layer rounds: one knob per cell on the llm k=100 base (the 0.3.0 LOCOMO best);
    # every cell sets its knobs with --set, so new variables need no new flags.
    k100 = ["--k", "100"]
    hyb = dict(decision_model="jev", memories_engine="hybrid", relations_engine="hybrid")

    def sets(*kv: str) -> list[str]:
        return [x for item in kv for x in ("--set", item)]

    if round_id == 14:  # Layer G: park below the floor instead of escalating (hybrid base)
        return [
            cell("r14_bucket_off", extra_flags=k100, **hyb),
            cell("r14_floor0.4", extra_flags=k100 + sets("bucket.enabled=true", "bucket.floor=0.4"), **hyb),
            cell("r14_floor0.6", extra_flags=k100 + sets("bucket.enabled=true", "bucket.floor=0.6"), **hyb),
            cell(
                "r14_floor0.6_pen",
                extra_flags=k100 + sets("bucket.enabled=true", "bucket.floor=0.6", "bucket.provisional_penalty=0.3"),
                **hyb,
            ),
        ]
    if round_id == 15:  # Layer F: rule packs
        return [
            cell("r15_policy_off", extra_flags=k100),
            cell("r15_policy_all", extra_flags=[*k100, "--policy"]),
            cell("r15_compliance_only", extra_flags=[*k100, "--policy", *sets("policy.packs=compliance")]),
        ]
    if round_id == 16:  # Layer B: scoping off / soft / filter
        return [
            cell("r16_tax_off", extra_flags=k100),
            cell("r16_tax_soft", extra_flags=[*k100, "--taxonomy"]),
            cell("r16_tax_filter", extra_flags=[*k100, "--taxonomy", *sets("taxonomy.mode=filter")]),
        ]
    if round_id == 17:  # Layer D: retrieval order and the Allen filter
        return [
            cell("r17_vector", extra_flags=k100),
            cell("r17_det_d0", extra_flags=k100 + sets("retrieval.order=deterministic", "retrieval.expansion_depth=0")),
            cell("r17_det_d1", extra_flags=k100 + sets("retrieval.order=deterministic")),
            cell("r17_allen", extra_flags=k100 + sets("retrieval.temporal_filter=true")),
        ]
    if round_id == 18:  # Layer D: write-time rung 3 and interval closing
        return [
            cell("r18_onto_off", extra_flags=k100),
            cell("r18_onto_r3", extra_flags=[*k100, "--ontology", *sets("ontology.write_embedding_rung=true")]),
            cell("r18_closing", extra_flags=k100 + sets("ontology.interval_closing=true")),
        ]
    if round_id == 19:  # Layer E: compile with growth + certified ranking (lifecycle promotion)
        return [
            cell("r19_base", extra_flags=k100),
            cell(
                "r19_certified",
                extra_flags=[*k100, "--policy", *sets("policy.packs=lifecycle", "retrieval.certified_boost=0.2")],
            ),
            cell("r19_compile_tax", extra_flags=[*k100, "--compile", "--taxonomy"]),
        ]
    raise ValueError(f"unknown round {round_id}")


def cell_cmd(cell: dict, convs: str, limit: int, extra: list[str]) -> list[str]:
    cmd = [
        PY,
        "-m",
        "grafeo_memory.bench.locomo_eval",
        "all",
        "--run",
        f"mx_{cell['name']}",
        "--convs",
        convs,
        "--decision-model",
        cell["decision_model"],
        "--memories-engine",
        cell["memories_engine"],
        "--relations-engine",
        cell["relations_engine"],
        "--search-engine",
        cell["search_engine"],
        "--granularity",
        cell["granularity"],
        "--escalation",
        cell["escalation"],
        "--conf-threshold",
        str(cell["conf_threshold"]),
        "--blend",
        str(cell["blend"]),
    ]
    if cell["question_pack"]:
        cmd += ["--question-pack", cell["question_pack"]]
    cmd += cell.get("extra_flags", [])
    if limit:
        cmd += ["--limit", str(limit)]
    return cmd + extra


def est_cost(usage: dict, prices: dict) -> float:
    """Estimated $ from tracked tokens. llm ops + mistral_direct priced by model."""
    total = 0.0
    for model, t in usage.get("mistral_direct", {}).items():
        pin, pout = prices.get(model, (2.0, 6.0))
        total += (t["input_tokens"] * pin + t["output_tokens"] * pout) / 1e6
    # pydantic-ai agent calls all ran on the ingest model
    llm = usage.get("llm", {})
    pin, pout = prices.get("mistral-small-latest", (0.2, 0.6))
    total += (llm.get("total_input_tokens", 0) * pin + llm.get("total_output_tokens", 0) * pout) / 1e6
    dm = usage.get("decision_model")
    if dm:
        pin, pout = prices.get("jev", (0.042, 0.0))
        total += (dm.get("input_tokens", 0) * pin + dm.get("output_tokens", 0) * pout) / 1e6
    return round(total, 4)


def aggregate(cells: list[dict]) -> list[dict]:
    rows = []
    for cell in cells:
        run_dir = RUNS / f"mx_{cell['name']}"
        row = {
            "cell": cell["name"],
            "config": {k: v for k, v in cell.items() if k != "name" and v not in ("", "llm")} or "llm-all",
        }
        for key in ("judge_overall_1_4", "judge_per_cat", "f1_overall_1_4", "evidence_recall", "n_questions"):
            row[key] = None
        row["est_cost_usd"] = None
        sp = run_dir / "summary.json"
        if sp.exists():
            s = json.load(open(sp))
            row["judge_overall_1_4"] = (s.get("llm_judge") or {}).get("overall_cats_1_4")
            row["judge_per_cat"] = (s.get("llm_judge") or {}).get("per_category")
            row["f1_overall_1_4"] = (s.get("f1") or {}).get("overall_cats_1_4")
            row["evidence_recall"] = (s.get("evidence_recall") or {}).get("overall_cats_1_4")
            row["n_questions"] = (s.get("llm_judge") or {}).get("n")
        up = run_dir / "usage.json"
        if up.exists():
            u = json.load(open(up))
            row["llm_calls"] = sum(e["calls"] for e in u.get("llm", {}).get("ops", {}).values())
            row["llm_ops"] = {k: v["calls"] for k, v in u.get("llm", {}).get("ops", {}).items()}
            row["decision"] = u.get("decision_model")
            row["est_cost_usd"] = est_cost(u, DEFAULT_PRICES)
        adds, add_times, engines = 0, [], {}
        for ip in run_dir.glob("ingest_*.json"):
            log = json.load(open(ip))
            adds += len(log.get("sessions", []))
            add_times += [s["add_s"] for s in log.get("sessions", []) if "add_s" in s]
            for s in log.get("sessions", []):
                for e in s.get("events", []):
                    engines[e.get("engine", "?")] = engines.get(e.get("engine", "?"), 0) + 1
        row["adds"] = adds
        row["mean_add_s"] = round(sum(add_times) / len(add_times), 2) if add_times else None
        row["event_engines"] = engines
        rows.append(row)
    return rows


def print_table(rows: list[dict]) -> None:
    hdr = [
        "cell",
        "judge1-4",
        "f1",
        "ev_recall",
        "n",
        "llm_calls",
        "dec_calls",
        "dec_q",
        "dec_tok_in",
        "$/cell",
        "mean_add_s",
        "engines",
    ]
    print(" | ".join(hdr))
    print(" | ".join("---" for _ in hdr))
    for r in rows:
        d = r.get("decision") or {}
        print(
            " | ".join(
                str(x)
                for x in [
                    r["cell"],
                    r.get("judge_overall_1_4"),
                    r.get("f1_overall_1_4"),
                    r.get("evidence_recall"),
                    r.get("n_questions"),
                    r.get("llm_calls"),
                    d.get("calls"),
                    d.get("questions"),
                    d.get("input_tokens"),
                    r.get("est_cost_usd"),
                    r.get("mean_add_s"),
                    r.get("event_engines"),
                ]
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=0, help="matrix round id (0-19)")
    ap.add_argument("--convs", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--extra", default="", help="extra args forwarded to locomo_eval.py, e.g. '--workers 8 --judge-model ...'"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--report", action="store_true", help="only aggregate existing runs")
    ap.add_argument("--only", default="", help="comma-separated cell names to run and print")
    args = ap.parse_args()

    all_cells = cells_for_round(args.round)
    cells = all_cells
    if args.only:
        wanted = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = sorted(set(wanted) - {c["name"] for c in all_cells})
        if unknown:
            raise SystemExit(f"--only: no such cell(s) in round {args.round}: {', '.join(unknown)}")
        cells = [c for c in all_cells if c["name"] in wanted]

    extra = args.extra.split() if args.extra else []
    if not args.report:
        for cell in cells:
            cmd = cell_cmd(cell, args.convs, args.limit, extra)
            print(">>>", " ".join(cmd), flush=True)
            if args.dry_run:
                continue
            # retry once: transient API/network failures kill a cell mid-phase, and
            # every finished call is cached, so the retry only redoes the tail
            for attempt in (1, 2):
                t0 = time.time()
                r = subprocess.run(cmd, cwd=REPO)
                print(
                    f"<<< cell {cell['name']} attempt {attempt} rc={r.returncode} in {time.time() - t0:.0f}s",
                    flush=True,
                )
                if r.returncode == 0:
                    break
            else:
                print(f"cell {cell['name']} FAILED twice, continuing with the next cell", flush=True)

    # the round file always covers every cell (read back from disk), so --only never truncates it
    rows = aggregate(all_cells)
    RUNS.mkdir(exist_ok=True)
    out = RUNS / f"matrix_round{args.round}_results.json"
    json.dump({"round": args.round, "convs": args.convs, "limit": args.limit, "rows": rows}, open(out, "w"), indent=1)
    print(f"\n=== matrix round {args.round} (convs={args.convs}, limit={args.limit or 'all'}) -> {out}")
    shown = {c["name"] for c in cells}
    rows = [r for r in rows if r["cell"] in shown]
    print_table(rows)
    for r in rows:
        if r.get("judge_per_cat"):
            print(r["cell"], "per-cat judge:", r["judge_per_cat"])


if __name__ == "__main__":
    main()
