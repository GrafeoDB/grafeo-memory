"""Benchmark gate, offline edition ($0 in CI).

- Structural: every matrix round 0-9 dry-runs to the expected cells/flags.
- Aggregation: matrix --report produces the required result schema.
- Live smoke (only with GRAFEO_MEMORY_BENCH_LIVE=1 + API keys): the fake-engine
  cell runs end-to-end and writes a well-formed summary.

Score-threshold gating (assert judge >= baseline) lands once the headline runs
exist; until then the gate protects structure and cache behavior, and the
matrix results JSON is the scoreboard.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from .conftest import PY, REPO

EXPECTED_CELLS = {
    0: ["r0_baseline_llm"],
    1: ["r1_decision_perfact", "r1_decision_pairwise"],
    2: ["r2_hybrid_conf", "r2_hybrid_probgap", "r2_hybrid_combined"],
    3: ["r3_hybrid_t0.6", "r3_hybrid_t0.75", "r3_hybrid_t0.9"],
    4: ["r4_llm", "r4_decision", "r4_hybrid"],
    5: ["r5_temporal_off", "r5_temporal_on"],
    6: ["r6_onto_off", "r6_onto_on"],
    7: ["r7_d_t0.6", "r7_d_t0.75", "r7_d_t0.9"],
    8: ["r8_e_off", "r8_e_on"],
    9: ["r9_full_t0.75", "r9_full_t0.6"],
    10: ["r10_llm", "r10_llm_pre", "r10_hyb", "r10_hyb_pre"],
    11: ["r11_t0.6_pf", "r11_t0.9_pf", "r11_t0.75_pw", "r11_t0.6_pw"],
    12: ["r12_k100", "r12_rk300", "r12_depth2", "r12_diverse", "r12_rrf30"],
    13: ["r13_bitemp_off", "r13_bitemp_on"],
}


def _matrix(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [PY, "-m", "grafeo_memory.bench.matrix", *args],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_all_rounds_dry_run_to_expected_cells():
    for rnd, cells in EXPECTED_CELLS.items():
        r = _matrix("--round", str(rnd), "--convs", "0,1,2", "--dry-run")
        assert r.returncode == 0, r.stderr
        for cell in cells:
            assert f"mx_{cell}" in r.stdout, f"round {rnd} missing cell {cell}"


def test_layer_flags_present_in_future_rounds():
    r = _matrix("--round", "7", "--convs", "0,1,2", "--dry-run")
    assert "--temporal-precheck" in r.stdout and "--ontology" in r.stdout
    r = _matrix("--round", "9", "--convs", "all", "--dry-run")
    assert "--compile" in r.stdout


SCOREBOARD_KEYS = ("judge_overall_1_4", "f1_overall_1_4", "evidence_recall", "n_questions", "est_cost_usd")


def _runs() -> Path:
    return Path(os.environ["GRAFEO_MEMORY_BENCH_HOME"]) / "locomo_runs"


def test_report_schema():
    # one finished cell (seeded) and one never run: both rows carry the full scoreboard schema
    seeded = _runs() / "mx_r1_decision_perfact"
    seeded.mkdir(parents=True)
    summary = {
        "llm_judge": {"overall_cats_1_4": 0.5, "per_category": {}, "n": 2},
        "f1": {"overall_cats_1_4": 0.4},
        "evidence_recall": {"overall_cats_1_4": 0.9},
    }
    (seeded / "summary.json").write_text(json.dumps(summary))
    (seeded / "usage.json").write_text(json.dumps({"llm": {"ops": {}}, "mistral_direct": {}}))
    r = _matrix("--round", "1", "--report")
    assert r.returncode == 0, r.stderr
    out = _runs() / "matrix_round1_results.json"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["round"] == 1 and data["rows"], "report must carry rows"
    rows = {row["cell"]: row for row in data["rows"]}
    for row in rows.values():
        assert "config" in row
        for key in SCOREBOARD_KEYS:
            assert key in row, f"missing {key} in {row['cell']}"
    assert rows["r1_decision_perfact"]["judge_overall_1_4"] == 0.5
    assert rows["r1_decision_perfact"]["est_cost_usd"] == 0.0
    assert rows["r1_decision_pairwise"]["judge_overall_1_4"] is None


def test_only_keeps_the_full_round_file_and_rejects_unknown_cells():
    r = _matrix("--round", "1", "--report", "--only", "r1_decision_pairwise")
    assert r.returncode == 0, r.stderr
    data = json.loads((_runs() / "matrix_round1_results.json").read_text())
    assert {row["cell"] for row in data["rows"]} == {"r1_decision_perfact", "r1_decision_pairwise"}
    r = _matrix("--round", "1", "--report", "--only", "r1_typo")
    assert r.returncode != 0 and "r1_typo" in r.stderr


def test_live_smoke_fake_engine(live: bool, tmp_path: Path):
    if not live:
        import pytest

        pytest.skip("set GRAFEO_MEMORY_BENCH_LIVE=1 with API keys to run the live smoke cell")
    cache = tmp_path / "gate_cache.jsonl"
    r = subprocess.run(
        [
            PY,
            "-m",
            "grafeo_memory.bench.locomo_eval",
            "all",
            "--run",
            "gate_smoke",
            "--convs",
            "0",
            "--limit",
            "8",
            "--decision-model",
            "fake",
            "--memories-engine",
            "decision",
            "--relations-engine",
            "decision",
            "--judge",
            "f1",
            "--cache-path",
            str(cache),
            "--workers",
            "4",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "GRAFEO_MEMORY_BENCH_HOME": str(tmp_path)},
    )
    assert r.returncode == 0, r.stderr[-3000:]
    summary = tmp_path / "locomo_runs" / "gate_smoke" / "summary.json"  # the subprocess's bench home
    assert summary.exists()
    s = json.loads(summary.read_text())
    assert "evidence_recall" in s and "f1" in s


def test_locomo_parser_layer_d_flags():
    """--temporal-precheck / --ontology are accepted by the bench CLIs and map
    onto PipelineConfig (getattr defaults keep lme/beam-style arg objects working)."""
    import subprocess
    import sys

    import pytest

    for mod in ("locomo_eval", "lme_eval", "beam_eval"):
        out = subprocess.run(
            [sys.executable, "-m", f"grafeo_memory.bench.{mod}", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode != 0 and "ModuleNotFoundError" in out.stderr:
            pytest.skip(f"bench extra not installed: {mod}")
        assert out.returncode == 0, out.stderr
        assert "--temporal-precheck" in out.stdout, mod
        assert "--ontology" in out.stdout, mod
        for flag in ("--compile", "--taxonomy", "--policy", "--bucket"):
            assert flag in out.stdout, (mod, flag)

    from grafeo_memory.bench.locomo_eval import build_pipeline
    from grafeo_memory.common.types import PipelineConfig

    class Args:
        memories_engine = "llm"
        relations_engine = "llm"
        search_engine = "llm"
        granularity = "per_fact"
        conf_threshold = 0.75
        escalation = "confidence"
        blend = 0.5
        question_pack = ""
        temporal_precheck = True
        ontology = True
        bitemporal = True
        episodes = False
        compile = True
        taxonomy = True
        policy = False
        bucket = True

    pipe = build_pipeline(Args())
    assert isinstance(pipe, PipelineConfig)
    assert pipe.temporal_precheck is True and pipe.ontology is True
    for flag in ("bitemporal", "episodes", "compile", "taxonomy", "policy", "bucket"):
        assert getattr(pipe, flag) is getattr(Args(), flag), flag

    class ArgsOff:
        memories_engine = "llm"
        relations_engine = "llm"
        search_engine = "llm"
        granularity = "per_fact"
        conf_threshold = 0.75
        escalation = "confidence"
        blend = 0.5
        question_pack = ""
        # no temporal_precheck/ontology attrs -> getattr default False

    pipe_off = build_pipeline(ArgsOff())
    assert pipe_off.temporal_precheck is False and pipe_off.ontology is False
    for flag in ("bitemporal", "episodes", "compile", "taxonomy", "policy", "bucket"):
        assert getattr(pipe_off, flag) is False, flag
