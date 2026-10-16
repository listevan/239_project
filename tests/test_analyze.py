"""analyze.py on synthetic graded rows. Run: uv run tests/test_analyze.py"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import analyze as an  # noqa: E402

ARMS = {"tool_rl": ("tool_on", "env_denied", "no_code"), "base": ("tool_on", "env_denied", "no_code"),
        "text_rl": ("no_code",)}


def synth(n_problems=12, n=8, seed=0, arms=ARMS):
    rng = np.random.default_rng(seed)
    p_model = {"base": 0.3, "tool_rl": 0.45, "text_rl": 0.4}
    shift = {"tool_on": 0.15, "env_denied": -0.05, "no_code": 0.0}
    rows = []
    for tag, conds in arms.items():
        for cond in conds:
            for i in range(n_problems):
                p = float(np.clip(p_model[tag] + shift[cond] + rng.normal(0, 0.2), 0, 1))
                for j in range(n):
                    ok = bool(rng.random() < p)
                    rows.append(dict(tag=tag, condition=cond, template="t", model="m", problem_id=f"p{i}", sample_idx=j,
                                     correct=ok, correct_boxed=ok, truncated=False, boxed_present=True, code_emitted=False,
                                     fake_outputs=0, n_calls=int(cond != "no_code"), n_denied=int(cond == "env_denied"),
                                     n_exec_err=0, arith_n=2, arith_wrong=0, gen_tokens=100))
    df = pd.DataFrame(rows)
    df["arm"] = df.tag + " / " + df.condition
    df["file"] = "x.jsonl"
    return df


def test_tables_and_order():
    df = synth()
    pk = an.passk_table(df)
    assert len(pk) == 7 and list(pk.columns[-4:]) == ["pass@1", "pass@2", "pass@4", "pass@8"]
    assert (pk["pass@8"] >= pk["pass@1"]).all()
    assert list(pk.condition)[:2] == ["tool_on", "tool_on"] and list(pk.tag)[:2] == ["base", "tool_rl"]
    s = an.summary(df)
    assert len(s) == 7 and {"pass1_any", "pass@8", "unfinished_pct", "early_stop_pct", "code_like_pct", "called_pct",
                            "arith_err_pct"} <= set(s.columns)
    assert an.pass_at_k(4, 0, 2) == 0.0 and an.pass_at_k(4, 4, 2) == 1.0 and np.isnan(an.pass_at_k(2, 1, 4))


def test_calls_table():
    d = synth(n_problems=2, n=2, arms={"tool_rl": ("env_denied",)})
    d["n_calls"] = [0, 1, 3, 6]
    d["n_denied"] = [0, 1, 3, 6]
    t = an.calls_table(d).iloc[0]
    assert t["called_pct"] == 75.0 and t["calls_total"] == 10 and t["retry_after_denial"] == round(7 / 3, 2)
    assert [t[f"calls={k}"] for k in range(5)] == [1, 1, 0, 1, 0] and t["calls>=5"] == 1


def test_plot():
    df = synth()
    fig = an.plot(df, ["tool_rl no_code", "base no_code", ("text_rl", "no_code")], ci=False, title="t")
    assert [l.get_label() for l in fig.axes[0].get_lines()] == ["tool_rl no_code", "base no_code", "text_rl no_code"]
    assert fig.axes[0].get_title(loc="left") == "t"
    fig = an.plot(df, {"tool_rl": ["tool_rl tool_on", "tool_rl no_code"], "base": ["base no_code"]}, color="condition", ci=False)
    assert [a.get_title(loc="left") for a in fig.axes] == ["tool_rl", "base"]
    with tempfile.TemporaryDirectory() as d:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            an.plot(df, ["base no_code", "nobody no_code"], ci=True, iters=20, save=f"{d}/x.png", caption="c")
        assert "skipped: nobody no_code" in buf.getvalue() and Path(d, "x.png").stat().st_size > 1000
    try:
        an.plot(df, ["base"], ci=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_verdict():
    df = synth(n_problems=10)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        an.verdict(df, "no_code", iters=50)
        an.verdict(df, "tool_on", iters=50)
        an.verdict(df, "env_denied", iters=50)
    out = buf.getvalue()
    first = [l for l in out.splitlines() if " - " in l and "pass@" in l][0]
    assert "tool_rl - base" in first
    assert "erosion test (tool_rl vs untrained base)" in out and "attribution (vs text_rl)" in out
    assert "positive control" in out and "Holm-adjusted" in out
    env = out.split("[env_denied]")[1]
    assert "attribution" not in env                    # no text_rl env_denied arm: base-only readout


def test_load_results_and_stamps():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        recs = synth(n_problems=2, n=2).drop(columns=["arm", "file"]).to_dict("records")
        recs[0]["correct"] = None
        with open(folder / "x.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        try:
            an.load_results(folder)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "grade.py" in str(e)
        assert an.load_results(folder, allow_ungraded=True)["correct"].isna().sum() == 1
        (folder / "x.jsonl").unlink()
        t0 = datetime(2026, 10, 16, 20, 55, tzinfo=timezone.utc)
        for i, tag in enumerate(("base", "tool_rl")):
            p = folder / f"{tag}__no_code__t.jsonl"
            p.write_text("{}\n")
            p.with_suffix(".meta.json").write_text(json.dumps(dict(finished_at=(t0 + timedelta(hours=i)).isoformat())))
        assert an.data_stamp(folder) == (t0 + timedelta(hours=1)).astimezone().strftime("%Y-%m-%d_%H%M")
        assert an.provenance(folder).startswith("2 arms generated " + t0.astimezone().strftime("%Y-%m-%d %H:%M") + " to ")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}", an.data_stamp(folder / "empty"))


def test_plot_caps_at_shared_k():
    big = synth(n_problems=6, n=64, arms={"tool_rl": ("no_code",)})
    small = synth(n_problems=6, n=16, seed=1, arms={"base": ("no_code",)})
    df = pd.concat([big, small], ignore_index=True)
    assert an.shared_kmax(df) == 16
    fig = an.plot(df, ["tool_rl no_code", "base no_code"], ci=False)
    ax = fig.axes[0]
    assert max(max(l.get_xdata()) for l in ax.get_lines()) == 16 and list(ax.get_xticks()) == [1, 2, 4, 8, 16]
    fig = an.plot(df, ["tool_rl no_code"], ci=False, kmax=None)
    assert max(fig.axes[0].get_lines()[0].get_xdata()) == 64

def _two_arm(p_tool, p_base, n_problems=60, n=16, cond="no_code", seed=0, p_text=None):
    rng = np.random.default_rng(seed)
    rows = []
    arms = [("tool_rl", p_tool), ("base", p_base)] + ([("text_rl", p_text)] if p_text is not None else [])
    for tag, pfn in arms:
        for i in range(n_problems):
            p = pfn(i)
            for j in range(n):
                rows.append(dict(tag=tag, condition=cond, problem_id=f"p{i}", sample_idx=j, correct=bool(rng.random() < p),
                                 truncated=False, boxed_present=True, n_calls=0, n_denied=0, n_exec_err=0,
                                 fake_outputs=0, arith_n=0, arith_wrong=0, gen_tokens=10))
    df = pd.DataFrame(rows)
    df["arm"] = df.tag + " / " + df.condition
    return df


def test_verdict_wording():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        an.verdict(_two_arm(lambda i: 0.2, lambda i: 0.2, cond="tool_on"), "tool_on", iters=300)   # control fails
        an.verdict(_two_arm(lambda i: 0.2, lambda i: 0.2), "no_code", iters=300)                   # equal
        # sharpening: tool_rl sure on half the problems, never on the rest; base 30% everywhere
        an.verdict(_two_arm(lambda i: 0.9 if i % 2 else 0.0, lambda i: 0.3), "no_code", iters=300)
        # text_rl gained, tool_rl level with base
        an.verdict(_two_arm(lambda i: 0.1, lambda i: 0.1, p_text=lambda i: 0.4), "no_code", iters=300)
    out = buf.getvalue()
    assert "positive control: FAILED" in out
    assert "inconclusive: no difference detected" in out and "not evidence of no erosion" in out
    assert "sharpening: tool_rl above base at pass@1 but below at pass@16" in out
    assert "text_rl rose above base and tool_rl did not" in out and "tool-specific cost" not in out
    assert an._holm([0.01, 0.04, float("nan")])[:2] == [0.02, 0.04] and an._describe(float("nan"), 0, 0, 0.01) == "no data"

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
