"""Laptop tests for analyze.py plots plus the paste-able plot cell (docs/paste_plots_cell.py),
which must also run against the analyze.py of the notebook already uploaded (commit 76a58ae)."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import analyze as an  # noqa: E402


def synth(n_problems=12, n=8, seed=0, tags=("base", "text_rl", "tool_rl")):
    rng = np.random.default_rng(seed)
    rows = []
    for tag, base_p in zip(tags, (0.3, 0.5, 0.45)):
        for cond, shift in (("prompt_off", 0.0), ("env_denied", -0.05), ("tool_on", 0.1)):
            for i in range(n_problems):
                p = float(np.clip(base_p + shift + rng.normal(0, 0.2), 0, 1))
                for j in range(n):
                    rows.append(dict(tag=tag, condition=cond, template="t", model="m", problem_id=f"p{i}",
                                     sample_idx=j, correct=bool(rng.random() < p), truncated=False,
                                     boxed_present=True, code_emitted=False, fake_outputs=0, n_calls=0,
                                     n_denied=0, n_exec_err=0, arith_n=2, arith_wrong=0, gen_tokens=100))
    df = pd.DataFrame(rows)
    df["arm"] = df.tag + " / " + df.condition
    df["file"] = "x.jsonl"
    return df


def test_passk_table_and_summary():
    df = synth()
    pk = an.passk_table(df)
    assert len(pk) == 9 and list(pk.columns[-4:]) == ["pass@1", "pass@2", "pass@4", "pass@8"]
    assert (pk["pass@8"] >= pk["pass@1"]).all()
    assert len(an.summary(df)) == 9


def test_plot_by_model_panels_and_lines():
    fig = an.plot_passk_by_model(synth(), ci=False)
    assert len(fig.axes) == 3
    assert [ax.get_title(loc="left") for ax in fig.axes] == ["base", "text_rl", "tool_rl"]
    assert all(len(ax.get_lines()) == 3 for ax in fig.axes)
    assert [l.get_label() for l in fig.axes[0].get_lines()] == ["base prompt_off", "base env_denied", "base tool_on"]


def test_plot_main_default_arms_and_missing():
    df = synth()
    fig = an.plot_passk_main(df, ci=False)
    labels = [l.get_label() for l in fig.axes[0].get_lines()]
    assert labels == ["tool_rl prompt_off", "text_rl prompt_off", "text_rl tool_on",
                      "base prompt_off", "base tool_on"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fig = an.plot_passk_main(df, arms=[("tool_rl", "plain_cot"), ("base", "prompt_off")], ci=True, iters=20)
    assert "skipped: tool_rl plain_cot" in buf.getvalue() and len(fig.axes[0].get_lines()) == 1


def test_paste_cell_runs_against_uploaded_analyze():
    old_src = subprocess.run(["git", "show", "76a58ae:src/analyze.py"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout
    assert "def plot_passk_by_model" not in old_src  # it really is the older module
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"; (out / "math500").mkdir(parents=True)
        with open(out / "math500" / "x.jsonl", "w") as f:
            for r in synth().drop(columns=["arm", "file"]).to_dict("records"):
                f.write(json.dumps(r) + "\n")
        old = types.ModuleType("analyze"); exec(compile(old_src, "analyze_old.py", "exec"), old.__dict__)
        saved = sys.modules.get("analyze")
        sys.modules["analyze"] = old
        try:
            ns = dict(OUT=str(out), DATASET="math500", RUN_NAME="t")
            exec(compile((ROOT / "docs" / "paste_plots_cell.py").read_text(), "paste_plots_cell.py", "exec"), ns)
        finally:
            sys.modules["analyze"] = saved
        assert (out / "math500" / "passk_by_model.png").stat().st_size > 1000
        assert (out / "math500" / "passk_main.png").stat().st_size > 1000
        assert len(ns["fig1"].axes) == 3 and len(ns["fig2"].axes[0].get_lines()) == 5


def test_plot_accepts_strings_tuples_lists_and_dicts():
    df = synth()
    fig = an.plot(df, ["tool_rl prompt_off", "text_rl/prompt_off", ("base", "prompt_off")], ci=False, title="t")
    ax = fig.axes[0]
    assert [l.get_label() for l in ax.get_lines()] == ["tool_rl prompt_off", "text_rl prompt_off", "base prompt_off"]
    assert ax.get_title(loc="left") == "t"
    assert {l.get_linestyle() for l in ax.get_lines()} == {"-"}        # condition does not vary: all solid
    assert len({l.get_color() for l in ax.get_lines()}) == 3           # color = model
    fig = an.plot(df, {"A": ["base prompt_off", "base tool_on"], "B": ["tool_rl prompt_off", "text_rl prompt_off"]},
                  color="condition", ci=False)
    assert [a.get_title(loc="left") for a in fig.axes] == ["A", "B"]
    a, b = fig.axes
    assert len({l.get_color() for l in a.get_lines()}) == 2 and {l.get_linestyle() for l in a.get_lines()} == {"-"}
    assert len({l.get_color() for l in b.get_lines()}) == 1 and len({l.get_linestyle() for l in b.get_lines()}) == 2
    with tempfile.TemporaryDirectory() as d:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            an.plot(df, ["base prompt_off", "nobody tool_on"], ci=False, save=f"{d}/x.png")
        assert "skipped: nobody tool_on" in buf.getvalue() and Path(d, "x.png").stat().st_size > 1000
    try:
        an.plot(df, ["base"], ci=False)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

def test_load_results_ungraded_guard():
    with tempfile.TemporaryDirectory() as d:
        recs = synth(n_problems=2, n=2).drop(columns=["arm", "file"]).to_dict("records")
        recs[0]["correct"] = None
        with open(Path(d) / "x.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        try:
            an.load_results(d)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "grade.py" in str(e)
        assert an.load_results(d, allow_ungraded=True)["correct"].isna().sum() == 1

def test_calls_table_semantics():
    df = synth(n_problems=2, n=2, tags=("tool_rl",))
    d = df[df.condition == "env_denied"].copy()
    # four samples: 0 calls, 1 call (denied, no retry), 3 calls (denied, 2 retries), 6 calls
    d["n_calls"] = [0, 1, 3, 6]; d["n_denied"] = [0, 1, 3, 6]
    t = an.calls_table(d).iloc[0]
    assert t["called_pct"] == 75.0 and t["calls_total"] == 10 and t["denied_total"] == 10
    assert t["calls_per_caller"] == round(10 / 3, 2)
    assert t["retry_after_denial"] == round((0 + 2 + 5) / 3, 2) and t["retry_pct"] == round(200 / 3, 1)
    assert [t[f"calls={k}"] for k in range(5)] == [1, 1, 0, 1, 0] and t["calls>=5"] == 1
    # tool_on with a 4-call budget: 6 calls -> 2 denied -> 1 retry after the budget message
    d["condition"] = "tool_on"; d["n_denied"] = [0, 0, 0, 2]
    t = an.calls_table(d).iloc[0]
    assert t["denied_total"] == 2 and t["retry_after_denial"] == 1.0 and t["retry_pct"] == 100.0
    d["n_calls"] = 0; d["n_denied"] = 0
    t = an.calls_table(d).iloc[0]
    assert t["called_pct"] == 0.0 and np.isnan(t["calls_per_caller"]) and np.isnan(t["retry_after_denial"])
    assert "called_pct" in an.summary(df).columns

def test_file_stamp_provenance_and_dated_figure():
    import re
    from datetime import datetime, timedelta, timezone
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}", an.file_stamp())
    df = synth(n_problems=2, n=2)
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        t0 = datetime(2026, 10, 14, 20, 55, tzinfo=timezone.utc)
        for i, tag in enumerate(("base", "text_rl")):
            p = folder / f"{tag}__prompt_off__x.jsonl"; p.write_text("{}\n")
            p.with_suffix(".meta.json").write_text(json.dumps(dict(finished_at=(t0 + timedelta(hours=i)).isoformat())))
        (folder / "tool_rl__prompt_off__x.jsonl").write_text("{}\n")   # no meta: falls back to mtime (now)
        cap = an.provenance(folder)
        assert cap.startswith("3 arms generated " + t0.astimezone().strftime("%Y-%m-%d %H:%M") + " to ")
        assert "; plotted " in cap
        assert an.provenance(folder / "empty") == ""
        out = folder / f"passk_main_{an.file_stamp()}.png"
        fig = an.plot(df, ["base prompt_off"], ci=False, caption=cap, save=out)
        assert out.stat().st_size > 1000
        assert any(cap == txt.get_text() for txt in fig.texts)

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
