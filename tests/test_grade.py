"""grade.py with spawned worker processes and the real math-verify grader. Run: python tests/test_grade.py"""
import contextlib
import io
import json
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gen  # noqa: E402
import grade  # noqa: E402


def row(pid, k, text, gold, graded=False):
    return dict(tag="base", condition="prompt_off", problem_id=pid, sample_idx=k, text=text, gold=gold,
                correct=(True if graded else None), arith_n=None, arith_wrong=None, truncated=False,
                n_calls=0, n_denied=0)


def _slow_worker(text, gold):
    time.sleep(30)
    return True, 0, 0, 30.0


def test_pool_grades_real_rows_and_updates_meta():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        path = folder / "base__prompt_off__qwen_math_nocode.jsonl"
        recs = [row("p0", 0, r"so \boxed{\frac{1}{2}}", r"\frac12"),
                row("p0", 1, r"so 2 + 2 = 5, \boxed{0.25}", r"\frac12"),
                row("p1", 0, r"\boxed{10^{10^{10}}}", "156"),      # math-verify's own 5 s alarm, False
                row("p1", 1, "no box " * 300, "156"),
                row("p2", 0, r"\boxed{(3, \frac{\pi}{2})}", r"\left( 3, \frac{\pi}{2} \right)"),
                row("p2", 1, r"\boxed{7}", "7", graded=True)]        # already graded: untouched
        gen.write_jsonl_atomic(path, recs)
        gen.meta_path(path).write_text(json.dumps(dict(n=2, n_problems=3, pass1=None, graded=False)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = grade.grade_dir(folder, workers=2, timeout=60)
        assert rows[0]["graded_now"] == 5 and rows[0]["timeouts"] == 0, rows
        out = [json.loads(line) for line in open(path)]
        assert [r["correct"] for r in out] == [True, False, False, False, True, True]
        assert out[1]["arith_n"] == 1 and out[1]["arith_wrong"] == 1
        assert all("grade_timeout" not in r for r in out)
        meta = json.load(open(gen.meta_path(path)))
        assert meta["graded"] is True and meta["grader"] == "math_verify" and abs(meta["pass1"] - 0.5) < 1e-9
        assert "slowest" in buf.getvalue()
        with contextlib.redirect_stdout(buf):
            assert grade.grade_dir(folder, workers=2)[0]["graded_now"] == 0


def test_per_row_timeout_marks_incorrect_and_kills_workers():
    recs = [row("p0", 0, "x", "1"), row("p0", 1, "y", "1")]
    buf = io.StringIO()
    t0 = time.time()
    with contextlib.redirect_stdout(buf):
        stats = grade.grade_records(recs, workers=2, timeout=2, worker=_slow_worker)
    assert stats["n_timeout"] == 2 and all(r["correct"] is False and r["grade_timeout"] for r in recs)
    assert time.time() - t0 < 25, "terminate() should kill the sleeping workers"
    assert buf.getvalue().count("TIMEOUT") == 2


def test_report_run_grades_then_writes_figures_and_tables():
    import os
    import report
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "runs"; run = root / "2026-10-15_t"; folder = run / "math500"; folder.mkdir(parents=True)
        for tag in ("base", "text_rl", "tool_rl"):
            for cond, tmpl in (("prompt_off", "qwen_math_nocode"), ("env_denied", "qwen_math_tir"), ("tool_on", "qwen_math_tir")):
                path = folder / f"{tag}__{cond}__{tmpl}.jsonl"
                recs = [dict(row(f"p{i}", k, r"\boxed{7}" if (i + k) % 2 else r"\boxed{8}", "7"), tag=tag, condition=cond,
                             template=tmpl, model="m", subject="s", level=("5" if i % 2 else "1"), boxed_present=True,
                             finish="eos", gen_tokens=10, rounds=1, code_emitted=False, fake_outputs=0, n_exec_err=0)
                        for i in range(4) for k in range(2)]
                gen.write_jsonl_atomic(path, recs)
                gen.meta_path(path).write_text(json.dumps(dict(n=2, n_problems=4, pass1=None, graded=False)))
        old = os.environ.get("RUNS_ROOT"); os.environ["RUNS_ROOT"] = str(root)
        buf = io.StringIO()
        try:
            assert report.find_runs_root() == root
            with contextlib.redirect_stdout(buf):
                report.main(["--list"])
                report.main(["--run", "2026-10-15_t", "--no-ci", "--workers", "2"])
        finally:
            if old is None:
                del os.environ["RUNS_ROOT"]
            else:
                os.environ["RUNS_ROOT"] = old
        out = buf.getvalue()
        assert "2026-10-15_t  [math500]" in out
        assert "grading 8 of 8 rows" in out and "72 samples, 4 problems, 9 arms" in out
        assert "level 5 only (2 problems)" in out and "written: " in out
        figs = sorted(p.name for p in (run / "figures").iterdir()); tabs = sorted(p.name for p in (run / "tables").iterdir())
        assert len(figs) == 7 and any(re.fullmatch(r"passk_main_denied_math500_\d{4}-\d{2}-\d{2}_\d{4}\.png", f) for f in figs)
        assert not any("prose" in f for f in figs)   # no prose_only arms in this synthetic run: figure skipped
        assert any(f.startswith("passk_main_promptoff_math500-l5_") for f in figs) and any(f.startswith("passk_by_model_math500_") for f in figs)
        assert sum(f.startswith("passk_main_toolon_") for f in figs) == 2
        assert len(tabs) == 6 and any(t.startswith("summary_math500_") for t in tabs) and any(t.startswith("calls_math500-l5_") for t in tabs)
        assert json.load(open(folder / "base__prompt_off__qwen_math_nocode.meta.json"))["graded"] is True
        try:
            report.main(["--run", "nope"])
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "no such run" in str(e)


def test_find_runs_root_searches_nested_drive_folders():
    import os
    import report
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        nested = home / "Library/CloudStorage/GoogleDrive-x@y.z/My Drive/UCLA (Jun 2026 onward)/Classes/CS 239/239_project/runs"
        nested.mkdir(parents=True)
        old = os.environ.pop("RUNS_ROOT", None)
        try:
            assert report.find_runs_root(home=home) == nested
            shallow = home / "Library/CloudStorage/GoogleDrive-x@y.z/My Drive/239_project/runs"
            shallow.mkdir(parents=True)
            assert report.find_runs_root(home=home) == shallow   # shallowest wins
            assert report.find_runs_root(home=home / "nowhere").name == "runs"  # repo fallback
        finally:
            if old is not None:
                os.environ["RUNS_ROOT"] = old

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
