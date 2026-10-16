"""grade.py (spawned workers, real math-verify) and report.py end to end. Run: uv run tests/test_grade.py"""
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gen  # noqa: E402
import grade  # noqa: E402


def row(pid, k, text, gold, tag="base", cond="no_code", graded=False):
    return dict(tag=tag, condition=cond, template="t", model="m", problem_id=pid, sample_idx=k, text=text, gold=gold,
                correct=(True if graded else None), correct_boxed=None, **({"correct_any": True} if graded else {}),
                arith_n=None, arith_wrong=None, truncated=False,
                boxed_present="\\boxed" in text, fake_outputs=0, n_calls=0, n_denied=0, n_exec_err=0, gen_tokens=10)


def _slow_worker(text, gold):
    time.sleep(30)
    return True, 0, 0, 30.0, True


def test_old_scheme_rows_are_regraded_and_string_fallback():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "a.jsonl"
        old = row("p0", 0, "answer 7, no box", "7")
        old.update(correct=True, correct_boxed=False)            # graded before correct_any existed
        gen.write_jsonl_atomic(path, [old])
        with contextlib.redirect_stdout(io.StringIO()):
            assert grade.grade_dir(Path(d), workers=0)[0]["graded_now"] == 1
        r = json.loads(open(path).readline())
        assert r["correct"] is False and r["correct_any"] is True


def test_grading_with_workers():
    with tempfile.TemporaryDirectory() as d:
        folder = Path(d)
        path = folder / "base__no_code__t.jsonl"
        recs = [row("p0", 0, r"so \boxed{\frac{1}{2}}", r"\frac12"),
                row("p0", 1, r"so 2 + 2 = 5, \boxed{0.25}", r"\frac12"),
                row("p1", 0, r"\boxed{10^{10^{10}}}", "156"),          # math-verify's own alarm: False, fast
                row("p1", 1, "the answer is 156", "156"),               # unboxed: correct but not correct_boxed
                row("p2", 1, r"\boxed{7}", "7", graded=True)]           # already graded: left alone
        gen.write_jsonl_atomic(path, recs)
        gen.meta_path(path).write_text(json.dumps(dict(n=2, graded=False)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rows = grade.grade_dir(folder, workers=2, timeout=60)
        assert rows[0]["graded_now"] == 4 and rows[0]["timeouts"] == 0
        out = [json.loads(line) for line in open(path)]
        assert [r["correct"] for r in out] == [True, False, False, False, True]      # boxed answer only
        assert [r["correct_any"] for r in out[:4]] == [True, False, False, True]    # whole text, for reference
        assert out[1]["arith_n"] == 1 and out[1]["arith_wrong"] == 1
        meta = json.load(open(gen.meta_path(path)))
        assert meta["graded"] is True and meta["grader"].startswith("math_verify") and meta["pass1"] == 0.4
        with contextlib.redirect_stdout(buf):
            assert grade.grade_dir(folder, workers=2)[0]["graded_now"] == 0


def test_timeout_marks_incorrect():
    recs = [row("p0", 0, "x", "1"), row("p0", 1, "y", "1")]
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        stats = grade.grade_records(recs, workers=2, timeout=2, worker=_slow_worker)
    assert stats["n_timeout"] == 2 and all(r["correct"] is False and r["grade_timeout"] for r in recs)
    assert time.time() - t0 < 25


def make_run(root):
    run = root / "2026-10-16_t"
    folder = run / "aime_all"
    folder.mkdir(parents=True)
    arms = {"tool_rl": ("tool_on", "env_denied", "no_code"), "base": ("tool_on", "env_denied", "no_code"), "text_rl": ("no_code",)}
    for tag, conds in arms.items():
        for cond in conds:
            path = folder / f"{tag}__{cond}__verltool_torl.jsonl"
            recs = [row(f"p{i}", k, r"\boxed{7}" if (i + k) % 2 else r"\boxed{8}", "7", tag=tag, cond=cond)
                    for i in range(4) for k in range(2)]
            gen.write_jsonl_atomic(path, recs)
            gen.meta_path(path).write_text(json.dumps(dict(n=2, n_problems=4, graded=False,
                                                           finished_at="2026-10-16T20:00:00+00:00")))
    return run


def test_report_end_to_end():
    import report
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "runs"
        run = make_run(root)
        old_env = os.environ.get("RUNS_ROOT")
        os.environ["RUNS_ROOT"] = str(root)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                report.main(["--list"])
                report.main(["--run", run.name, "--no-ci", "--workers", "2"])
            out = buf.getvalue()
            assert f"{run.name}  [aime_all]" in out and "grading 8 of 8 rows" in out and "56 samples, 4 problems, 7 arms" in out
            assert "positive control" in out and "erosion test" in out and "written: " in out
            figs = sorted(p.name for p in (run / "figures").iterdir() if p.is_file())
            assert [re.sub(r"_\d{4}-\d{2}-\d{2}_\d{4}", "", f) for f in figs] == [
                "passk_env_denied_aime_all.png", "passk_models_aime_all.png", "passk_no_code_aime_all.png",
                "passk_tool_on_aime_all.png"]
            assert len([p for p in (run / "tables").iterdir() if p.is_file()]) == 3
            # unchanged data: same stamp; no terminal and no --yes -> nothing overwritten; --yes -> rewritten
            first = {p.name: p.stat().st_mtime_ns for p in (run / "figures").iterdir() if p.is_file()}
            with contextlib.redirect_stdout(buf):
                report.main(["--run", run.name, "--no-ci", "--no-grade"])
            assert "nothing written" in buf.getvalue()
            assert {p.name: p.stat().st_mtime_ns for p in (run / "figures").iterdir() if p.is_file()} == first
            time.sleep(0.05)
            stale = run / "figures" / "passk_no_code_aime_all_2020-01-01_0000.png"
            stale.write_bytes(b"x")
            same_stamp_old_name = run / "figures" / ("passk_main_denied_aime_all_" + sorted(first)[0].rsplit("_", 2)[-2] + "_"
                                                     + sorted(first)[0].rsplit("_", 1)[-1])
            same_stamp_old_name.write_bytes(b"x")
            with contextlib.redirect_stdout(buf):
                report.main(["--run", run.name, "--no-ci", "--no-grade", "--yes"])
            after = {p.name: p.stat().st_mtime_ns for p in (run / "figures").iterdir() if p.is_file()}
            assert set(after) == set(first) and all(after[k] > first[k] for k in first)
            assert (run / "figures" / "old" / stale.name).exists()
            assert (run / "figures" / "old" / same_stamp_old_name.name).exists()   # same stamp, name no longer written
            try:
                report.main(["--run", "nope"])
                raise AssertionError("expected SystemExit")
            except SystemExit as e:
                assert "no such run" in str(e)
        finally:
            if old_env is None:
                del os.environ["RUNS_ROOT"]
            else:
                os.environ["RUNS_ROOT"] = old_env


def test_find_runs_root():
    import report
    with tempfile.TemporaryDirectory() as d:
        home = Path(d)
        nested = home / "Library/CloudStorage/GoogleDrive-x@y.z/My Drive/UCLA/Classes/CS 239/239_project/runs"
        nested.mkdir(parents=True)
        old_env = os.environ.pop("RUNS_ROOT", None)
        try:
            assert report.find_runs_root(home=home) == nested
            shallow = home / "Library/CloudStorage/GoogleDrive-x@y.z/My Drive/239_project/runs"
            shallow.mkdir(parents=True)
            assert report.find_runs_root(home=home) == shallow
            assert report.find_runs_root(home=home / "nowhere").name == "runs"
        finally:
            if old_env is not None:
                os.environ["RUNS_ROOT"] = old_env


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
