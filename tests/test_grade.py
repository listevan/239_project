"""grade.py with spawned worker processes and the real math-verify grader. Run: python tests/test_grade.py"""
import contextlib
import io
import json
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


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
