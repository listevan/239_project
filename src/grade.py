#!/usr/bin/env python
"""Grade trajectories on any machine (no GPU, no vllm).

    python src/grade.py results/<run>/<dataset> [--workers N] [--timeout S] [--regrade]

gen.py --no-grade writes rows with `correct`, `arith_n` and `arith_wrong` null. This fills them in
place (atomic rewrite of each *.jsonl) and updates the .meta.json (graded, grader, pass1). Rows that
already carry `correct` are left alone unless --regrade.

Grading runs in a pool of spawned worker processes: math-verify's own 5 s alarm catches most slow
cases, and a row that still exceeds --timeout seconds is marked incorrect with grade_timeout=true
so one pathological answer cannot stall an arm. The slowest rows are reported at the end.
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arith_check  # noqa: E402
import gen  # noqa: E402

_GRADER = None


def _grade_one(text, gold):
    """Worker: (correct, arith_n, arith_wrong, seconds). Builds the grader once per process."""
    global _GRADER
    if _GRADER is None:
        _GRADER = gen.make_grader()
    grade, _ = _GRADER
    t = time.time()
    correct = bool(grade(text, gold))
    correct_boxed = gen.grade_boxed(grade, text, gold)
    a_n, a_wrong, _ = arith_check.check(gen.strip_fenced(text))
    return correct, a_n, a_wrong, time.time() - t, correct_boxed


def _apply(r, result):
    correct, a_n, a_wrong, secs = result[:4]
    r["correct"], r["arith_n"], r["arith_wrong"] = correct, a_n, a_wrong
    r["correct_boxed"] = result[4] if len(result) > 4 else None
    r.pop("grade_timeout", None)
    return secs


def grade_records(recs, workers, timeout=60.0, regrade=False, worker=_grade_one, log=print):
    """Fill the grading fields of `recs` in place. workers=0 grades serially in this process.
    Returns dict(n_graded, n_timeout, seconds, slowest=[(seconds, problem_id, sample_idx), ...])."""
    todo = [r for r in recs if regrade or r.get("correct") is None]
    t0, n_timeout, slow = time.time(), 0, []
    step = max(10, len(todo) // 10)

    def note(r, secs):
        slow.append((secs, r["problem_id"], r["sample_idx"]))
        slow.sort(reverse=True)
        del slow[5:]

    if not todo:
        return dict(n_graded=0, n_timeout=0, seconds=0.0, slowest=[])
    if workers == 0:
        for i, r in enumerate(todo, 1):
            note(r, _apply(r, worker(r["text"], r["gold"])))
            if i % step == 0:
                log(f"   {i}/{len(todo)} graded, {time.time() - t0:.0f}s")
    else:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(workers)
        try:
            futs = [(r, pool.apply_async(worker, (r["text"], r["gold"]))) for r in todo]
            for i, (r, f) in enumerate(futs, 1):
                try:
                    note(r, _apply(r, f.get(timeout=timeout)))
                except mp.TimeoutError:
                    r.update(correct=False, correct_boxed=False, arith_n=None, arith_wrong=None, grade_timeout=True)
                    n_timeout += 1
                    log(f"   TIMEOUT after {timeout:.0f}s: problem {r['problem_id']} sample {r['sample_idx']} marked incorrect")
                if i % step == 0:
                    log(f"   {i}/{len(todo)} graded, {time.time() - t0:.0f}s")
        finally:
            pool.terminate()
            pool.join()
    return dict(n_graded=len(todo), n_timeout=n_timeout, seconds=time.time() - t0, slowest=slow)


def grade_dir(results_dir, workers, timeout=60.0, regrade=False, log=print):
    """Grade every *.jsonl under results_dir in place. Returns one summary row per file."""
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl under {results_dir}")
    grader_name = gen.make_grader()[1]
    rows = []
    for path in paths:
        recs = [json.loads(line) for line in open(path) if line.strip()]
        n_todo = sum(1 for r in recs if regrade or r.get("correct") is None)
        if not n_todo:
            log(f"{path.name}: {len(recs)} rows already graded, skipped (use --regrade to redo)")
            rows.append(dict(file=path.name, rows=len(recs), graded_now=0, pass1=_pass1(recs)))
            continue
        log(f"{path.name}: grading {n_todo} of {len(recs)} rows with {workers or 'no'} worker processes")
        stats = grade_records(recs, workers, timeout, regrade, log=log)
        gen.write_jsonl_atomic(path, recs)
        acc = _pass1(recs)
        mpath = gen.meta_path(path)
        meta = json.loads(mpath.read_text()) if mpath.exists() else {}
        meta.update(graded=True, grader=grader_name, pass1=acc, grade_seconds=stats["seconds"],
                    grade_timeouts=stats["n_timeout"], grade_workers=workers)
        mpath.write_text(json.dumps(meta, indent=1))
        log(f"   pass@1={acc:.3f}  {stats['seconds']:.0f}s  timeouts={stats['n_timeout']}  slowest: "
            + ", ".join(f"{s:.1f}s ({pid} #{k})" for s, pid, k in stats["slowest"][:3]))
        rows.append(dict(file=path.name, rows=len(recs), graded_now=stats["n_graded"], pass1=acc,
                         timeouts=stats["n_timeout"], seconds=round(stats["seconds"], 1)))
    return rows


def _pass1(recs):
    vals = [r["correct"] for r in recs if r.get("correct") is not None]
    return sum(vals) / len(vals) if vals else None


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="folder holding the *.jsonl files, e.g. results/math-1.5b/math500")
    ap.add_argument("--workers", type=int, default=default_workers(), help="worker processes (0 = serial)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per row before it is marked incorrect")
    ap.add_argument("--regrade", action="store_true", help="grade every row again, not only the null ones")
    args = ap.parse_args(argv)
    rows = grade_dir(args.results_dir, args.workers, args.timeout, args.regrade)
    print(f"\n{len(rows)} file(s) in {args.results_dir}; next:  python src/report.py {args.results_dir}")


if __name__ == "__main__":
    main()
