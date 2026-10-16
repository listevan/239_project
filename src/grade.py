#!/usr/bin/env python
"""Grade a folder of trajectories in place, on any machine (no GPU, no vllm).

    python src/grade.py <run>/<dataset> [--workers N] [--timeout S] [--regrade]

gen.py writes rows with correct / correct_any / arith_n / arith_wrong null. This fills them and
updates each .meta.json (graded, grader, pass1). Rows already graded under this scheme are left alone
unless --regrade; rows graded under the old scheme (no correct_any) are regraded automatically.

  correct        math-verify on the last complete \\boxed{...} only. The headline score: the prompt asks
                 for a boxed answer, and the 2026-10-16 audit found whole-text grading crediting stray
                 numbers ("step 3.", a 60-degree angle from the problem) in up to 41% of a tool-off arm's
                 correct rows on OlymMATH.
  correct_any    math-verify on the whole response (also credits an unboxed answer), for reference
  correct_boxed  same as correct (kept for older readers)
  arith_*        steps checked / wrong by src/arith_check.py in the prose, code blocks excluded

Grading runs in spawned worker processes: math-verify's own 5 s alarm catches most slow cases, and a
row that still exceeds --timeout is marked incorrect with grade_timeout=true, so one pathological
answer cannot stall an arm. src/report.py calls this before reporting.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arith_check  # noqa: E402
import gen  # noqa: E402


def make_grader():
    """(grade(response, gold) -> bool, name). math-verify if installed, else a boxed-string match."""
    try:
        from math_verify import parse, verify  # type: ignore

        def grade(response: str, gold: str) -> bool:
            try:
                return bool(verify(parse(gold if "\\boxed" in gold else f"${gold}$"), parse(response)))
            except Exception:  # noqa: BLE001
                return False
        return grade, "math_verify"
    except ImportError:
        def norm(s):
            s = re.sub(r"\s+|\\left|\\right|\\!|\\,|\$", "", s or "").replace("dfrac", "frac")
            return (s.lstrip("0") or "0") if s.isdigit() else s   # AIME golds like "033"

        def grade(response: str, gold: str) -> bool:
            b = gen.last_boxed(response)
            return b is not None and norm(b) == norm(gen.last_boxed(gold) or gold)
        return grade, "string"


def grade_boxed(grade, text, gold):
    """Strict variant: only the last \\boxed{...} counts."""
    b = gen.last_boxed(text)
    return False if b is None else bool(grade("\\boxed{" + b + "}", gold))


_GRADER = None


def _grade_one(text, gold):
    """Worker: (correct_boxed, arith_n, arith_wrong, seconds, correct_any). Builds the grader once per process."""
    global _GRADER
    if _GRADER is None:
        _GRADER = make_grader()
    grade, _ = _GRADER
    t = time.time()
    correct_boxed = grade_boxed(grade, text, gold)
    correct_any = bool(grade(text, gold))
    a_n, a_wrong, _ = arith_check.check(gen.strip_fenced(text))
    return correct_boxed, a_n, a_wrong, time.time() - t, correct_any


def _apply(r, result):
    boxed, a_n, a_wrong, secs = result[:4]
    r["correct"] = r["correct_boxed"] = boxed
    r["correct_any"] = result[4] if len(result) > 4 else None
    r["arith_n"], r["arith_wrong"] = a_n, a_wrong
    r.pop("grade_timeout", None)
    return secs


def needs_grading(r, regrade=False):
    return regrade or r.get("correct") is None or "correct_any" not in r


def grade_records(recs, workers, timeout=60.0, regrade=False, worker=_grade_one, log=print):
    """Fill the grading fields of `recs` in place; workers=0 grades serially in this process.
    Returns dict(n_graded, n_timeout, seconds, slowest=[(seconds, problem_id, sample_idx), ...])."""
    todo = [r for r in recs if needs_grading(r, regrade)]
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
        pool = mp.get_context("spawn").Pool(workers)
        try:
            futs = [(r, pool.apply_async(worker, (r["text"], r["gold"]))) for r in todo]
            for i, (r, f) in enumerate(futs, 1):
                try:
                    note(r, _apply(r, f.get(timeout=timeout)))
                except mp.TimeoutError:
                    r.update(correct=False, correct_boxed=False, correct_any=False, arith_n=None, arith_wrong=None,
                             grade_timeout=True)
                    n_timeout += 1
                    log(f"   TIMEOUT after {timeout:.0f}s: problem {r['problem_id']} sample {r['sample_idx']} marked incorrect")
                if i % step == 0:
                    log(f"   {i}/{len(todo)} graded, {time.time() - t0:.0f}s")
        finally:
            pool.terminate()
            pool.join()
    return dict(n_graded=len(todo), n_timeout=n_timeout, seconds=time.time() - t0, slowest=slow)


def _pass1(recs):
    vals = [r["correct"] for r in recs if r.get("correct") is not None]
    return sum(vals) / len(vals) if vals else None


def grade_dir(results_dir, workers, timeout=60.0, regrade=False, log=print):
    """Grade every *.jsonl under results_dir in place. Returns one summary row per file."""
    results_dir = Path(results_dir)
    paths = sorted(results_dir.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl under {results_dir}")
    grader_name = make_grader()[1]
    rows = []
    for path in paths:
        recs = [json.loads(line) for line in open(path) if line.strip()]
        n_todo = sum(1 for r in recs if needs_grading(r, regrade))
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
        meta.update(graded=True, grader=f"{grader_name}, boxed answer only", pass1=acc, grade_seconds=stats["seconds"],
                    grade_timeouts=stats["n_timeout"], grade_workers=workers)
        mpath.write_text(json.dumps(meta, indent=1))
        log(f"   pass@1={acc:.3f}  {stats['seconds']:.0f}s  timeouts={stats['n_timeout']}  slowest: "
            + ", ".join(f"{s:.1f}s ({pid} #{k})" for s, pid, k in stats["slowest"][:3]))
        rows.append(dict(file=path.name, rows=len(recs), graded_now=stats["n_graded"], pass1=acc,
                         timeouts=stats["n_timeout"], seconds=round(stats["seconds"], 1)))
    return rows


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", help="folder holding the *.jsonl files, e.g. <run>/aime_all")
    ap.add_argument("--workers", type=int, default=default_workers(), help="worker processes (0 = serial)")
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per row before it is marked incorrect")
    ap.add_argument("--regrade", action="store_true", help="grade every row again, not only the null ones")
    args = ap.parse_args(argv)
    rows = grade_dir(args.results_dir, args.workers, args.timeout, args.regrade)
    print(f"\n{len(rows)} file(s) graded in {args.results_dir}")


if __name__ == "__main__":
    main()
