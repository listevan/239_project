#!/usr/bin/env python
"""Grade (if needed) and report a run, on any machine. The one local command:

    uv run src/report.py --run 2026-10-15_math-1.5b                 # every dataset in the run
    uv run src/report.py --run 2026-10-15_math-1.5b --dataset aime24
    uv run src/report.py <path to a run folder>
    uv run src/report.py --list                                     # runs visible from this machine

Runs live in <runs root>/<run id>/<dataset>/*.jsonl. The runs root is $RUNS_ROOT, else the Google
Drive for desktop mirror of 239_project/runs (any depth under My Drive), else ./runs.

For each dataset: ungraded rows are graded in place (src/grade.py, graded rows are skipped), then the
summary, pass@k, tool-call and verdict tables are printed and saved as CSV under <run>/tables/, and
two figures are saved under <run>/figures/, passk_by_model_<dataset>_<stamp>.png and
passk_main_<dataset>_<stamp>.png, each captioned with the arms' generation times. For math500 the
level-5 subset gets its own tables and main figure (tagged math500-l5).
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import analyze as an  # noqa: E402

BY_MODEL = {
    "base":    ["base prompt_off",    "base env_denied",    "base tool_on"],
    "text_rl": ["text_rl prompt_off", "text_rl env_denied", "text_rl tool_on"],
    "tool_rl": ["tool_rl prompt_off", "tool_rl env_denied", "tool_rl tool_on"],
}
MAIN = ["tool_rl prompt_off", "text_rl prompt_off", "text_rl tool_on", "base prompt_off", "base tool_on"]


def find_runs_root(home=None):
    """$RUNS_ROOT, else a 239_project/runs folder up to five levels under any Google Drive for
    desktop mount (shallowest first), else ./runs in the repo."""
    if os.environ.get("RUNS_ROOT"):
        return Path(os.environ["RUNS_ROOT"])
    hits = []
    for mount in sorted(Path(home or Path.home()).glob("Library/CloudStorage/GoogleDrive-*/My Drive")):
        for depth in range(6):
            hits += [q for q in mount.glob("*/" * depth + "239_project/runs") if q.is_dir()]
    if hits:
        return sorted(hits, key=lambda q: len(q.parts))[0]
    return Path(__file__).resolve().parents[1] / "runs"


def list_runs(root):
    root = Path(root)
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")) if root.is_dir() else []


def datasets_in(run_dir):
    return sorted(p.name for p in Path(run_dir).iterdir() if p.is_dir() and any(p.glob("*.jsonl")))


def _report_frame(df, title, tag, stamp, cap, figs, tabs, ci, by_model):
    import matplotlib.pyplot as plt
    print(f"{len(df)} samples, {df.problem_id.nunique()} problems, {df.arm.nunique()} arms\n")
    tables = dict(summary=an.summary(df), passk=an.passk_table(df).round(3), calls=an.calls_table(df))
    for name, t in tables.items():
        print(f"{name}:")
        print(t.to_string(index=False), "\n")
    for cond in an.COND_ORDER:
        if cond in set(df.condition):
            an.verdict(df, cond)
            print()
    written = []
    for name, t in tables.items():
        p = tabs / f"{name}_{tag}_{stamp}.csv"
        t.to_csv(p, index=False)
        written.append(p)
    if by_model:
        p = figs / f"passk_by_model_{tag}_{stamp}.png"
        fig = an.plot(df, BY_MODEL, color="condition", ci=ci, title=f"{title}: pass@k within each model",
                      save=p, caption=cap)
        plt.close(fig)
        written.append(p)
    p = figs / f"passk_main_{tag}_{stamp}.png"
    fig = an.plot(df, MAIN, ci=ci, title=f"{title}: tool-RL with the tool off vs control and base",
                  save=p, caption=cap)
    plt.close(fig)
    written.append(p)
    return written


def report_dataset(run_dir, dataset, ci=True, grade_first=True, workers=None):
    """Grade (optional), print and save everything for one dataset of a run. Returns the written paths."""
    run_dir = Path(run_dir)
    d = run_dir / dataset
    title = f"{run_dir.name} / {dataset}"
    if grade_first:
        import grade
        grade.grade_dir(d, grade.default_workers() if workers is None else workers)
        print()
    df = an.load_results(d)
    stamp, cap = an.file_stamp(), f"{run_dir.name}: " + an.provenance(d)
    figs, tabs = run_dir / "figures", run_dir / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)
    written = _report_frame(df, title, dataset, stamp, cap, figs, tabs, ci, by_model=True)
    if dataset == "math500" and (df["level"].astype(str) == "5").any():
        d5 = df[df["level"].astype(str) == "5"]
        print(f"----- {title}, level 5 only ({d5.problem_id.nunique()} problems) -----")
        written += _report_frame(d5, title + " level 5", "math500-l5", stamp, cap, figs, tabs, ci, by_model=False)
    print("written: " + ", ".join(str(p.relative_to(run_dir)) for p in written))
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="path to a run folder (<runs root>/<run id>)")
    ap.add_argument("--run", help="run id under the runs root, e.g. 2026-10-15_math-1.5b")
    ap.add_argument("--dataset", action="append", help="only this dataset (repeatable); default: all in the run")
    ap.add_argument("--list", action="store_true", help="list the runs under the runs root and exit")
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap bands (faster)")
    ap.add_argument("--no-grade", action="store_true", help="do not grade ungraded rows first")
    ap.add_argument("--workers", type=int, default=None, help="grading worker processes")
    args = ap.parse_args(argv)
    root = find_runs_root()
    if args.list:
        print(f"runs root: {root}")
        print("\n".join(f"  {r}  [{', '.join(datasets_in(root / r)) or 'empty'}]" for r in list_runs(root)) or "  (none)")
        return
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.run:
        run_dir = root / args.run
    else:
        ap.error("give a run folder, --run ID, or --list")
    if not run_dir.is_dir():
        sys.exit(f"no such run: {run_dir}\n(runs root: {root}; known runs: {', '.join(list_runs(root)) or 'none'}; "
                 "install Google Drive for desktop, set RUNS_ROOT, or pass the unzipped run folder)")
    dsets = args.dataset or datasets_in(run_dir)
    if not dsets:
        sys.exit(f"no dataset folders with *.jsonl under {run_dir}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for ds in dsets:
        print(f"===== {run_dir.name} / {ds} =====")
        report_dataset(run_dir, ds, ci=not args.no_ci, grade_first=not args.no_grade, workers=args.workers)
        print()


if __name__ == "__main__":
    main()
