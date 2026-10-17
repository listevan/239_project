#!/usr/bin/env python
"""Grade (if needed) and report a run, on any machine. The one local command:

    uv run src/report.py --list                                   # runs visible from this machine
    uv run src/report.py --run 2026-10-16_verltool-1.5b           # every dataset in the run
    uv run src/report.py --run <run id> --dataset aime_all
    uv run src/report.py <path to a run folder>

Runs live in <runs root>/<run id>/<dataset>/*.jsonl. The runs root is $RUNS_ROOT, else the Google
Drive for desktop mirror of 239_project/runs (any depth under My Drive), else ./runs.

Per dataset: ungraded rows are graded in place (src/grade.py), then the summary, pass@k and tool-call
tables and the verdicts are printed and saved under <run>/tables/, and the figures under <run>/figures/:
  passk_models_<dataset>_<stamp>.png       one panel per model, its conditions as lines
  passk_<condition>_<dataset>_<stamp>.png  the models compared under one condition
The stamp is the data's own time (newest arm's finish, local), so an unchanged dataset maps to the same
names: you are asked before they are overwritten, and sets with other stamps are offered a move into
old/ (--yes answers yes to both).
"""
import argparse
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402
import analyze as an  # noqa: E402

COMPARE_TITLES = {
    "no_code": "same prompt, code blocked: tool model vs base (main test)",
    "env_denied": "tool off, code not run: tool model vs base (main test)",
    "tool_on": "tool on: tool model vs base (positive control)",
    "plain": "bare question, code banned: tool model vs base (main test)",
}


def find_runs_root(home=None):
    """$RUNS_ROOT, else a 239_project/runs folder up to five levels under any Google Drive for desktop
    mount (shallowest first), else ./runs in the repo."""
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


def confirm(question, yes):
    """y/n on a terminal; --yes answers yes; with no terminal and no --yes the answer is no."""
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> no (no terminal; pass --yes to answer yes)")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def figures_for(df):
    """{name: (lines, title_suffix, color)} for the figures this frame supports."""
    present = set(df.tag + " " + df.condition)
    tags = sorted(set(df.tag), key=an.role_rank)
    conds = sorted(set(df.condition), key=lambda c: (an.cond_rank(c), c))
    figs = {"models": ({t: [f"{t} {c}" for c in conds if f"{t} {c}" in present] for t in tags},
                       "pass@k within each model", "condition")}
    for c in conds:
        lines = [f"{t} {c}" for t in tags if f"{t} {c}" in present]
        if len(lines) >= 2:
            figs[c] = (lines, COMPARE_TITLES.get(c, f"{c}: models compared"), "model")
    return figs


def outputs(df, dataset, stamp, figs_dir, tabs_dir):
    out = {f"table:{n}": tabs_dir / f"{n}_{dataset}_{stamp}.csv" for n in ("summary", "passk", "calls")}
    out.update({f"fig:{k}": figs_dir / f"passk_{k}_{dataset}_{stamp}.png" for k in figures_for(df)})
    return out


def superseded(run_dir, dataset, stamp, current=()):
    """Output files for this dataset that the current report does not produce: another stamp, or the same
    stamp under a name no longer written (e.g. figures from an older version of this script)."""
    pat = re.compile(rf"^[a-z_]+_{re.escape(dataset)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})\.(png|csv)$")
    keep = {Path(c).name for c in current}
    return [p for sub in ("figures", "tables") for p in sorted((Path(run_dir) / sub).glob("*"))
            if (m := pat.match(p.name)) and (m.group(1) != stamp or (keep and p.name not in keep))]


def report_dataset(run_dir, dataset, ci=True, grade_first=True, workers=None, yes=False):
    """Grade (optional), print and save everything for one dataset of a run. Returns the written paths."""
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    d = run_dir / dataset
    if grade_first:
        import grade
        grade.grade_dir(d, grade.default_workers() if workers is None else workers)
        print()
    df = an.load_results(d)
    stamp, cap = an.data_stamp(d), f"{run_dir.name}: " + an.provenance(d)
    figs_dir, tabs_dir = run_dir / "figures", run_dir / "tables"
    figs_dir.mkdir(exist_ok=True)
    tabs_dir.mkdir(exist_ok=True)

    print(f"{len(df)} samples, {df.problem_id.nunique()} problems, {df.arm.nunique()} arms\n")
    tables = dict(summary=an.summary(df), passk=an.passk_table(df).round(3), calls=an.calls_table(df))
    for name, t in tables.items():
        print(f"{name}:\n{t.to_string(index=False)}\n")
    for cond in sorted(set(df.condition), key=lambda c: (an.cond_rank(c), c)):
        an.verdict(df, cond)
        bc = an.best_case(df, cond)
        if not bc.empty:
            print(f"  best case, pass@1 on the samples that are reasoning rather than artifacts (an upper bound):")
            print("    " + bc[["view", "contrast", "diff", "lo", "hi", "p"]].to_string(index=False).replace("\n", "\n    "))
        print()

    outs = outputs(df, dataset, stamp, figs_dir, tabs_dir)
    existing = [p for p in outs.values() if p.exists()]
    if existing and not confirm(f"{len(existing)} output file(s) for data stamp {stamp} already exist in "
                                f"{run_dir.name}; overwrite them?", yes):
        print(f"nothing written (kept the existing {stamp} files)")
        written = []
    else:
        for name, t in tables.items():
            t.to_csv(outs[f"table:{name}"], index=False)
        for key, (lines, suffix, color) in figures_for(df).items():
            fig = an.plot(df, lines, color=color, ci=ci, title=f"{run_dir.name} / {dataset}: {suffix}",
                          save=outs[f"fig:{key}"], caption=cap)
            plt.close(fig)
        written = list(outs.values())
        print("written: " + ", ".join(str(p.relative_to(run_dir)) for p in written))
    old = superseded(run_dir, dataset, stamp, outs.values())
    if old and confirm(f"{len(old)} older output file(s) for {dataset}; move them into old/?", yes):
        for p in old:
            (p.parent / "old").mkdir(exist_ok=True)
            p.rename(p.parent / "old" / p.name)
        print(f"moved {len(old)} file(s) into figures/old and tables/old")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="path to a run folder (<runs root>/<run id>)")
    ap.add_argument("--run", help="run id under the runs root")
    ap.add_argument("--dataset", action="append", help="only this dataset (repeatable); default: all in the run")
    ap.add_argument("--list", action="store_true", help="list the runs under the runs root and exit")
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap bands (faster)")
    ap.add_argument("--no-grade", action="store_true", help="do not grade ungraded rows first")
    ap.add_argument("--workers", type=int, default=None, help="grading worker processes")
    ap.add_argument("--yes", "-y", action="store_true", help="answer yes to overwrite and move-to-old questions")
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
                 "install Google Drive for desktop, set RUNS_ROOT, or pass the run folder)")
    dsets = args.dataset or datasets_in(run_dir)
    if not dsets:
        sys.exit(f"no dataset folders with *.jsonl under {run_dir}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for ds in dsets:
        print(f"===== {run_dir.name} / {ds} =====")
        report_dataset(run_dir, ds, ci=not args.no_ci, grade_first=not args.no_grade, workers=args.workers,
                       yes=args.yes)
        print()


if __name__ == "__main__":
    main()
