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
    "base":    ["base prompt_off",    "base env_denied",    "base prose_only",    "base tool_on"],
    "text_rl": ["text_rl prompt_off", "text_rl env_denied", "text_rl prose_only", "text_rl tool_on"],
    "tool_rl": ["tool_rl prompt_off", "tool_rl env_denied", "tool_rl prose_only", "tool_rl tool_on"],
}
# Three headline figures. Each tool-off one shows the three models under the same tool-off condition
# plus the base with the tool on; the tool-on one is the dose check: are the two RL models comparable
# when each has the interpreter?
MAIN = {
    "denied":    ["tool_rl env_denied", "text_rl env_denied", "base env_denied", "base tool_on"],
    "promptoff": ["tool_rl prompt_off", "text_rl prompt_off", "base prompt_off", "base tool_on"],
    "prose":     ["tool_rl prose_only", "text_rl prose_only", "base prose_only", "base tool_on"],
    "toolon":    ["tool_rl tool_on", "text_rl tool_on", "base tool_on"],
}
MAIN_TITLES = {"denied": "tool denied, tool-RL vs text-RL vs base",
               "promptoff": "told not to code, tool-RL vs text-RL vs base",
               "prose": "no code possible, tool-RL vs text-RL vs base",
               "toolon": "with the tool, tool-RL vs text-RL vs base"}


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


def _outputs(df, tag, stamp, figs, tabs, by_model):
    """Every file one dataset frame produces, name -> path, before anything is written."""
    out = {f"table:{name}": tabs / f"{name}_{tag}_{stamp}.csv" for name in ("summary", "passk", "calls")}
    if by_model:
        out["fig:by_model"] = figs / f"passk_by_model_{tag}_{stamp}.png"
    present = set(df.tag + " " + df.condition)
    for key, arms in MAIN.items():
        if any(a in present for a in arms[:3]):   # skip e.g. the prose figure when no prose_only arm exists
            out[f"fig:{key}"] = figs / f"passk_main_{key}_{tag}_{stamp}.png"
    return out


def _report_frame(df, title, tag, stamp, cap, figs, tabs, ci, by_model, write=True):
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
    outs = _outputs(df, tag, stamp, figs, tabs, by_model)
    if not write:
        return []
    for name, t in tables.items():
        t.to_csv(outs[f"table:{name}"], index=False)
    if "fig:by_model" in outs:
        fig = an.plot(df, BY_MODEL, color="condition", ci=ci, title=f"{title}: pass@k within each model",
                      save=outs["fig:by_model"], caption=cap)
        plt.close(fig)
    for key, arms in MAIN.items():
        if f"fig:{key}" in outs:
            fig = an.plot(df, arms, ci=ci, title=f"{title}: {MAIN_TITLES[key]}", save=outs[f"fig:{key}"], caption=cap)
            plt.close(fig)
    return list(outs.values())


def confirm(question, yes):
    """y/n on a tty; --yes answers yes everywhere; with no tty and no --yes the answer is no."""
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{question} -> no (no terminal; pass --yes to answer yes)")
        return False
    return input(f"{question} [y/N] ").strip().lower() in ("y", "yes")


def superseded(run_dir, dataset, stamp):
    """Output files for this dataset carrying a different stamp (figures and tables)."""
    import re
    pat = re.compile(rf"^[a-z_]+_{re.escape(dataset)}(-l5)?_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}})\.(png|csv)$")
    found = []
    for sub in ("figures", "tables"):
        for p in sorted((Path(run_dir) / sub).glob("*")):
            m = pat.match(p.name)
            if m and m.group(2) != stamp:
                found.append(p)
    return found


def report_dataset(run_dir, dataset, ci=True, grade_first=True, workers=None, yes=False):
    """Grade (optional), print and save everything for one dataset of a run. Output files are stamped
    with the data's own time (newest arm's finished_at), so an unchanged dataset maps to the same
    file names: existing ones are overwritten after a y/n, and sets with other stamps are offered a
    move into old/. Returns the written paths."""
    run_dir = Path(run_dir)
    d = run_dir / dataset
    title = f"{run_dir.name} / {dataset}"
    if grade_first:
        import grade
        grade.grade_dir(d, grade.default_workers() if workers is None else workers)
        print()
    df = an.load_results(d)
    stamp, cap = an.data_stamp(d), f"{run_dir.name}: " + an.provenance(d)
    figs, tabs = run_dir / "figures", run_dir / "tables"
    figs.mkdir(exist_ok=True)
    tabs.mkdir(exist_ok=True)
    frames = [(df, title, dataset, True)]
    if dataset == "math500" and (df["level"].astype(str) == "5").any():
        d5 = df[df["level"].astype(str) == "5"]
        frames.append((d5, title + " level 5", "math500-l5", False))
    planned = [p for f, _, tag, bm in frames for p in _outputs(f, tag, stamp, figs, tabs, bm).values()]
    existing = [p for p in planned if p.exists()]
    write = True
    if existing:
        write = confirm(f"{len(existing)} output file(s) for data stamp {stamp} already exist in {run_dir.name}; overwrite them?", yes)
    written = []
    for f, ttl, tag, bm in frames:
        if tag == "math500-l5":
            print(f"----- {title}, level 5 only ({f.problem_id.nunique()} problems) -----")
        written += _report_frame(f, ttl, tag, stamp, cap, figs, tabs, ci, bm, write=write)
    print(("written: " + ", ".join(str(p.relative_to(run_dir)) for p in written)) if written
          else f"nothing written (kept the existing {stamp} files)")
    old = superseded(run_dir, dataset, stamp)
    if old:
        stamps = sorted({p.name.rsplit("_", 2)[-2] + "_" + p.name.rsplit("_", 2)[-1].split(".")[0] for p in old})
        if confirm(f"{len(old)} older output file(s) for {dataset} (stamps {', '.join(stamps)}); move them into old/?", yes):
            for p in old:
                dest = p.parent / "old"
                dest.mkdir(exist_ok=True)
                p.rename(dest / p.name)
            print(f"moved {len(old)} file(s) into figures/old and tables/old")
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
                 "install Google Drive for desktop, set RUNS_ROOT, or pass the unzipped run folder)")
    dsets = args.dataset or datasets_in(run_dir)
    if not dsets:
        sys.exit(f"no dataset folders with *.jsonl under {run_dir}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    for ds in dsets:
        print(f"===== {run_dir.name} / {ds} =====")
        report_dataset(run_dir, ds, ci=not args.no_ci, grade_first=not args.no_grade, workers=args.workers, yes=args.yes)
        print()


if __name__ == "__main__":
    main()
