#!/usr/bin/env python
"""Grade (if needed) and report a results folder, on any machine. The one local command:

    python src/report.py --run math-1.5b-smoke [--dataset math500]
    python src/report.py <results>/<run>/<dataset>            # explicit folder

--run finds the folder under the synced Google Drive for desktop
(~/Library/CloudStorage/GoogleDrive-*/My Drive/239_project/results), else $RESULTS_ROOT, else
./results. Ungraded rows are graded first with src/grade.py (skip with --no-grade). Then: the summary
table, the pass@k table, the verdict for every condition, and passk_by_model.png (one panel per model,
conditions as lines) plus passk_main.png (tool-RL with the tool off against the control and the base),
saved next to the data.
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


def find_results_root():
    """$RESULTS_ROOT, else the Drive for desktop mirror of MyDrive/239_project/results, else ./results."""
    if os.environ.get("RESULTS_ROOT"):
        return Path(os.environ["RESULTS_ROOT"])
    for p in sorted(Path.home().glob("Library/CloudStorage/GoogleDrive-*/My Drive/239_project/results")):
        return p
    return Path(__file__).resolve().parents[1] / "results"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir", nargs="?", help="explicit <run>/<dataset> folder")
    ap.add_argument("--run", help="run name, e.g. math-1.5b-smoke (folder found under the synced Drive)")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--title", default=None, help="figure title prefix (default: run / dataset)")
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap bands (faster)")
    ap.add_argument("--no-grade", action="store_true", help="do not grade ungraded rows first")
    ap.add_argument("--workers", type=int, default=None, help="grading worker processes")
    args = ap.parse_args(argv)
    if args.results_dir:
        d = Path(args.results_dir)
    elif args.run:
        d = find_results_root() / args.run / args.dataset
    else:
        ap.error("give a results folder or --run NAME")
    if not d.is_dir():
        sys.exit(f"no such folder: {d}\n(results root: {find_results_root()}; install Google Drive for desktop, "
                 f"set RESULTS_ROOT, or pass the unzipped folder explicitly)")
    title = args.title or f"{d.parent.name} / {d.name}"
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

    if not args.no_grade:
        import grade
        grade.grade_dir(d, grade.default_workers() if args.workers is None else args.workers)
        print()
    df = an.load_results(d)
    print(f"{len(df)} samples, {df.problem_id.nunique()} problems, {df.arm.nunique()} arms\n")
    print(an.summary(df).to_string(index=False), "\n")
    print(an.passk_table(df).round(3).to_string(index=False), "\n")
    for cond in an.COND_ORDER:
        if cond in set(df.condition):
            an.verdict(df, cond)
            print()
    ci = not args.no_ci
    an.plot(df, BY_MODEL, color="condition", ci=ci, title=f"{title}: pass@k within each model",
            save=d / "passk_by_model.png")
    an.plot(df, MAIN, ci=ci, title=f"{title}: tool-RL with the tool off vs control and base",
            save=d / "passk_main.png")
    print(f"figures -> {d / 'passk_by_model.png'}, {d / 'passk_main.png'}")


if __name__ == "__main__":
    main()
