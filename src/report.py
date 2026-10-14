#!/usr/bin/env python
"""Tables and figures from a graded results folder, on any machine.

    python src/report.py results/<run>/<dataset> [--title T] [--no-ci]

Prints the summary table, the pass@k table and the verdict for every condition, and saves
passk_by_model.png (one panel per model, conditions as lines) and passk_main.png (tool-RL with the
tool off against the text-RL control and the base) next to the results.
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--title", default=None, help="figure title prefix (default: the folder name)")
    ap.add_argument("--no-ci", action="store_true", help="skip the bootstrap bands (faster)")
    args = ap.parse_args(argv)
    d = Path(args.results_dir)
    title = args.title or f"{d.parent.name} / {d.name}"
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

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
