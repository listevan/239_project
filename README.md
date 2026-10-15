# 239_project: tool atrophy pilot

Does RL with a tool erode the skill the tool replaces? One base model, three
checkpoints (base, text-only RL, tool RL), all evaluated with the tool turned off:
pass@k curves plus an arithmetic error rate. Math first, because the published
hook (CIR 2505.24480 Fig. 6) and the controls (Oat-Zero, SimpleRL-Zoo) exist there.
The full plan is in `docs/experiment1_plan.md`.

## Layout

| Path | What |
|---|---|
| `notebooks/tool_atrophy_pilot.ipynb` | the Colab notebook: install, clone this repo onto the VM, configure, run |
| `src/gen.py` | vLLM sampler with the tool loop and the conditions below; writes JSONL |
| `src/analyze.py` | pass@k (unbiased estimator), paired bootstrap CIs, summary table, plot, verdict |
| `src/arith_check.py` | exact-rational checker for `a op b = c` in prose |
| `tools/build_notebook.py` | regenerates the notebook from its cell templates (code is cloned, not embedded) |
| `tests/test_gen.py` | laptop tests for the tool loop, executor, grader, records |
| `docs/` | experiment plan and the two-idea memo |

## Run it in Colab

1. Open `notebooks/tool_atrophy_pilot.ipynb` from this repo in VS Code (Colab
   extension) or upload it to Colab. Runtime: a GPU. A100 runs every preset; L4
   is fine for 1.5B; T4 only 1.5B.
2. Run the cells top to bottom. Cell 3 clones or pulls the repo (public, no
   token needed; a private fork can supply one, see the cell), so the VM runs
   whatever is on GitHub: push local edits first. The config cell defaults to
   the real run (`SMOKE = True` is a 10-minute wiring check).
3. One run cell per checkpoint. Each cell first looks at Drive: finished arms
   are loaded, not regenerated; an arm that was interrupted resumes from its
   last checkpointed chunk (`CHUNK` problems, default 100). After a disconnect,
   re-run the cell that was going.
5. Colab only generates (`GRADE_ON_COLAB = False`). Grade and analyze on your
   laptop, see below.

## Where runs live

One folder per run on Drive, named by date and preset, under `239_project/runs/`:

```
runs/2026-10-15_math-1.5b/
    config.json                      the notebook settings that produced it
    aime24/ aime25/                  one <model>__<condition>__<template>.jsonl + .meta.json per arm
    figures/  tables/                written by report.py, every file dated
```

`DATASETS` in the notebook maps each dataset to (problems, samples). The pilot
is AIME24 and AIME25 at 64 samples, the hard end where the interpreter should
matter. Also available, one line each: `gsm_hard` (GSM8K with huge numbers,
1016 integer-answer problems), `math500_l5` (the 134 level-5 problems) and
`math500` (all 500; not tool-load-bearing at 1.5B). `RUN_ID` defaults to
today's date plus the preset; set it to an existing run to resume or extend one.

## Grade and analyze locally

Trajectories reach the laptop through Google Drive for desktop (the `239_project`
folder may sit anywhere up to five levels under My Drive; both the notebook and
`report.py` find it) or through the zip cell at the bottom of the notebook.

```
uv sync                                       # once: .venv from pyproject.toml (uv.lock pins it)
uv run src/report.py --list                   # runs visible from this machine
uv run src/report.py --run 2026-10-15_math-1.5b            # every dataset in the run
uv run src/report.py --run 2026-10-15_math-1.5b --dataset gsm_hard
```

For each dataset, `report.py` grades ungraded rows in place first (`src/grade.py`:
math-verify in worker processes with a per-row timeout, slowest rows reported,
graded rows skipped), prints the summary, pass@k, tool-call and verdict tables,
saves them as CSV under `<run>/tables/`, and saves `passk_by_model_<dataset>_<date>.png`
and `passk_main_<dataset>_<date>.png` under `<run>/figures/`, each captioned with
when its arms were generated. In a notebook, `import analyze as an` gives
`an.load_results`, `an.summary`, `an.calls_table`, `an.passk_table`, `an.verdict`
and `an.plot(df, ["model condition", ...])`.
