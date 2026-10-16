# 239_project: tool atrophy pilot

Does RL with a code interpreter erode the reasoning the interpreter replaces? Three
checkpoints from one base model, evaluated with the same prompt under three ways of
handling code, on hard math where the tool matters.

## Design

| condition | what happens to code |
|---|---|
| `tool_on` | it runs (`python -I` in a temp dir, 10 s timeout, 4 calls, then a budget message) |
| `env_denied` | it does not run: the output slot says "No interpreter is available; the code was not executed" |
| `no_code` | it cannot be written: logit ban on backticks and program-only keywords ([docs/no_code_ban.md](docs/no_code_ban.md)) |

| model | arms | |
|---|---|---|
| `tool_rl` | all three | VerlTool ToRL recipe on Qwen2.5-Math-1.5B, interpreter on (360 steps) |
| `base` | all three | Qwen2.5-Math-1.5B, untrained |
| `text_rl` | `no_code` | the same VerlTool recipe, interpreter off (430 steps) |

Both RL runs share data (MATH train), algorithm (GRPO n16 b128 lr1e-6, one tool call)
and prompt; every arm uses that training prompt (`verltool_torl`, read from their
released rollouts).

- **Main test:** tool_rl vs base with the tool off (`env_denied`, `no_code`). Below
  base is atrophy: the model lost something it had.
- **Positive control:** tool_rl above base with the tool on.
- **Attribution:** RL without a tool can fall below base at large k by itself
  (Yue et al. 2504.13837), so text_rl `no_code` shows whether a deficit belongs to the tool.

Datasets, 64 samples each:

- `aime_all`: AIME 2022-2025, 120 problems, integer answers. On AIME the tool was
  worth 14 to 15 points of pass@1 to ToRL-1.5B.
- `olymmath_easy`: OlymMATH-EN-Easy, 100 AIME-level problems from printed sources,
  released 2025, after Qwen2.5-Math. The Hard split floors at 1.5B (a stronger 1.5B
  model scores 1.5% pass@1 on it), MATH-500 is too easy for the tool to matter.

Caveat: AIME 2022-2024 may be in Qwen2.5-Math's pretraining data. It affects all
three models alike, and the report lists each problem's year (`subject`) for a check.

## Pilot gates

Each stage runs only if the one before passes; numbers are pass@k with strict boxed grading.

1. **The tool matters** for tool_rl (tool_on far above env_denied and no_code). Passed:
   AIME pass@64 0.47 on vs 0.24 off.
2. **Positive control: tool_rl beats base with the tool.** FAILED for the VerlTool 360-step
   checkpoint (pass@1 .163 vs .157, pass@16 .372 vs .392, inconclusive). Gate 1 could not catch
   this, because the interpreter helps base by about 10 points too. Its tool-off scores are
   therefore not a test of atrophy. Details and every audit finding: docs/audit_2026-10-16.md.
3. Main comparison and scale-up: on hold until a tool checkpoint passes gate 2.

## Layout

| path | what |
|---|---|
| `src/gen.py` | vLLM sampler: the three conditions, tool loop, no_code ban, dataset loaders, chunked checkpoints, ungraded JSONL |
| `src/grade.py` | grades rows in place: math-verify (`correct`), boxed-only (`correct_boxed`), arithmetic check |
| `src/arith_check.py` | exact-rational check of arithmetic steps in the prose |
| `src/analyze.py` | pass@k, bootstrap CIs, summary and tool-call tables, `plot`, `verdict` |
| `src/report.py` | the local command: grade, tables, verdicts, dated figures and CSVs |
| `notebooks/tool_atrophy_pilot.ipynb` | Colab: install, Drive, clone this repo, configure, one cell per model |
| `tools/build_notebook.py` | regenerates the notebook (`uv run tools/build_notebook.py`) |
| `tests/` | laptop tests, no GPU (`for t in tests/test_*.py; do uv run $t; done`) |
| `docs/` | the no_code ban checks, the 2026-10-15 AIME analysis, earlier plans |

## Run

1. Open the notebook in VS Code (Colab extension) or Colab, on a GPU runtime (A100).
2. Run cells 1 to 6, then `run_model("tool_rl")`, `run_model("base")`,
   `run_model("text_rl")`. Cell 4 clones the repo, so the VM runs what is on GitHub.
   About 3.5 to 4 h; each model does AIME first. `SMOKE = True` is a 10-minute check.
3. Every arm checkpoints to Drive every 100 problems; finished arms are never redone.
   After a disconnect, re-run cells 1 to 6 and the model cell that was going.

Runs land on Drive as `239_project/runs/<date>_verltool-1.5b/` with `config.json`,
one folder per dataset, and `figures/` and `tables/` once reported.

## Report (laptop)

```
uv sync                                   # once
uv run src/report.py --list               # runs visible through Google Drive for desktop
uv run src/report.py --run <run id>       # grade, tables, verdicts, figures
```

Figures per dataset: `passk_models_*` (one panel per model) and `passk_<condition>_*`
(the models compared under one condition), stamped with when the data finished; the
script asks before overwriting and offers to move older sets into `old/`.
