# 239_project: tool atrophy pilot

Does RL with a code interpreter erode the reasoning the interpreter replaces? A tool-RL
checkpoint against the untrained model it started from, under three ways of handling code,
on math where the tool helps but cannot replace the reasoning.

## Design

| condition | prompt | what happens to code |
|---|---|---|
| `tool_on` | the model's tool prompt | it runs (`python -I` in a temp dir, 10 s timeout, 4 calls, then a budget message) |
| `env_denied` | the model's tool prompt | it does not run: the output slot says "No interpreter is available; the code was not executed" |
| `plain` | the bare question, no wrapper | it cannot be written: logit ban on backticks and program-only keywords ([docs/no_code_ban.md](docs/no_code_ban.md)) |

| model | tool prompt | |
|---|---|---|
| `tool_rl` | `torl` (ToRL's own, verbatim) | GAIR/ToRL-1.5B, RL with the interpreter |
| `base` | `qwen_math_tir` | Qwen/Qwen2.5-Math-1.5B, untrained (ToRL's starting point) |
| `tool_sft` | `verltool_sft` (its SFT data's system prompt) | VerlTool/Qwen2.5-Math-1.5B-TIR-SFT-new: SFT on OpenMathReasoning TIR traces, from the same base (checked by weight distance) |
| `text_rl` | none for now | Oat-Zero, RL without a tool: only if tool_rl lands below base |

- **Main test:** tool_rl vs base with the tool off (`env_denied`, `plain`). Below base is
  atrophy: the model lost something it had.
- **Positive control:** tool_rl above base with the tool on.
- `plain` uses the bare question because it is this base model's best prompt: Liu et al.
  (arXiv 2503.20783, Table 1) report Qwen2.5-Math-1.5B at AMC 43.4 with no template vs 32.5
  with the Qwen chat template (greedy, 3000 tokens, boxed-only grading). Under the chat
  template base derails and loops in ~43% of samples, which RL of any kind fixes, so a
  handicapped base would hide atrophy.

Pilot set: `math500_l5`, MATH-500 level 5 minus the 10 problems found in ToRL's training data
(124 left), 30 sampled x 4 samples (pass@1 to pass@4). The first pilot set, `amc24` (AMC 12
2024), left base too weak tool-free.

Known limitation: Qwen2.5-Math already reasons in Python-style code without running it, and
RL amplifies that even with random rewards (Shao et al., arXiv 2506.10947). Nearly every
released tool-RL math checkpoint is Qwen2.5-based, so no non-Qwen pair exists at this size.

Why `tool_sft`: SFT on tool traces is known to cut tool-free accuracy (Cheng et al., arXiv
2605.06326: 54.3 -> 34.1 on Qwen3-4B, with the model writing code and inventing its output), while RL
does not. Untested: whether telling the model there is no interpreter (`env_denied`) or blocking code
(`plain`) recovers it, i.e. whether the drop is a habit or lost skill.

## Pilot gates

Each stage runs only if the one before passes; numbers are pass@k with strict boxed grading.

1. **Positive control: tool_rl beats base with the tool.** ToRL-1.5B passes (AIME24 .272 vs
   .123; AMC24 .508 vs .117). The VerlTool 360-step checkpoint failed it (.163 vs .157) and was
   dropped (docs/audit_2026-10-16.md).
2. **Base is capable tool-free** (`plain` pass@1 about .25 or more), so a drop is measurable.
   AMC24: .150 with the chat prompt, .175 with the bare question (ToRL .183, a tie): failed,
   base is too weak there. Next: MATH-500 level 5.
3. Main comparison and scale-up only after both pass.

## Contamination check (2026-10-16)

ToRL's released training set (`data/torl_data/train.parquet`, 28,740 problems) searched by normalised text
for every evaluation problem: AMC 2023 has 13 exact and about 9 reworded copies of its 40 problems, so it is
not used; AMC 12 2024 (44 problems), AIME 2024 and AIME 2025 have none (their few substring hits are stock
phrases such as "m and n are relatively prime"). The 2024 contests also postdate Qwen2.5-Math, so the pilot
set is unseen by both models.

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
