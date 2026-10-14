# 239_project: tool atrophy pilot

Does RL with a tool erode the skill the tool replaces? One base model, three
checkpoints (base, text-only RL, tool RL), all evaluated with the tool turned off:
pass@k curves plus an arithmetic error rate. Math first, because the published
hook (CIR 2505.24480 Fig. 6) and the controls (Oat-Zero, SimpleRL-Zoo) exist there.
The full plan is in `docs/experiment1_plan.md`.

## Layout

| Path | What |
|---|---|
| `notebooks/tool_atrophy_pilot.ipynb` | the Colab notebook: install, run, analyze. Self-contained. |
| `src/gen.py` | vLLM sampler with the tool loop and the conditions below; writes JSONL |
| `src/analyze.py` | pass@k (unbiased estimator), paired bootstrap CIs, summary table, plot, verdict |
| `src/arith_check.py` | exact-rational checker for `a op b = c` in prose |
| `tools/build_notebook.py` | regenerates the notebook from `src/` (the notebook embeds the sources as `%%writefile` cells) |
| `tests/test_gen.py` | laptop tests for the tool loop, executor, grader, records |
| `docs/` | experiment plan and the two-idea memo |

## Run it in Colab

1. Upload `notebooks/tool_atrophy_pilot.ipynb` to Colab (or open it from GitHub once pushed).
2. Runtime: pick a GPU. A100 runs every preset; L4 is fine for 1.5B; T4 only 1.5B.
3. Run top to bottom with `SMOKE = True` (about 5 minutes). Results land in
   `MyDrive/239_project/results/<preset>-smoke/<dataset>/`.
4. Set `SMOKE = False` and re-run the config and run cells. There is one run
   cell per checkpoint. Each cell first looks at Drive: finished arms are loaded,
   not regenerated; an arm that was interrupted resumes from its last
   checkpointed chunk (`CHUNK` problems, default 100). After a disconnect, re-run
   the cell that was going.
5. Colab only generates (`GRADE_ON_COLAB = False`). Grade and analyze on your
   laptop, see below.

## Grade and analyze locally

Trajectories reach the laptop through Google Drive for desktop (the Drive folder
`239_project/results` appears under `~/Library/CloudStorage/GoogleDrive-<account>/My Drive/`,
mark it "available offline") or through the zip cell at the bottom of the notebook.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements-local.txt   # once
.venv/bin/python src/grade.py  "<results>/math-1.5b/math500"    # fills `correct` in place, skips graded rows
.venv/bin/python src/report.py "<results>/math-1.5b/math500"    # tables, verdicts, passk_by_model.png, passk_main.png
```

`grade.py` runs math-verify in worker processes with a per-row timeout and reports
the slowest rows. In a notebook, `import analyze as an` gives `an.load_results`,
`an.summary`, `an.passk_table`, `an.verdict` and `an.plot(df, ["model condition", ...])`.

Rough A100 cost at MATH-500 x 16 samples for the nine-arm grid: 1.5B about two
hours, 7B five to seven hours.

## Presets

| Preset | base | text-RL control | tool-RL | tool-RL prompt |
|---|---|---|---|---|
| `math-1.5b` | Qwen/Qwen2.5-Math-1.5B | sail/Qwen2.5-Math-1.5B-Oat-Zero | GAIR/ToRL-1.5B | ToRL Fig. 3 |
| `math-7b` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | RUC-AIBOX/Qwen-Math-7B-CIR | CIR Fig. 5 |
| `math-7b-torl` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | GAIR/ToRL-7B | ToRL Fig. 3 |
| `general-7b-zerotir` | Qwen/Qwen2.5-7B | hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo | htxu91/zero-tir-7b-550step | Qwen-Math TIR (unverified) |

Templates were checked on 2026-10-13 against the Oat-Zero model card, the ToRL
paper (Fig. 3), the CIR paper (Fig. 5), and the Qwen2.5-Math card. The CIR prompt's
placement in the system turn is an assumption.

## Conditions

Every model runs the whole grid, so each condition is its own three-way comparison.

| Condition | Prompt | If the model writes a code block |
|---|---|---|
| `prompt_off` | shared no-code prompt ("do not write code") | intercepted, denial injected, nothing runs |
| `env_denied` | the model's tool prompt | intercepted, denial injected, nothing runs |
| `tool_on` | the model's tool prompt | executed with `python -I`, 10 s timeout, 4-call budget |
| `plain_cot` (optional) | shared plain CoT prompt | nothing intercepted; the model fabricates its own output block (CIR's condition) |

Tool prompts: ToRL's own for ToRL, the Qwen2.5-Math TIR prompt (same instruction
sentence, chat format) for base and Oat-Zero, CIR's Fig. 5 text for the CIR
checkpoint. The denial text is "No interpreter is available; the code was not
executed. Continue reasoning without running code and give the final answer."
Calls past the budget under `tool_on` get AdaTIR's "budget exceeded" message.

## Local development

```
python -m venv .venv && . .venv/bin/activate
pip install math-verify nbformat pandas numpy matplotlib
python tests/test_gen.py           # no GPU needed
python tools/build_notebook.py     # after editing src/
```

`gen.py --help` lists every knob. Result rows carry the full generation text, so
any later metric can be computed without re-sampling.
