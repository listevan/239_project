#!/usr/bin/env python
"""Build notebooks/tool_atrophy_pilot.ipynb from src/*.py.

The notebook is self-contained for Colab: the three source files are embedded as
%%writefile cells, so src/ stays the single source of truth. Re-run this after
editing anything under src/:   python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUT = ROOT / "notebooks" / "tool_atrophy_pilot.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def writefile(name):
    body = (SRC / name).read_text().rstrip("\n")
    return code(f"%%writefile src/{name}\n{body}")


INTRO = r"""
# Tool atrophy pilot: does tool-RL erode the skill the tool replaces?

One model family, three checkpoints from the same base, evaluated **with the tool
turned off**: the base model, a tool-RL checkpoint (trained with a Python
interpreter), and a text-only RL checkpoint (same base, same kind of data, no
tool). Same prompt, same decoding, pass@k plus an arithmetic error rate.

| Readout | Meaning |
|---|---|
| tool-RL below text-RL at high k, or more arithmetic slips | **atrophy** |
| tool-RL above both at every k | **internalization** |
| tool-RL and text-RL overlap at every k | tool added nothing beyond RL, stop |

The only published hint is CIR (2505.24480) Fig. 6: one 7B model, no text-RL
control. This notebook adds the control and a direct skill metric.

**What runs.** `src/gen.py` samples one checkpoint under one or more conditions
with vLLM and writes one JSONL per (model, condition) to Google Drive, so a
disconnect costs at most one condition. `src/analyze.py` turns the JSONL into
pass@k curves, bootstrap CIs, and a verdict.

| Condition | Prompt | Code the model writes |
|---|---|---|
| `prompt_off` | shared plain CoT prompt (Qwen-Math) | never executed, logged |
| `env_denied` | the checkpoint's own tool prompt | every call returns the AdaTIR denial message |
| `tool_on` | the checkpoint's own tool prompt | executed with `python -I`, 4-call budget |

**Hardware.** Pick a GPU runtime. A100 40 GB runs everything. L4 24 GB runs the
1.5B preset comfortably and the 7B preset slowly. T4 only fits 1.5B (fp16).
Rough wall-clock on an A100 with vLLM, MATH-500 at 16 samples/problem:

| Preset | per condition | whole run (5 conditions) |
|---|---|---|
| math-1.5b | 10 to 15 min | about 1 h |
| math-7b | 30 to 45 min | 3 to 4 h |

Run the **smoke test** first (about 5 minutes). It exercises every code path on
20 problems and tells you the pipeline is wired before you spend compute units.
"""

INSTALL_MD = r"""
## 1. Install

`vllm` pins its own torch. The kernel never imports torch or vllm (generation runs
in a subprocess), so no runtime restart is needed.
"""

INSTALL = r"""
import subprocess, sys
PKGS = "vllm math-verify datasets hf_transfer pandas matplotlib"
r = subprocess.run(f"pip install -q uv && uv pip install --system -q {PKGS}", shell=True)
if r.returncode != 0:
    print("uv failed, falling back to pip")
    subprocess.run(f"pip install -q {PKGS}", shell=True, check=True)
print(subprocess.run([sys.executable, "-c", "import vllm, math_verify, datasets; print('vllm', vllm.__version__)"],
                     capture_output=True, text=True).stdout or "import check failed, see above")
"""

DRIVE_MD = r"""
## 2. Where results go

Results are written under `MyDrive/239_project/results/<run>/<dataset>/` so they
survive the session. Outside Colab this falls back to a local folder.
"""

DRIVE = r"""
import os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = "/content/drive/MyDrive/239_project"
except Exception:
    ROOT = os.path.abspath("239_project_runs")
RESULTS = f"{ROOT}/results"
os.makedirs(RESULTS, exist_ok=True)
WORK = "/content/239_project" if os.path.isdir("/content") else os.path.abspath("239_project_work")
os.makedirs(f"{WORK}/src", exist_ok=True)
os.chdir(WORK)
print("results ->", RESULTS)
print("work    ->", WORK)
"""

FILES_MD = r"""
## 3. Code

Three files, embedded from the repo's `src/`. Edit them in the repo and rebuild
the notebook with `python tools/build_notebook.py`; editing here works too for a
one-off run. `gen.py` prints its full CLI with `python src/gen.py --help`.
"""

CONFIG_MD = r"""
## 4. Configure

Presets name a (base, text-RL, tool-RL) trio from one base model. Prompt
templates were checked against the model cards and papers on 2026-10-13:

| Preset | base | text-RL (control) | tool-RL | tool prompt |
|---|---|---|---|---|
| `math-1.5b` | Qwen/Qwen2.5-Math-1.5B | sail/Qwen2.5-Math-1.5B-Oat-Zero | GAIR/ToRL-1.5B | ToRL Fig. 3, verbatim |
| `math-7b` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | RUC-AIBOX/Qwen-Math-7B-CIR (the Fig. 6 model) | CIR Fig. 5 text (system turn assumed) |
| `math-7b-torl` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | GAIR/ToRL-7B | ToRL Fig. 3 |
| `general-7b-zerotir` | Qwen/Qwen2.5-7B | hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo | htxu91/zero-tir-7b-550step | Qwen-Math TIR (unverified) |

Context is capped at 4096 tokens for the Qwen2.5-Math trios because that is the
base model's and Oat-Zero's native context (CIR evaluated at 16k by extrapolating;
`ALLOW_LONG=True` reproduces that). Truncation rate is reported either way.

`BASE_TOOL_ON=True` adds a cheap fourth arm, the base model with the tool prompt
and execution on (CIR's `Base_code`).
"""

CONFIG = r"""
PRESETS = {
    "math-1.5b": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-1.5B",            native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-1.5B-Oat-Zero",   native="qwen_math_cot"),
        "tool_rl": dict(model="GAIR/ToRL-1.5B",                     native="torl"),
    }),
    "math-7b": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-7B",              native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-7B-Oat-Zero",     native="qwen_math_cot"),
        "tool_rl": dict(model="RUC-AIBOX/Qwen-Math-7B-CIR",        native="cir"),
    }),
    "math-7b-torl": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-7B",              native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-7B-Oat-Zero",     native="qwen_math_cot"),
        "tool_rl": dict(model="GAIR/ToRL-7B",                       native="torl"),
    }),
    "general-7b-zerotir": dict(max_model_len=8192, models={
        "base":    dict(model="Qwen/Qwen2.5-7B",                   native="qwen_math_tir"),
        "text_rl": dict(model="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo", native="qwen_math_cot"),
        "tool_rl": dict(model="htxu91/zero-tir-7b-550step",         native="qwen_math_tir"),
    }),
}

PRESET = "math-1.5b"       # start here; "math-7b" once the 1.5B run is clean
DATASET = "math500"        # math500 | aime24 | aime25
SMOKE = True               # True: 20 problems x 4 samples x 1024 tokens, ~5 min. False: the real run.

N_PROBLEMS = 20 if SMOKE else 0        # 0 = all 500 (AIME sets have 30)
N_SAMPLES = 4 if SMOKE else 16         # raise to 32 or 64 for a sharper high-k tail
MAX_TOKENS = 1024 if SMOKE else 3000
TEMPERATURE, TOP_P, SEED = 0.6, 0.95, 0   # CIR Fig. 6 settings
MAX_CALLS = 4
ALLOW_LONG = False         # True + MAX_MODEL_LEN=16384 mimics CIR's 16k budget on 4k-native models
MAX_MODEL_LEN = PRESETS[PRESET]["max_model_len"]
BASE_TOOL_ON = False
SHARED_TEMPLATE = "qwen_math_cot"
GPU_MEM = 0.90

CONDITIONS = {
    "base":    ["prompt_off"] + (["tool_on"] if BASE_TOOL_ON else []),
    "text_rl": ["prompt_off"],
    "tool_rl": ["prompt_off", "env_denied", "tool_on"],
}
RUN_NAME = f"{PRESET}{'-smoke' if SMOKE else ''}"
OUT = f"{RESULTS}/{RUN_NAME}"
print(RUN_NAME, "->", OUT)
"""

RUN_MD = r"""
## 5. Run

One subprocess per checkpoint: it loads the model once, runs its conditions, and
writes `<tag>__<condition>__<template>.jsonl` plus a `.meta.json` with timings.
Finished files are skipped, so after a disconnect just re-run this cell. Expect
the first run of each model to spend a few minutes downloading weights.
"""

RUN = r"""
import subprocess, sys, time
for tag, spec in PRESETS[PRESET]["models"].items():
    cmd = [sys.executable, "src/gen.py",
           "--model", spec["model"], "--tag", tag,
           "--conditions", ",".join(CONDITIONS[tag]),
           "--shared-template", SHARED_TEMPLATE, "--native-template", spec["native"],
           "--dataset", DATASET, "--n-problems", str(N_PROBLEMS), "--n", str(N_SAMPLES),
           "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
           "--temperature", str(TEMPERATURE), "--top-p", str(TOP_P), "--seed", str(SEED),
           "--max-calls", str(MAX_CALLS), "--gpu-mem", str(GPU_MEM), "--out", OUT]
    if ALLOW_LONG:
        cmd.append("--allow-long")
    print("\n$", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"[{tag}] exit {r.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode != 0:
        print("stopping: fix the error above, then re-run this cell (finished conditions are skipped)")
        break
"""

ANALYZE_MD = r"""
## 6. Analyze

`summary` is the headline table. `trunc_pct`, `no_boxed_pct`, `code_pct` and
`fake_out_pct` are the failure modes that would make a pass@k gap uninterpretable,
so read them before the curves. `arith_err_pct` is the direct skill metric
(exact-rational check of every `a op b = c` in the prose, code blocks excluded);
`arith_coverage_pct` says how many samples contained anything to check.
"""

ANALYZE = r"""
import sys, importlib
sys.path.insert(0, "src")
import analyze as an
importlib.reload(an)
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

df = an.load_results(f"{OUT}/{DATASET}")
print(len(df), "samples,", df.problem_id.nunique(), "problems,", df.arm.nunique(), "arms")
an.summary(df)
"""

PLOT = r"""
an.passk_table(df).round(3)
"""

PLOT2 = r"""
ax = an.plot_passk(df, title=f"{RUN_NAME} / {DATASET}: pass@k, tool off unless marked")
ax.figure.savefig(f"{OUT}/{DATASET}/passk.png", bbox_inches="tight")
"""

VERDICT = r"""
an.verdict(df, "prompt_off")
print()
an.verdict(df, "env_denied") if "env_denied" in set(df.condition) else None
"""

NEXT_MD = r"""
## 7. Reading the result

- **prompt_off vs env_denied for tool_rl.** A deficit under `prompt_off` that
  vanishes under `env_denied` is prompt mismatch, not lost ability. A deficit under
  both is the real thing.
- **Compare to text_rl, not just base.** RL alone sharpens pass@1 and can lower
  high-k pass@k (the sharpening confound). The question is whether tool-RL sits
  below text-RL.
- **Check failure modes first.** If tool_rl has 30% truncation or writes code in
  every `prompt_off` sample, its curve measures format collapse.
- **Arithmetic error rate** is the skill the interpreter replaces. Tool-RL above
  text-RL here, with the same accuracy, is atrophy of exactly that skill.

Then: flip `SMOKE=False`, raise `N_SAMPLES` to 32 or 64 for the high-k tail, add
`aime24`/`aime25` with `N_SAMPLES=64`, and move to `math-7b` (the CIR checkpoint
reproduces Fig. 6 with the control added). Per-subject breakdowns are one
`df.groupby(["tag","condition","subject"]).correct.mean()` away.

**Caveats.** `tool_on` executes model-written Python on this VM with a 10 s
timeout; fine for Colab, not for a shared machine. The CIR prompt's turn placement
is assumed. The ZeroTIR prompt is unverified (its repo has no card), so trust only
its `prompt_off` row. Qwen2.5-Math's native context is 4096, so long solutions
truncate; the rate is in the table.
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [
        md(INTRO),
        code("!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"),
        md(INSTALL_MD), code(INSTALL),
        md(DRIVE_MD), code(DRIVE),
        md(FILES_MD), writefile("arith_check.py"), writefile("gen.py"), writefile("analyze.py"),
        md(CONFIG_MD), code(CONFIG),
        md(RUN_MD), code(RUN),
        md(ANALYZE_MD), code(ANALYZE), code(PLOT), code(PLOT2), code(VERDICT),
        md(NEXT_MD),
    ]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
