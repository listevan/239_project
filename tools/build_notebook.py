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

**What runs.** `src/gen.py` samples one checkpoint under each condition with vLLM
and writes one JSONL per (model, condition) to Google Drive, checkpointing every
`CHUNK` problems, so a disconnect costs at most one chunk. `src/analyze.py` turns the JSONL into pass@k curves,
bootstrap CIs, and a verdict. Every model gets the same grid, so each condition is
read as its own three-way comparison.

| Condition | Prompt | If the model writes a code block |
|---|---|---|
| `prompt_off` | shared no-code prompt ("do not write code") | intercepted, denial message injected, nothing runs |
| `env_denied` | the model's tool prompt (asks for code) | intercepted, denial message injected, nothing runs |
| `tool_on` | the model's tool prompt | executed with `python -I`, 4-call budget, then the budget message |
| `plain_cot` (optional) | shared plain CoT prompt | nothing intercepted; the model fabricates its own output block. CIR's condition. |

**Hardware.** Pick a GPU runtime. A100 40 GB runs everything. L4 24 GB runs the
1.5B preset comfortably and the 7B preset slowly. T4 only fits 1.5B (fp16).
Rough wall-clock on an A100 with vLLM, MATH-500 at 16 samples/problem:

| Preset | per arm | whole grid (9 arms) |
|---|---|---|
| math-1.5b | 10 to 15 min | about 2 h |
| math-7b | 30 to 45 min | 5 to 7 h |

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

Presets name a (base, text-RL, tool-RL) trio from one base model. Each model also
carries its tool prompt: ToRL's own (paper Fig. 3) for ToRL, the Qwen2.5-Math TIR
prompt (same instruction sentence, chat format) for base and Oat-Zero, CIR's Fig. 5
text for the CIR checkpoint. Set every `native` to the same key if you want one
identical tool prompt for all three.

| Preset | base | text-RL (control) | tool-RL |
|---|---|---|---|
| `math-1.5b` | Qwen/Qwen2.5-Math-1.5B | sail/Qwen2.5-Math-1.5B-Oat-Zero | GAIR/ToRL-1.5B |
| `math-7b` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | RUC-AIBOX/Qwen-Math-7B-CIR (the Fig. 6 model) |
| `math-7b-torl` | Qwen/Qwen2.5-Math-7B | sail/Qwen2.5-Math-7B-Oat-Zero | GAIR/ToRL-7B |
| `general-7b-zerotir` | Qwen/Qwen2.5-7B | hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo | htxu91/zero-tir-7b-550step (tool prompt unverified) |

Context is capped at 4096 tokens for the Qwen2.5-Math trios because that is the
base model's and Oat-Zero's native context (CIR evaluated at 16k by extrapolating;
`ALLOW_LONG=True` reproduces that). Truncation rate is reported either way.

`PLAIN_COT=True` adds the fourth, non-intercepting condition for a tie to CIR's
numbers. `DENIAL_TEXT` overrides the message injected under `prompt_off` and
`env_denied`.
"""

CONFIG = r"""
PRESETS = {
    "math-1.5b": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-1.5B",            native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-1.5B-Oat-Zero",   native="qwen_math_tir"),
        "tool_rl": dict(model="GAIR/ToRL-1.5B",                     native="torl"),
    }),
    "math-7b": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-7B",              native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-7B-Oat-Zero",     native="qwen_math_tir"),
        "tool_rl": dict(model="RUC-AIBOX/Qwen-Math-7B-CIR",        native="cir"),
    }),
    "math-7b-torl": dict(max_model_len=4096, models={
        "base":    dict(model="Qwen/Qwen2.5-Math-7B",              native="qwen_math_tir"),
        "text_rl": dict(model="sail/Qwen2.5-Math-7B-Oat-Zero",     native="qwen_math_tir"),
        "tool_rl": dict(model="GAIR/ToRL-7B",                       native="torl"),
    }),
    "general-7b-zerotir": dict(max_model_len=8192, models={
        "base":    dict(model="Qwen/Qwen2.5-7B",                   native="qwen_math_tir"),
        "text_rl": dict(model="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo", native="qwen_math_tir"),
        "tool_rl": dict(model="htxu91/zero-tir-7b-550step",         native="qwen_math_tir"),
    }),
}

PRESET = "math-1.5b"       # start here; "math-7b" once the 1.5B run is clean
DATASET = "math500"        # math500 | aime24 | aime25
SMOKE = True               # True: 20 problems x 4 samples x 1024 tokens, ~10 min. False: the real run.

N_PROBLEMS = 20 if SMOKE else 0        # 0 = all 500 (AIME sets have 30)
N_SAMPLES = 4 if SMOKE else 16         # raise to 32 or 64 for a sharper high-k tail
MAX_TOKENS = 1024 if SMOKE else 3000
TEMPERATURE, TOP_P, SEED = 0.6, 0.95, 0   # CIR Fig. 6 settings
MAX_CALLS = 4
ALLOW_LONG = False         # True + MAX_MODEL_LEN=16384 mimics CIR's 16k budget on 4k-native models
MAX_MODEL_LEN = PRESETS[PRESET]["max_model_len"]
SHARED_TEMPLATE = "qwen_math_nocode"   # prompt_off prompt, identical for every model
PLAIN_COT = False          # True adds the non-intercepting plain_cot arm (CIR's condition)
DENIAL_TEXT = None         # None = "No interpreter is available; the code was not executed. ..."
GPU_MEM = 0.90
CHUNK = 100                # problems per checkpoint; a disconnect loses at most this much work
OVERWRITE = False          # True regenerates arms that are already on Drive

CONDITIONS = ["prompt_off", "env_denied", "tool_on"] + (["plain_cot"] if PLAIN_COT else [])
RUN_NAME = f"{PRESET}{'-smoke' if SMOKE else ''}"
OUT = f"{RESULTS}/{RUN_NAME}"
print(RUN_NAME, "->", OUT)
"""

RUN_MD = r"""
## 5. Run

One cell per checkpoint, below. Each cell first looks at what Drive already holds
for that model:

| On Drive | What the cell does |
|---|---|
| `.meta.json` for the arm | finished: loaded as is, nothing generated |
| `.jsonl.partial` only | interrupted: resumes from the last checkpointed chunk of `CHUNK` problems |
| nothing | generated from scratch |

So after a disconnect, reconnect, re-run cells 1 to 4 (install, Drive, code,
config), then re-run the model cell that was going. Finished arms and finished
chunks are never redone unless `OVERWRITE = True`. The three cells can run in any
order, one at a time (one GPU). Each one ends with a quick pass@1 table read back
from Drive. The first run of each model spends a few minutes downloading weights.
"""

RUN_SETUP = r"""
import json, subprocess, sys, time, importlib
from pathlib import Path
sys.path.insert(0, "src")


def arm_files(tag):
    spec = PRESETS[PRESET]["models"][tag]
    out = Path(OUT) / DATASET
    rows = []
    for cond in CONDITIONS:
        tmpl = SHARED_TEMPLATE if cond == "prompt_off" else ("qwen_math_cot" if cond == "plain_cot" else spec["native"])
        stem = out / f"{tag}__{cond}__{tmpl}"
        rows.append((cond, stem.with_suffix(".jsonl"), stem.with_suffix(".meta.json"), Path(str(stem) + ".jsonl.partial")))
    return rows


def partial_problems(partial):
    pids = set()
    for line in open(partial):
        try:
            pids.add(json.loads(line)["problem_id"])
        except (json.JSONDecodeError, KeyError):
            pass  # torn last line
    return len(pids)


def status(tag):
    # What Drive holds for this model. Returns how many arms still need generation.
    missing = 0
    for cond, jsonl, meta, partial in arm_files(tag):
        if meta.exists():
            m = json.load(open(meta))
            note = f"done     pass@1={m['pass1']:.3f}  n={m['n']}  problems={m['n_problems']}  {m['gen_seconds'] / 60:.0f} min"
            if m["n"] != N_SAMPLES or (N_PROBLEMS and m["n_problems"] != N_PROBLEMS):
                note += "   <- written with a different N_SAMPLES/N_PROBLEMS; set OVERWRITE=True or change RUN_NAME"
        elif partial.exists():
            note = f"partial  {partial_problems(partial)} problems checkpointed, will resume"
            missing += 1
        else:
            note = "missing"
            missing += 1
        print(f"  {tag:8s} {cond:11s} {note}")
    return missing


def show(tag):
    import analyze as an
    importlib.reload(an)
    try:
        df = an.load_results(f"{OUT}/{DATASET}")
    except FileNotFoundError:
        print("no results yet"); return
    d = df[df.tag == tag]
    if d.empty:
        print(f"no finished arms for {tag} yet"); return
    tbl = d.groupby("condition").agg(samples=("correct", "size"), problems=("problem_id", "nunique"),
                                      pass1=("correct", "mean"), trunc=("truncated", "mean"),
                                      calls=("n_calls", "mean"), denied=("n_denied", "mean")).round(3)
    tbl = tbl.reindex([c for c in an.COND_ORDER if c in tbl.index])
    try:
        display(tbl)
    except NameError:
        print(tbl.to_string())


def run_model(tag):
    spec = PRESETS[PRESET]["models"][tag]
    print(f"[{tag}] {spec['model']}  ->  {OUT}/{DATASET}")
    if status(tag) == 0 and not OVERWRITE:
        print(f"[{tag}] every arm is already on Drive, loading instead of regenerating")
    else:
        cmd = [sys.executable, "src/gen.py",
               "--model", spec["model"], "--tag", tag,
               "--conditions", ",".join(CONDITIONS),
               "--shared-template", SHARED_TEMPLATE, "--native-template", spec["native"],
               "--dataset", DATASET, "--n-problems", str(N_PROBLEMS), "--n", str(N_SAMPLES),
               "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
               "--temperature", str(TEMPERATURE), "--top-p", str(TOP_P), "--seed", str(SEED),
               "--max-calls", str(MAX_CALLS), "--gpu-mem", str(GPU_MEM), "--chunk", str(CHUNK), "--out", OUT]
        if ALLOW_LONG:
            cmd.append("--allow-long")
        if OVERWRITE:
            cmd.append("--overwrite")
        if DENIAL_TEXT:
            cmd += ["--denial-text", DENIAL_TEXT]
        print("\n$", " ".join(cmd), flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        print(f"\n[{tag}] exit {r.returncode} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if r.returncode != 0:
            print("fix the error above and re-run this cell; finished arms and chunks are kept")
        status(tag)
    show(tag)
"""

RUN_BASE = r"""
run_model("base")
"""

RUN_TEXT_RL = r"""
run_model("text_rl")
"""

RUN_TOOL_RL = r"""
run_model("tool_rl")
"""

ANALYZE_MD = r"""
## 6. Analyze

`summary` is the headline table, one row per arm. `trunc_pct`, `no_boxed_pct`,
`code_pct` and `fake_out_pct` are the failure modes that would make a pass@k gap
uninterpretable, so read them before the curves. Under `prompt_off`, `calls_mean`
is how often the model wrote code after being told not to. `arith_err_pct` is the direct skill metric
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
fig = an.plot_passk(df, title=f"{RUN_NAME} / {DATASET}: pass@k by condition")
fig.savefig(f"{OUT}/{DATASET}/passk.png", bbox_inches="tight")
"""

VERDICT = r"""
for cond in an.COND_ORDER:
    if cond in set(df.condition):
        an.verdict(df, cond)
        print()
"""

NEXT_MD = r"""
## 7. Reading the result

- **Read within a column.** Each condition puts all three models under the same
  prompt and the same harness, so `verdict` compares them per condition.
  `env_denied` is the headline: every model asked for code, nobody gets to run it.
- **prompt_off tests compliance too.** `calls_mean` on those rows is how often a
  model wrote code after being told not to. A tool-RL model that cannot stop is
  itself a finding.
- **Compare to text_rl, not just base.** RL alone sharpens pass@1 and can lower
  high-k pass@k (the sharpening confound). The question is whether tool-RL sits
  below text-RL.
- **Check failure modes first.** If an arm has 30% truncation or never boxes an
  answer, its curve measures format collapse.
- **Arithmetic error rate** is the skill the interpreter replaces. Tool-RL above
  text-RL here, at the same accuracy, is atrophy of exactly that skill.

Then: flip `SMOKE=False`, raise `N_SAMPLES` to 32 or 64 for the high-k tail, add
`aime24`/`aime25` with `N_SAMPLES=64`, and move to `math-7b` (the CIR checkpoint
reproduces Fig. 6 with the control added). Per-subject breakdowns are one
`df.groupby(["tag","condition","subject"]).correct.mean()` away.

**Caveats.** `tool_on` executes model-written Python on this VM with a 10 s
timeout; fine for Colab, not for a shared machine. The CIR prompt's turn placement
is assumed. The ZeroTIR tool prompt is unverified (its repo has no card).
Qwen2.5-Math's native context is 4096, so long solutions truncate; the rate is in
the table.
"""

DOWNLOAD = r"""
# Pull the trajectories to your laptop: every JSONL row carries the full generation
# text plus call / denial / truncation counts. Uncomment, run, then unzip locally and
# point analyze.load_results() at the <run>/<dataset> folder.
#
# from google.colab import files
# import shutil
# shutil.make_archive(f"/content/{RUN_NAME}", "zip", RESULTS, RUN_NAME)
# files.download(f"/content/{RUN_NAME}.zip")
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
        md(RUN_MD), code(RUN_SETUP), code(RUN_BASE), code(RUN_TEXT_RL), code(RUN_TOOL_RL),
        md(ANALYZE_MD), code(ANALYZE), code(PLOT), code(PLOT2), code(VERDICT),
        md(NEXT_MD), code(DOWNLOAD),
    ]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
