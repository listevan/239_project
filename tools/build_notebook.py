#!/usr/bin/env python
"""Build notebooks/tool_atrophy_pilot.ipynb from the cell templates below.

The notebook clones this repo onto the Colab VM, so src/ is never embedded and code changes only need a
push. Re-run after editing a cell template:   uv run tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "tool_atrophy_pilot.ipynb"


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


INTRO = r"""
# Tool atrophy pilot

Does RL with a code interpreter erode the reasoning the interpreter replaces? Three
checkpoints from Qwen2.5-Math-1.5B, four ways of handling code:

| condition | prompt | what happens to code |
|---|---|---|
| `tool_on` | the model's tool prompt | it runs (`python -I`, 10 s timeout, 4 calls) |
| `env_denied` | the model's tool prompt | not run; the output slot says "No interpreter is available" |
| `no_code` | the model's tool prompt | cannot be written: logit ban on backticks and program-only keywords |
| `plain` | plain step-by-step prompt | cannot be written (same ban): the best case for tool-free reasoning |

| model | arms | |
|---|---|---|
| `tool_rl` | all four | GAIR/ToRL-1.5B, RL with the interpreter |
| `base` | all four | Qwen2.5-Math-1.5B, untrained |
| `text_rl` | `plain` | Oat-Zero, RL without a tool (different recipe): attribution only |

**Read in order:** `tool_on` first (positive control: tool_rl must beat base with the
tool), then tool_rl vs base with the tool off (below base = atrophy), then text_rl
`plain` for whether any gap belongs to the tool.

**Pilot size:** AMC 2023, 30 problems x 4 samples (pass@1 to pass@4), 9 arms, about
20 minutes. Problems where computation helps but insight is still needed, and base
already solves a fair share.

**Run:** cells 1 to 6, then the three model cells. Then on the laptop:
`uv run src/report.py --run <run id>`.
"""

GPU_CHECK = r"""
# 1. GPU
import shutil, subprocess
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    print("NO GPU in this runtime. Colab: Runtime > Change runtime type > GPU. VS Code: reconnect choosing a "
          "GPU runtime. Then run again from this cell.")
"""

INSTALL = r"""
# 2. Install (vllm pins its own torch; generation runs in a subprocess, so no restart is needed)
import subprocess, sys
PKGS = "vllm datasets pandas"
if subprocess.run(f"pip install -q uv && uv pip install --system -q {PKGS}", shell=True).returncode != 0:
    subprocess.run(f"pip install -q {PKGS}", shell=True, check=True)
print(subprocess.run([sys.executable, "-c", "import vllm, datasets; print('vllm', vllm.__version__)"],
                     capture_output=True, text=True).stdout or "import check failed, see above")
"""

DRIVE = r"""
# 3. Drive: reuse an existing 239_project folder anywhere up to 5 levels under MyDrive, else create one
import glob, os
try:
    from google.colab import drive
    drive.mount("/content/drive")
    assert os.path.isdir("/content/drive/MyDrive"), "Drive did not mount; re-run this cell and accept the popup"
    hits = [p for d in range(6) for p in glob.glob("/content/drive/MyDrive/" + "*/" * d + "239_project") if os.path.isdir(p)]
    hits.sort(key=lambda p: (not os.path.isdir(f"{p}/runs"), p.count("/")))
    ROOT = hits[0] if hits else "/content/drive/MyDrive/239_project"
except ImportError:
    ROOT = os.path.abspath("239_project_local")   # outside Colab: a local folder
RUNS = f"{ROOT}/runs"
os.makedirs(RUNS, exist_ok=True)
print("runs ->", RUNS)
"""

CLONE = r"""
# 4. Code: clone the repo onto the VM, or pull it to the latest commit. The VM runs what is on GitHub.
import os, subprocess, sys
REPO, BRANCH = "https://github.com/listevan/239_project.git", "main"
WORK = "/content/239_project" if os.path.isdir("/content") else os.path.abspath("239_project_work")
if os.path.isdir(f"{WORK}/.git"):
    subprocess.run(["git", "-C", WORK, "fetch", "-q", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", f"origin/{BRANCH}"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, REPO, WORK], check=True)
os.chdir(WORK)
if f"{WORK}/src" not in sys.path:
    sys.path.insert(0, f"{WORK}/src")
print(subprocess.run(["git", "log", "-1", "--format=%h  %s  (%cr)"], capture_output=True, text=True).stdout.strip())
"""

CONFIG = r"""
# 5. Configure
import datetime, json, os
PRESET = "torl-1.5b"
MODELS = {
    "tool_rl": "GAIR/ToRL-1.5B",                  # RL with a code interpreter; beats base with the tool
    "base":    "Qwen/Qwen2.5-Math-1.5B",          # the untrained start
    "text_rl": "sail/Qwen2.5-Math-1.5B-Oat-Zero", # RL without a tool (different recipe): attribution only
}
TEMPLATES = {   # each model's tool prompt (tool_on, env_denied, no_code); "plain" always uses qwen_math_cot
    "tool_rl": "torl",            # ToRL's own raw prompt (paper Fig. 3)
    "base":    "qwen_math_tir",   # the Qwen2.5-Math tool-integrated prompt
    "text_rl": "qwen_math_tir",
}
CONDITIONS_BY_MODEL = {
    "tool_rl": ["tool_on", "env_denied", "no_code", "plain"],
    "base":    ["tool_on", "env_denied", "no_code", "plain"],
    "text_rl": ["plain"],
}
SMOKE = False                  # True: 6 problems x 2 samples x 1024 tokens, a wiring check in its own folder
# Pilot: one set, ~30 problems, pass@1..4. AMC 2023: computation helps, insight is still needed.
DATASETS = {"amc23": (6, 2)} if SMOKE else {"amc23": (30, 4)}   # (problems, samples)
MAX_TOKENS = 1024 if SMOKE else 3000
MAX_MODEL_LEN = 4096
TEMPERATURE, TOP_P, SEED = 0.6, 0.95, 0
MAX_CALLS = 4
GPU_MEM = 0.90
CHUNK = 100                    # problems per checkpoint
OVERWRITE = False              # False: only missing arms. True: redo all. ["no_code"]: redo those conditions only.
RUN_ID = None                  # None: today's date + preset. Set an existing id to resume or extend a run.

DATASET_SPEC = ",".join(f"{k}:{p}:{n}" for k, (p, n) in DATASETS.items())
RUN_ID = RUN_ID or f"{datetime.date.today():%Y-%m-%d}_{PRESET}"
if SMOKE and not RUN_ID.endswith("-smoke"):
    RUN_ID += "-smoke"           # a smoke run never writes into the real run folder
OUT = f"{RUNS}/{RUN_ID}"
os.makedirs(RUNS, exist_ok=True)
existing = sorted(d for d in os.listdir(RUNS) if os.path.isdir(f"{RUNS}/{d}"))
print("runs on Drive:", ", ".join(existing) or "(none)")
print("this run     :", RUN_ID, "(resuming)" if RUN_ID in existing else "(new)")
print("datasets     :", ", ".join(f"{k} {p or 'all'} problems x {n} samples" for k, (p, n) in DATASETS.items()))
print("arms         :", "; ".join(f"{t}: {', '.join(c)}" for t, c in CONDITIONS_BY_MODEL.items()))
os.makedirs(OUT, exist_ok=True)
CONFIG = dict(run_id=RUN_ID, preset=PRESET, models=MODELS, templates=TEMPLATES, conditions_by_model=CONDITIONS_BY_MODEL,
              datasets=DATASETS, max_tokens=MAX_TOKENS, max_model_len=MAX_MODEL_LEN, temperature=TEMPERATURE,
              top_p=TOP_P, seed=SEED, max_calls=MAX_CALLS, chunk=CHUNK,
              created=datetime.datetime.now().isoformat(timespec="seconds"))
cfg_path = f"{OUT}/config.json"
KEYS = ("models", "templates", "conditions_by_model", "datasets", "max_tokens", "max_model_len", "temperature",
        "top_p", "seed", "max_calls")
old = json.load(open(cfg_path)) if os.path.exists(cfg_path) else None
diff = [k for k in KEYS if old and json.dumps(old.get(k), sort_keys=True) != json.dumps(CONFIG[k], sort_keys=True)]
if old is None or diff:
    # config.json always describes the latest settings; earlier ones are kept in its history
    CONFIG["history"] = (old.pop("history", []) + [old]) if old else []
    json.dump(CONFIG, open(cfg_path, "w"), indent=1)
print("config.json:", "new" if old is None else (f"updated, changed {diff} (earlier settings kept in its history). "
      "Finished arms keep the settings they were made with; arms with fewer samples than asked are redone"
      if diff else "same settings"))
"""

RUN_SETUP = r"""
# 6. Run helpers
import json, os, signal, shutil, subprocess, sys, time
from pathlib import Path


import gen


def arm_files(tag, dataset):
    # (condition, jsonl, meta, partial) for every arm this model runs on this dataset
    out = Path(OUT) / dataset
    rows = []
    for cond in CONDITIONS_BY_MODEL[tag]:
        stem = out / f"{tag}__{cond}__{gen.template_for(cond, TEMPLATES[tag])}"
        rows.append((cond, stem.with_suffix(".jsonl"), stem.with_suffix(".meta.json"), Path(str(stem) + ".jsonl.partial")))
    return rows


def status(tag):
    # What Drive holds for this model, using gen.py's own rule for "done". Returns how many arms gen.py
    # will (re)generate.
    import gen
    missing = 0
    for ds, (n_prob, n_samp) in DATASETS.items():
        for cond, jsonl, meta, partial in arm_files(tag, ds):
            if meta.exists():
                m = json.load(open(meta))
                note = f"done     n={m['n']}  problems={m['n_problems']}  {m['gen_seconds'] / 60:.0f} min"
                if not gen.arm_done(meta, n_samp, n_prob):
                    note += "   <- fewer problems/samples than DATASETS asks: will be REDONE (old kept as .replaced)"
                    missing += 1
                elif m["n"] > n_samp:
                    note += "   (more samples than DATASETS asks: fine, used as is)"
            elif partial.exists():
                pids = set()
                for line in open(partial):
                    try:
                        pids.add(json.loads(line)["problem_id"])
                    except (json.JSONDecodeError, KeyError):
                        pass   # torn last line
                note = f"partial  {len(pids)} problems checkpointed, will resume"
                missing += 1
            else:
                note = "missing"
                missing += 1
            print(f"  {tag:8s} {ds:14s} {cond:11s} {note}")
    return missing


def show(tag):
    # A quick table per dataset straight from Drive: rows, truncation, tool use (pass@1 once graded locally)
    import pandas as pd
    for ds in DATASETS:
        files = [jsonl for _, jsonl, _, _ in arm_files(tag, ds) if jsonl.exists()]
        if not files:
            continue
        d = pd.DataFrame([json.loads(l) for f in files for l in open(f) if l.strip()])
        aggs = dict(samples=("problem_id", "size"), problems=("problem_id", "nunique"), trunc=("truncated", "mean"),
                    called=("n_calls", lambda s: (s > 0).mean()), denied=("n_denied", "mean"))
        if d["correct"].notna().all():
            aggs["pass1"] = ("correct", "mean")
        tbl = d.groupby("condition").agg(**aggs).round(3)
        tbl.index.name = f"{tag} / {ds}"
        try:
            display(tbl)
        except NameError:
            print(tbl.to_string())


def stream(cmd):
    # Relay the subprocess's output through the kernel (a line per progress refresh). The child runs in its
    # own process group, so the stop button kills gen.py and vLLM's engine process and frees the GPU.
    env = dict(os.environ, TQDM_MININTERVAL="20", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            print(chunk.decode("utf-8", "replace").replace("\r", "\n"), end="", flush=True)
        return proc.wait()
    except BaseException:
        for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
            try:
                os.killpg(proc.pid, sig)
                proc.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                continue
            except ProcessLookupError:
                break
        print(f"\n[interrupted] killed pid {proc.pid} and its children; finished arms and chunks are kept", flush=True)
        raise


def run_model(tag):
    print(f"[{tag}] {MODELS[tag]}  ->  {OUT}")
    print(f"[{tag}] datasets {DATASET_SPEC}; conditions {', '.join(CONDITIONS_BY_MODEL[tag])}")
    if status(tag) == 0 and not OVERWRITE:
        print(f"[{tag}] every arm is already on Drive, nothing to generate")
    elif not shutil.which("nvidia-smi"):
        print(f"[{tag}] arms are missing but this runtime has NO GPU; switch to a GPU runtime and run from cell 1")
    else:
        cmd = [sys.executable, "src/gen.py", "--model", MODELS[tag], "--tag", tag,
               "--conditions", ",".join(CONDITIONS_BY_MODEL[tag]), "--template", TEMPLATES[tag], "--dataset", DATASET_SPEC,
               "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
               "--temperature", str(TEMPERATURE), "--top-p", str(TOP_P), "--seed", str(SEED),
               "--max-calls", str(MAX_CALLS), "--gpu-mem", str(GPU_MEM), "--chunk", str(CHUNK), "--out", OUT]
        if OVERWRITE is True:
            cmd.append("--overwrite")
        elif OVERWRITE:
            cmd += ["--overwrite-conditions", ",".join(OVERWRITE)]
        print("\n$", " ".join(cmd), flush=True)
        t0 = time.time()
        rc = stream(cmd)
        print(f"\n[{tag}] exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            print("fix the error above and re-run this cell; finished arms and chunks are kept")
        status(tag)
    show(tag)
"""

AFTER = r"""
## After the run

On the laptop (Google Drive for desktop syncs the run folder):

```
cd ~/projects/239_project
uv run src/report.py --list
uv run src/report.py --run <run id>
```

It grades the rows in place, prints the summary, pass@k, tool-call tables and the
verdicts per dataset, and saves dated figures and CSVs under `<run>/figures/` and
`<run>/tables/`. Read `tool_on` first (positive control: tool_rl above base), then
`no_code` and `env_denied` (main test: tool_rl below base is atrophy; check text_rl
`no_code` before attributing it to the tool).
"""


def build():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nb.cells = [md(INTRO), code(GPU_CHECK), code(INSTALL), code(DRIVE), code(CLONE), code(CONFIG), code(RUN_SETUP),
                code('run_model("tool_rl")'), code('run_model("base")'), code('run_model("text_rl")'), md(AFTER)]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
