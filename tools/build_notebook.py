#!/usr/bin/env python
"""Build notebooks/tool_atrophy_pilot.ipynb from the cell templates below.

The notebook clones this repo onto the Colab VM (cell 3), so src/ is never embedded
and code changes only need a push. Re-run this after editing a cell template:
    uv run tools/build_notebook.py
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

**What runs.** The repo is cloned onto the VM. `src/gen.py` loads one checkpoint, samples it under each condition
on each dataset, and writes one JSONL per (model, dataset, condition) to Google
Drive, checkpointing every `CHUNK` problems, so a disconnect costs at most one
chunk. Colab does GPU work only: grading (math-verify) and the arithmetic check run
on your laptop with `src/report.py`, which also makes the tables and figures
(`GRADE_ON_COLAB = True` grades here instead). Every model gets the same grid, so
each condition is read as its own three-way comparison.

**Where it goes.** One folder per run on Drive, named by date and preset:

```
239_project/runs/2026-10-15_math-1.5b/
    config.json          the settings of this run
    aime24/  aime25/         one JSONL + meta per arm (more datasets if you add them)
    figures/  tables/   written by report.py, every file dated
```

| Condition | Prompt | If the model writes a code block |
|---|---|---|
| `prompt_off` | shared no-code prompt ("do not write code") | intercepted, denial message injected, nothing runs |
| `env_denied` | the model's tool prompt (asks for code) | intercepted, denial message injected, nothing runs |
| `tool_on` | the model's tool prompt | executed with `python -I`, 4-call budget, then the budget message |
| `prose_only` | shared no-code prompt, plus a logit ban on every backtick token | cannot happen: no fence can open, so nothing is denied. The ability test. |
| `plain_cot` (optional) | shared plain CoT prompt | nothing intercepted; the model fabricates its own output block. CIR's condition. |

**Hardware.** Pick a GPU runtime. A100 40 GB runs everything. L4 24 GB runs the
1.5B preset comfortably and the 7B preset slowly. T4 only fits 1.5B (fp16).
Rough wall-clock on an A100 with vLLM for the default plan (AIME24 + AIME25 at
64 samples, about 3.8k samples per arm, long solutions):

| Preset | per model (3 arms) | whole grid |
|---|---|---|
| math-1.5b | 20 to 30 min | about 1 h |
| math-7b | 1 to 1.5 h | 3 to 4 h |

`SMOKE = True` in the config cell runs a 10-minute wiring check on 6 AIME problems
instead of the real run; use it after changing code, not before every run.
"""

GPU_CHECK = r"""
import shutil, subprocess
NO_GPU_MSG = ("NO GPU in this runtime. Colab: Runtime > Change runtime type > GPU (T4 / L4 / A100). "
              "VS Code Colab extension: disconnect and reconnect choosing a GPU runtime. Then run again from this cell.")
if shutil.which("nvidia-smi"):
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
                         capture_output=True, text=True).stdout)
else:
    print(NO_GPU_MSG)
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

Runs are written under `239_project/runs/<date>_<preset>/` on Drive so they
survive the session. An existing `239_project` folder anywhere up to five levels
under MyDrive is reused, so you can file it wherever you like in Drive; otherwise
`MyDrive/239_project` is created. Accept the Drive popup; if the path printed below
starts with `/content/239_project_local` the results are on the VM disk and die
with it. Outside Colab this falls back to a local folder.
"""

DRIVE = r"""
import glob, os
PROJECT_DIR = None   # None: reuse an existing 239_project folder anywhere up to 5 levels under MyDrive
                     # (so moving it around in Drive is fine), else create MyDrive/239_project.
                     # Or name it explicitly, e.g. "UCLA (Jun 2026 onward)/239_project".
try:
    from google.colab import drive  # noqa: F401
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
if ON_COLAB:
    # If the auth popup is dismissed this raises: results must land on Drive, not on the VM disk.
    drive.mount("/content/drive")
    assert os.path.isdir("/content/drive/MyDrive"), "Drive did not mount; re-run this cell and accept the popup"
    if PROJECT_DIR:
        ROOT = f"/content/drive/MyDrive/{PROJECT_DIR}"
    else:
        hits = [p for d in range(6) for p in glob.glob("/content/drive/MyDrive/" + "*/" * d + "239_project")
                if os.path.isdir(p)]
        hits.sort(key=lambda p: (not os.path.isdir(f"{p}/runs"), p.count("/")))   # one that already has runs/ first
        ROOT = hits[0] if hits else "/content/drive/MyDrive/239_project"
        if len(hits) > 1:
            print("several 239_project folders on Drive, using the first:", *hits, sep="\n  ")
else:
    ROOT = os.path.abspath("239_project_local")
RUNS = f"{ROOT}/runs"
os.makedirs(RUNS, exist_ok=True)
print("runs ->", RUNS)
"""

CODE_MD = r"""
## 3. Code

The repo is cloned onto this VM at `/content/239_project` (or pulled to the latest
commit if it is already there), and `src/` goes on the path. The VM runs what is
on GitHub, so push local edits before running. The repo is public, so no token is
needed. For a private fork, a fine-grained token with read access to the repo's
contents is picked up from the `GITHUB_TOKEN` environment variable, a Colab
secret of that name, or `github_token.txt` in the `239_project` folder on Drive;
nothing here prints it.
"""

CODE = r"""
import os, subprocess, sys
REPO = "listevan/239_project"
BRANCH = "main"
WORK = "/content/239_project" if os.path.isdir("/content") else os.path.abspath("239_project_work")


def github_token():
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok.strip(), "GITHUB_TOKEN env var"
    try:
        from google.colab import userdata
        tok = userdata.get("GITHUB_TOKEN")
        if tok:
            return tok.strip(), "Colab secret"
    except Exception:
        pass
    p = f"{ROOT}/github_token.txt"
    if os.path.exists(p):
        return open(p).read().strip(), "Drive file github_token.txt"
    return None, "none (public clone)"


tok, tok_src = github_token()
url = f"https://x-access-token:{tok}@github.com/{REPO}.git" if tok else f"https://github.com/{REPO}.git"
if os.path.isdir(f"{WORK}/.git"):
    subprocess.run(["git", "-C", WORK, "remote", "set-url", "origin", url], check=True)
    r = subprocess.run(["git", "-C", WORK, "fetch", "-q", "origin", BRANCH], capture_output=True, text=True)
    if r.returncode == 0:
        subprocess.run(["git", "-C", WORK, "reset", "-q", "--hard", f"origin/{BRANCH}"], check=True)
    action = "pulled" if r.returncode == 0 else "fetch failed, kept the copy on the VM"
else:
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", "-b", BRANCH, url, WORK], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"clone failed (token source: {tok_src}): {r.stderr.strip().replace(tok or '@@', '***')}\n"
                         "The repo is private: put a fine-grained GitHub token (contents: read) in "
                         f"{ROOT}/github_token.txt, or as a Colab secret GITHUB_TOKEN, then re-run this cell.")
    action = "cloned"
os.chdir(WORK)
if f"{WORK}/src" not in sys.path:
    sys.path.insert(0, f"{WORK}/src")
sha = subprocess.run(["git", "-C", WORK, "log", "-1", "--format=%h  %s  (%cr)"], capture_output=True, text=True).stdout.strip()
print(f"{action} {REPO}@{BRANCH} via {tok_src}\n  {sha}\n  work -> {WORK}  (the VM runs this commit; push local edits first)")
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

**Datasets.** `DATASETS` maps a name to (problems, samples per problem); 0 problems
means the whole set. The pilot is AIME24 plus AIME25, 60 problems at 64 samples:
the hard end, where the interpreter should matter. The interpreter is not
load-bearing on MATH-500 at 1.5B (the smoke run and the ToRL paper agree, a few
points), so MATH-500 is off. GSM-Hard (GSM8K with huge numbers) and the level-5
MATH-500 subset are one line each to add back. The gate before any verdict:
tool_on minus env_denied for the tool-RL model must be large on the set.

**Run id.** `RUN_ID` defaults to today's date plus the preset. Re-running the config
cell on another day makes a new run folder; to resume or extend an earlier one (a
disconnect, or adding a dataset), set `RUN_ID` to its name from the printed list.
`config.json` records the settings and warns if they changed.
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
SMOKE = False              # False: the real run. True: 6 AIME problems x 4 samples x 1024 tokens, ~10 min wiring check.

# name -> (problems, samples per problem); 0 problems = the whole set. The pilot is AIME.
#   aime24 / aime25   30 problems each, the hard end where the interpreter should matter
# Available, off by default:  "gsm_hard": (300, 16)  GSM8K with huge numbers, 1016 integer-answer problems
#                             "math500_l5": (0, 16)  the 134 level-5 MATH-500 problems
#                             "math500": (0, 16)     all 500, not tool-load-bearing at 1.5B
if SMOKE:
    DATASETS = {"aime24": (6, 4)}
    MAX_TOKENS = 1024
else:
    DATASETS = {"aime24": (0, 64), "aime25": (0, 64)}
    MAX_TOKENS = 3000
RUN_ID = None              # None: today's date + preset, e.g. 2026-10-15_math-1.5b. To resume or extend an
                           # earlier run, set its id from the list printed below.

TEMPERATURE, TOP_P, SEED = 0.6, 0.95, 0   # CIR Fig. 6 settings
MAX_CALLS = 4
ALLOW_LONG = False         # True + MAX_MODEL_LEN=16384 mimics CIR's 16k budget on 4k-native models
MAX_MODEL_LEN = PRESETS[PRESET]["max_model_len"]
SHARED_TEMPLATE = "qwen_math_nocode"   # prompt_off prompt, identical for every model
PLAIN_COT = False          # True adds the non-intercepting plain_cot arm (CIR's condition)
DENIAL_TEXT = None         # None = "No interpreter is available; the code was not executed. ..."
GPU_MEM = 0.90
CHUNK = 100                # problems per checkpoint; a disconnect loses at most this much work
OVERWRITE = False          # False: generate only missing arms. True: redo everything. A list such as
                           # ["prose_only"] redoes just those conditions and keeps the rest.
GRADE_ON_COLAB = False     # False: Colab only generates; grade + analyze on your laptop (section 6)

import datetime, json, os
CONDITIONS = ["prompt_off", "env_denied", "prose_only", "tool_on"] + (["plain_cot"] if PLAIN_COT else [])
DATASET_SPEC = ",".join(f"{k}:{p}:{n}" for k, (p, n) in DATASETS.items())
RUN_ID = RUN_ID or f"{datetime.date.today():%Y-%m-%d}_{PRESET}{'-smoke' if SMOKE else ''}"
OUT = f"{RUNS}/{RUN_ID}"
os.makedirs(RUNS, exist_ok=True)
existing = sorted(d for d in os.listdir(RUNS) if os.path.isdir(f"{RUNS}/{d}"))
print("runs on Drive:", ", ".join(existing) or "(none)")
print("this run     :", RUN_ID, "->", OUT, "(resuming)" if RUN_ID in existing else "(new)")
os.makedirs(OUT, exist_ok=True)
CONFIG = dict(run_id=RUN_ID, preset=PRESET, smoke=SMOKE, datasets=DATASETS, conditions=CONDITIONS,
              models=PRESETS[PRESET]["models"], max_tokens=MAX_TOKENS, max_model_len=MAX_MODEL_LEN,
              temperature=TEMPERATURE, top_p=TOP_P, seed=SEED, max_calls=MAX_CALLS, shared_template=SHARED_TEMPLATE,
              denial_text=DENIAL_TEXT, chunk=CHUNK, created=datetime.datetime.now().isoformat(timespec="seconds"))
cfg_path = f"{OUT}/config.json"
if os.path.exists(cfg_path):
    old = json.load(open(cfg_path))
    diff = [k for k in ("preset", "datasets", "conditions", "max_tokens", "max_model_len", "temperature", "top_p",
                        "seed", "max_calls", "shared_template", "denial_text")
            if json.dumps(old.get(k), sort_keys=True) != json.dumps(CONFIG[k], sort_keys=True)]
    print(f"config.json from {old.get('created', '?')}" + (f"; DIFFERS in {diff}: finished arms keep their old settings. "
          "Added conditions or datasets are fine (the missing arms get generated); changed sampling settings want a new RUN_ID"
          if diff else "; same settings"))
else:
    json.dump(CONFIG, open(cfg_path, "w"), indent=1)
    print("wrote", cfg_path)
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
config, with `RUN_ID` set to the run you were on if the date changed), then re-run
the model cell that was going. Finished arms and finished chunks are never redone unless `OVERWRITE` says so (`True` for everything, a list
such as `["prose_only"]` for those conditions only). The model is loaded once and runs every dataset in
`DATASETS`. The three cells can run in any order, one at a time (one GPU). Each one
ends with a quick table per dataset read back from Drive. The first run of each
model spends a few minutes downloading weights.

`samples` in those tables is rows, which is problems x samples. If it is a
multiple of what you expect, a file from an older run with a different prompt
template sits in the same folder and loads as the same arm. `status` flags this;
`move_stale()` moves such files into `old/`.
"""

RUN_SETUP = r"""
import json, os, shutil, subprocess, sys, time, importlib
from pathlib import Path
sys.path.insert(0, "src")
import gen   # the file-naming rules (which conditions use the shared prompt) come from gen.py itself


def arm_files(tag, dataset, conditions=None):
    spec = PRESETS[PRESET]["models"][tag]
    out = Path(OUT) / dataset
    rows = []
    for cond in (conditions or CONDITIONS):
        tmpl = (SHARED_TEMPLATE if cond in gen.SHARED_PROMPT_CONDITIONS
                else "qwen_math_cot" if cond == "plain_cot" else spec["native"])
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


DATA_SUFFIXES = (".jsonl", ".meta.json", ".jsonl.partial", ".jsonl.tmp")


def expected_stems(dataset):
    # Every condition gen.py knows, not only the ones selected right now, so an arm generated under a
    # condition that was later removed from CONDITIONS (or added by a newer notebook) is never "stale".
    return {jsonl.stem for tag in PRESETS[PRESET]["models"] for _, jsonl, _, _ in arm_files(tag, dataset, gen.CONDITIONS)}


def stale_files(dataset):
    # Data files in a dataset folder that the current config does not produce (an older template or
    # model), plus arms written by an older gen.py (no "chunk" key in meta). They inflate the
    # tables (two files load as one arm) and block regeneration (gen.py sees the arm as done).
    out = Path(OUT) / dataset
    exp = expected_stems(dataset)
    stale = []
    for f in sorted(out.iterdir()) if out.is_dir() else []:
        if not f.is_file() or not f.name.endswith(DATA_SUFFIXES):
            continue
        stem = f.name.split(".")[0]
        if stem not in exp:
            stale.append((f, "not an arm of the current config"))
        elif f.name.endswith(".meta.json") and "chunk" not in json.load(open(f)):
            stale.append((f, "written by an older notebook version"))
            stale.append((out / f"{stem}.jsonl", "written by an older notebook version"))
    return stale


def move_stale(dataset=None):
    # Quarantine stale files into <run>/<dataset>/old/ (nothing is deleted).
    for ds in ([dataset] if dataset else list(DATASETS)):
        out = Path(OUT) / ds
        stale = stale_files(ds)
        if not stale:
            print(f"{ds}: nothing stale"); continue
        (out / "old").mkdir(exist_ok=True)
        for f, why in stale:
            if f.exists():
                shutil.move(str(f), str(out / "old" / f.name))
                print(f"{ds}: moved {f.name} -> old/   ({why})")


def load_run(dataset, require_graded=False):
    # One dataset of this run, current config's files only; files from other runs in the folder are
    # reported and ignored. require_graded raises with instructions if rows still wait for grading.
    import analyze as an
    importlib.reload(an)
    df = an.load_results(f"{OUT}/{dataset}", allow_ungraded=not require_graded)
    exp = {f"{stem}.jsonl" for stem in expected_stems(dataset)}
    stale = {f.name for f, _ in stale_files(dataset)}
    stray = sorted((set(df.file) - exp) | (set(df.file) & stale))
    if stray:
        print(f"{dataset}: ignoring", len(stray), "file(s) from another run or notebook version:", ", ".join(stray))
        print("run move_stale() to quarantine them")
    return df[~df.file.isin(stray)].copy()


def status(tag):
    # What Drive holds for this model, per dataset. Returns how many arms still need generation.
    missing = 0
    for ds, (n_prob, n_samp) in DATASETS.items():
        for cond, jsonl, meta, partial in arm_files(tag, ds):
            if meta.exists():
                m = json.load(open(meta))
                acc = "ungraded" if m.get("pass1") is None else f"pass@1={m['pass1']:.3f}"
                note = f"done     {acc}  n={m['n']}  problems={m['n_problems']}  {m['gen_seconds'] / 60:.0f} min"
                if m["n"] != n_samp or (n_prob and m["n_problems"] != n_prob):
                    note += "   <- written with different problems/samples; set OVERWRITE=True or a new RUN_ID"
                if "chunk" not in m:
                    note += "   <- older notebook version; run move_stale(), then re-run this cell"
            elif partial.exists():
                note = f"partial  {partial_problems(partial)} problems checkpointed, will resume"
                missing += 1
            else:
                note = "missing"
                missing += 1
            print(f"  {tag:8s} {ds:9s} {cond:11s} {note}")
        n_stale = len({f for f, _ in stale_files(ds)})
        if n_stale:
            print(f"  ! {ds}: {n_stale} stale file(s) from another run in this folder: run move_stale() first")
    return missing


def show(tag):
    import analyze as an
    for ds in DATASETS:
        try:
            df = load_run(ds)
        except FileNotFoundError:
            print(f"{tag} / {ds}: no results yet"); continue
        d = df[df.tag == tag]
        if d.empty:
            print(f"{tag} / {ds}: no finished arms yet"); continue
        graded = d["correct"].notna().all()
        if not graded:
            print(f"{tag} / {ds}: rows are ungraded, pass@1 appears after report.py on your laptop")
        aggs = dict(samples=("correct", "size"), problems=("problem_id", "nunique"), trunc=("truncated", "mean"),
                    called=("n_calls", lambda s: (s > 0).mean()), calls_total=("n_calls", "sum"),
                    calls=("n_calls", "mean"), denied=("n_denied", "mean"))
        if graded:
            d = d.assign(correct=d["correct"].astype(float))
            aggs = dict(samples=aggs["samples"], problems=aggs["problems"], pass1=("correct", "mean"),
                        **{k: v for k, v in aggs.items() if k not in ("samples", "problems")})
        tbl = d.groupby("condition").agg(**aggs).round(3)
        tbl = tbl.reindex([c for c in an.COND_ORDER if c in tbl.index] + [c for c in tbl.index if c not in an.COND_ORDER])
        tbl.index.name = f"{tag} / {ds}"
        try:
            display(tbl)
        except NameError:
            print(tbl.to_string())


def stream(cmd):
    # Relay the subprocess's stdout+stderr through the kernel: a line per tqdm refresh instead of
    # a carriage-return bar, so progress shows even when Colab does not mirror child output.
    # The child runs in its own process group; the stop button kills the whole group (gen.py and
    # vLLM's engine-core process), so nothing is left holding the GPU.
    import signal
    env = dict(os.environ, TQDM_MININTERVAL="20", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            start_new_session=True)
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
        print(f"\n[interrupted] killed pid {proc.pid} and its children; finished arms and chunks are kept",
              flush=True)
        raise


def run_model(tag):
    spec = PRESETS[PRESET]["models"][tag]
    print(f"[{tag}] {spec['model']}  ->  {OUT}")
    if status(tag) == 0 and not OVERWRITE:
        print(f"[{tag}] every arm is already on Drive, loading instead of regenerating\n"
              "  (a condition missing from the list above? pull the repo so the notebook has it, re-run the config cell)")
    elif not shutil.which("nvidia-smi"):
        print(f"[{tag}] arms are missing but this runtime has NO GPU, nothing generated.\n"
              "  Colab: Runtime > Change runtime type > GPU. VS Code: reconnect choosing a GPU runtime.\n"
              "  Then run again from cell 1 (a runtime change resets the VM).")
    else:
        cmd = [sys.executable, "src/gen.py",
               "--model", spec["model"], "--tag", tag,
               "--conditions", ",".join(CONDITIONS),
               "--shared-template", SHARED_TEMPLATE, "--native-template", spec["native"],
               "--dataset", DATASET_SPEC,
               "--max-tokens", str(MAX_TOKENS), "--max-model-len", str(MAX_MODEL_LEN),
               "--temperature", str(TEMPERATURE), "--top-p", str(TOP_P), "--seed", str(SEED),
               "--max-calls", str(MAX_CALLS), "--gpu-mem", str(GPU_MEM), "--chunk", str(CHUNK), "--out", OUT]
        if ALLOW_LONG:
            cmd.append("--allow-long")
        if OVERWRITE is True:
            cmd.append("--overwrite")
        elif OVERWRITE:
            cmd += ["--overwrite-conditions", ",".join(OVERWRITE)]
        if not GRADE_ON_COLAB:
            cmd.append("--no-grade")
        if DENIAL_TEXT:
            cmd += ["--denial-text", DENIAL_TEXT]
        print("\n$", " ".join(cmd), flush=True)
        t0 = time.time()
        rc = stream(cmd)
        print(f"\n[{tag}] exit {rc} after {(time.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
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

With `GRADE_ON_COLAB = False` (the default) the rows on Drive are ungraded and the
cells below stop with a message. Grade and analyze on your laptop instead:

```
cd ~/projects/239_project
uv sync                                         # once
uv run src/report.py --list                     # runs visible through Google Drive for desktop
uv run src/report.py --run <run id>             # every dataset: grade, tables, figures
```

`report.py` finds `239_project/runs` under Google Drive for desktop (any depth
under My Drive), grades the ungraded rows in place first (`src/grade.py`, skips
rows already graded), prints the summary, pass@k, tool-call and verdict tables per
dataset (plus the MATH-500 level-5 subset), saves them as CSV under
`<run>/tables/` and the figures under `<run>/figures/`, every file dated. Pass a
run folder instead of `--run` for an unzipped download from the last cell. The
same functions are available in a local notebook via `import analyze as an`
(`an.load_results`, `an.summary`, `an.calls_table`, `an.plot`).

With `GRADE_ON_COLAB = True` everything below runs here.

`summary` is the headline table, one row per arm. `trunc_pct`, `no_boxed_pct`,
`code_pct` and `fake_out_pct` are the failure modes that would make a pass@k gap
uninterpretable, so read them before the curves. Under `prompt_off`, `called_pct`
is the share of samples that wrote code after being told not to; `calls_table`
breaks tool use down further (calls per caller, retries after a denial). `arith_err_pct` is the direct skill metric
(exact-rational check of every `a op b = c` in the prose, code blocks excluded);
`arith_coverage_pct` says how many samples contained anything to check.

Figures come from `an.plot(df, lines)`, where each line is a `"model condition"`
string. The two cells below draw the per-model view (what turning the tool off
costs each checkpoint) and the headline view (tool-RL with the tool off against the
control and the base). Edit the strings to draw anything else.
"""

ANALYZE = r"""
import sys, importlib
sys.path.insert(0, "src")
import analyze as an
importlib.reload(an)
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

if GRADE_ON_COLAB:
    SHOW = list(DATASETS)[0]        # the dataset the cells below display; change and re-run for another
    df = load_run(SHOW, require_graded=True)   # current config's files only; strays are listed and skipped
    print(SHOW, ":", len(df), "samples,", df.problem_id.nunique(), "problems,", df.arm.nunique(), "arms")
    STAMP, CAPTION = an.file_stamp(), f"{RUN_ID}: " + an.provenance(f"{OUT}/{SHOW}")   # figures dated + captioned
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    print(CAPTION)
    display(an.summary(df))
else:
    df = None
    print("Generation done. Rows are ungraded here (GRADE_ON_COLAB = False). On your laptop:\n"
          f"  cd ~/projects/239_project && uv run src/report.py --run {RUN_ID}")
"""

PLOT = r"""
if df is not None:
    display(an.passk_table(df).round(3))
"""

CALLS = r"""
# Tool-call behaviour. called_pct: % of samples that wrote a python block (under prompt_off: after being
# told not to). calls_per_caller: calls among those. retry_after_denial: further calls after the first
# denial (every call after a denial is denied too, so this is n_denied - 1). calls=k: samples by call count.
if df is not None:
    display(an.calls_table(df))
"""

PLOT2 = r"""
# an.plot draws pass@k curves. Each line is "model condition".
#   models:     base, text_rl, tool_rl
#   conditions: prompt_off, env_denied, tool_on
# A list makes one panel. A dict {panel title: [lines]} makes one panel per key.
# color="model" or "condition" picks what the color means; the other one sets the line style.
# save=... writes a PNG under <run>/figures/ (dated with STAMP); caption= puts the data's generation
# dates under the plot. Lines not in the results are skipped with a note.

# Figure 1: within each model, what does turning the tool off cost?
if df is not None:
    an.plot(df, {
        "base":    ["base prompt_off",    "base env_denied",    "base prose_only",    "base tool_on"],
        "text_rl": ["text_rl prompt_off", "text_rl env_denied", "text_rl prose_only", "text_rl tool_on"],
        "tool_rl": ["tool_rl prompt_off", "tool_rl env_denied", "tool_rl prose_only", "tool_rl tool_on"],
    }, color="condition", title=f"{RUN_ID} / {SHOW}: pass@k within each model",
       save=f"{OUT}/figures/passk_by_model_{SHOW}_{STAMP}.png", caption=CAPTION);
"""

PLOT3 = r"""
# Figure 2: the three models under one tool-off condition, plus the base with the tool on.
# Figure 3: the same under prompt_off. Figure 4: everyone with the tool (the dose check).
if df is not None:
    for key, arms, ttl in [
        ("denied",    ["tool_rl env_denied", "text_rl env_denied", "base env_denied", "base tool_on"], "tool denied"),
        ("promptoff", ["tool_rl prompt_off", "text_rl prompt_off", "base prompt_off", "base tool_on"], "told not to code"),
        ("prose",     ["tool_rl prose_only", "text_rl prose_only", "base prose_only", "base tool_on"], "no code possible"),
        ("toolon",    ["tool_rl tool_on", "text_rl tool_on", "base tool_on"], "with the tool"),
    ]:
        an.plot(df, arms, title=f"{RUN_ID} / {SHOW}: {ttl}, tool-RL vs text-RL vs base",
                save=f"{OUT}/figures/passk_main_{key}_{SHOW}_{STAMP}.png", caption=CAPTION);
"""

VERDICT = r"""
if df is not None:
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
- **prompt_off tests compliance too.** `called_pct` on those rows is how often a
  model wrote code after being told not to. A tool-RL model that cannot stop is
  itself a finding (ToRL: about 90% on AIME, Oat-Zero about 30%).
- **prose_only is the ability test.** No fence can open, so there is no denial to
  derail the model and no code to defer to; what is left is what it can compute by
  hand. `env_denied` minus `prose_only` is the behavioural cost of the habit.
- **Compare to text_rl, not just base.** RL alone sharpens pass@1 and can lower
  high-k pass@k (the sharpening confound). The question is whether tool-RL sits
  below text-RL.
- **Check failure modes first.** If an arm has 30% truncation or never boxes an
  answer, its curve measures format collapse.
- **Arithmetic error rate** is the skill the interpreter replaces. Tool-RL above
  text-RL here, at the same accuracy, is atrophy of exactly that skill.

- **Gate first.** On each dataset, tool_on minus env_denied for the tool-RL
  model is how much the interpreter was carrying. Where that gap is small (MATH-500
  at 1.5B), the set cannot test atrophy whatever the text-RL comparison says. Read
  the verdict on the sets where the gap is large (GSM-Hard, AIME, MATH level 5).
  The full MATH-500 is the compliance run, not the atrophy test.

Then: flip `SMOKE=False` for the default plan, then `math-7b` (the CIR checkpoint
makes several calls per problem and reproduces Fig. 6 with the control added).
Per-subject breakdowns are one `df.groupby(["tag","condition","subject"]).correct.mean()` away.

**Caveats.** `tool_on` executes model-written Python on this VM with a 10 s
timeout; fine for Colab, not for a shared machine. The CIR prompt's turn placement
is assumed. The ZeroTIR tool prompt is unverified (its repo has no card).
Qwen2.5-Math's native context is 4096, so long solutions truncate; the rate is in
the table.
"""

DOWNLOAD = r"""
# Without Google Drive for desktop: pull the run to your laptop as a zip. Every JSONL row
# carries the full generation text plus call / denial / truncation counts. Uncomment, run,
# unzip locally, then  uv run src/report.py <unzipped run folder>.
#
# from google.colab import files
# import shutil
# shutil.make_archive(f"/content/{RUN_ID}", "zip", RUNS, RUN_ID)
# files.download(f"/content/{RUN_ID}.zip")
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
        code(GPU_CHECK),
        md(INSTALL_MD), code(INSTALL),
        md(DRIVE_MD), code(DRIVE),
        md(CODE_MD), code(CODE),
        md(CONFIG_MD), code(CONFIG),
        md(RUN_MD), code(RUN_SETUP), code(RUN_BASE), code(RUN_TEXT_RL), code(RUN_TOOL_RL),
        md(ANALYZE_MD), code(ANALYZE), code(PLOT), code(CALLS), code(PLOT2), code(PLOT3), code(VERDICT),
        md(NEXT_MD), code(DOWNLOAD),
    ]
    nbf.validate(nb)
    OUT.parent.mkdir(exist_ok=True)
    nbf.write(nb, OUT)
    print(f"wrote {OUT} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
