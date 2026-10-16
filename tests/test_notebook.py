"""The notebook's config and run-helper cells, executed from the built notebook. Run: uv run tests/test_notebook.py"""
import contextlib
import io
import json
import os
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "tool_atrophy_pilot.ipynb"
sys.path.insert(0, str(ROOT / "src"))
import gen  # noqa: E402


def cells():
    return ["".join(c["source"]) for c in json.load(open(NB))["cells"] if c["cell_type"] == "code"]


def cell(marker):
    hits = [c for c in cells() if marker in c]
    assert len(hits) == 1, f"{marker}: {len(hits)} cells"
    return hits[0]


def setup_ns(out, datasets=None):
    datasets = datasets or {"aime_all": (0, 2)}
    ns = dict(OUT=str(out), MODELS={"tool_rl": "m/tool", "base": "m/base", "text_rl": "m/text"},
              TEMPLATES={"tool_rl": "verltool_torl", "base": "verltool_torl", "text_rl": "verltool_torl"},
              CONDITIONS_BY_MODEL={"tool_rl": ["tool_on", "env_denied", "no_code"], "base": ["tool_on", "no_code"],
                                   "text_rl": ["no_code"]},
              DATASETS=datasets, DATASET_SPEC=",".join(f"{k}:{p}:{n}" for k, (p, n) in datasets.items()),
              MAX_TOKENS=64, MAX_MODEL_LEN=4096, TEMPERATURE=0.6, TOP_P=0.95, SEED=0, MAX_CALLS=4, GPU_MEM=0.9,
              CHUNK=2, OVERWRITE=False)
    exec(compile(cell("def run_model(tag)"), "run_setup", "exec"), ns)
    return ns


def test_structure():
    c = cells()
    assert len(c) == 9 and not any(x.startswith("%%writefile") for x in c)
    assert c[-3:] == ['run_model("tool_rl")', 'run_model("base")', 'run_model("text_rl")']
    assert "github.com/listevan/239_project.git" in cell("git")


def test_config_cell():
    with tempfile.TemporaryDirectory() as d:
        runs = Path(d) / "runs"
        ns = dict(RUNS=str(runs))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(cell("DATASET_SPEC = "), "config", "exec"), ns)
        assert ns["SMOKE"] is False and ns["DATASET_SPEC"] == "math500_l5:30:4"
        assert ns["RUN_ID"].endswith("_torl-1.5b")
        cfg = json.load(open(Path(ns["OUT"]) / "config.json"))
        assert cfg["conditions_by_model"] == {"tool_rl": ["tool_on", "env_denied", "plain"],
                                              "base": ["tool_on", "env_denied", "plain"], "text_rl": []}
        assert cfg["templates"] == {"tool_rl": "torl", "base": "qwen_math_tir", "text_rl": "qwen_math_tir"}
        assert cfg["models"]["tool_rl"] == "GAIR/ToRL-1.5B"
        src = cell("DATASET_SPEC = ").replace("RUN_ID = None", f'RUN_ID = "{ns["RUN_ID"]}"').replace(
            "MAX_TOKENS = 1024 if SMOKE else 3000", "MAX_TOKENS = 999")
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "config", "exec"), dict(RUNS=str(runs)))
        assert "(resuming)" in buf.getvalue() and "changed ['max_tokens']" in buf.getvalue()
        cfg2 = json.load(open(Path(ns["OUT"]) / "config.json"))
        assert cfg2["max_tokens"] == 999 and cfg2["history"][-1]["max_tokens"] == 3000   # current settings + history
        with contextlib.redirect_stdout(buf):
            exec(compile(cell("DATASET_SPEC = ").replace("SMOKE = False", "SMOKE = True"), "config", "exec"),
                 dict(RUNS=str(runs)))
        assert "_torl-1.5b-smoke (new)" in buf.getvalue()   # smoke never writes into the real run


def test_arm_files_status_and_show():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        ns = setup_ns(out, {"aime_all": (0, 2), "olymmath_easy": (3, 2)})
        assert [r[0] for r in ns["arm_files"]("text_rl", "aime_all")] == ["no_code"]
        assert ns["arm_files"]("base", "aime_all")[1][1] == out / "aime_all" / "base__no_code__verltool_torl.jsonl"
        (out / "aime_all").mkdir(parents=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("tool_rl") == 6
        jsonl, meta, partial = ns["arm_files"]("base", "aime_all")[0][1:]
        rec = lambda pid, k: dict(problem_id=pid, sample_idx=k, condition="tool_on", correct=None, truncated=False,  # noqa: E731
                                  n_calls=1, n_denied=0)
        gen.write_jsonl_atomic(jsonl, [rec("p0", 0), rec("p0", 1)])
        meta.write_text(json.dumps(dict(n=2, n_problems=1, gen_seconds=60)))
        gen.write_jsonl(ns["arm_files"]("base", "aime_all")[1][3], [rec("p0", 0), rec("p0", 1)])
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 3          # aime_all: tool_on done, no_code partial; olymmath: 2 missing
            ns["show"]("base")
        txt = buf.getvalue()
        assert "base     aime_all       tool_on     done     n=2" in txt and "partial  1 problems checkpointed" in txt
        assert "base / aime_all" in txt and "pass1" not in txt.split("base / aime_all")[1]


def test_stream_output_and_interrupt():
    ns = setup_ns("/nonexistent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ns["stream"]([sys.executable, "-c", "import sys; print('a\\rb'); sys.stderr.write('err\\n'); sys.exit(3)"])
    assert rc == 3 and "a\nb" in buf.getvalue() and "err" in buf.getvalue()
    child = "import os, subprocess; p = subprocess.Popen(['sleep', '30']); print(os.getpid(), p.pid, flush=True); p.wait()"
    main = threading.main_thread().ident
    threading.Timer(0.8, lambda: signal.pthread_kill(main, signal.SIGINT)).start()
    buf = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(buf):
            ns["stream"]([sys.executable, "-c", child])
        raise AssertionError("expected KeyboardInterrupt")
    except KeyboardInterrupt:
        pass
    assert time.time() - t0 < 10 and "[interrupted] killed" in buf.getvalue()
    for p in map(int, buf.getvalue().split("\n")[0].split()):
        for _ in range(40):
            try:
                os.kill(p, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                break
        else:
            raise AssertionError(f"pid {p} still alive after interrupt")


def test_model_without_arms_is_skipped():
    ns = setup_ns("/nonexistent")
    ns["CONDITIONS_BY_MODEL"] = {"tool_rl": ["tool_on"], "base": ["tool_on"], "text_rl": []}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ns["run_model"]("text_rl")
    assert "no arms configured" in buf.getvalue()

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
