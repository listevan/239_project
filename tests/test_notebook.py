"""Tests for the notebook's run-cell helpers (stream / status / arm_files / stale handling), executed
from the built notebook so the glue between the cells and gen.py is what gets tested."""
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


def notebook_cell(marker):
    nb = json.load(open(NB))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    hits = [c for c in cells if marker in c]
    assert len(hits) == 1, f"{marker}: {len(hits)} cells"
    return hits[0]


def run_setup_namespace(out, datasets=None, conditions=("prompt_off", "env_denied", "tool_on")):
    datasets = datasets or {"math500": (0, 2)}
    ns = dict(
        PRESETS={"t": dict(max_model_len=4096, models={"base": dict(model="m/base", native="qwen_math_tir")})},
        PRESET="t", OUT=str(out), RUN_ID=Path(out).name, DATASETS=datasets, CONDITIONS=list(conditions),
        DATASET_SPEC=",".join(f"{k}:{p}:{n}" for k, (p, n) in datasets.items()),
        SHARED_TEMPLATE="qwen_math_nocode", OVERWRITE=False, MAX_TOKENS=64, MAX_MODEL_LEN=4096,
        TEMPERATURE=0.6, TOP_P=0.95, SEED=0, MAX_CALLS=4, GPU_MEM=0.9, CHUNK=2, ALLOW_LONG=False,
        DENIAL_TEXT=None, GRADE_ON_COLAB=False,
    )
    exec(compile(notebook_cell("def stream(cmd)"), "run_setup_cell", "exec"), ns)
    return ns


def rec(pid, k, tag="base", cond="prompt_off", correct=1):
    return dict(tag=tag, condition=cond, problem_id=pid, sample_idx=k, correct=correct, truncated=False,
                n_calls=0, n_denied=0)


def test_config_cell_builds_run_id_spec_and_config_json():
    with tempfile.TemporaryDirectory() as d:
        ns = dict(RUNS=str(Path(d) / "runs"))
        exec(compile(notebook_cell("DATASET_SPEC = "), "config_cell", "exec"), ns)
        assert ns["SMOKE"] is False and ns["DATASET_SPEC"] == "aime_all:0:64,olymmath_hard:0:64"
        assert ns["RUN_ID"].endswith("_verltool-1.5b") and ns["RUN_ID"][:4].isdigit()
        cfg = json.load(open(Path(ns["OUT"]) / "config.json"))
        assert cfg["datasets"] == {"aime_all": [0, 64], "olymmath_hard": [0, 64]} and cfg["run_id"] == ns["RUN_ID"]
        assert cfg["conditions_by_model"] == {"tool_rl": ["tool_on", "env_denied", "no_code"],
                                              "base": ["tool_on", "env_denied", "no_code"], "text_rl": ["no_code"]}
        assert {m["native"] for m in cfg["models"].values()} == {"verltool_torl"}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(notebook_cell("DATASET_SPEC = ").replace("SMOKE = False", "SMOKE = True"), "config_cell", "exec"),
                 dict(RUNS=str(Path(d) / "runs")))
        assert "runs on Drive: " + ns["RUN_ID"] in buf.getvalue() and "_verltool-1.5b-smoke" in buf.getvalue()
        ns2 = dict(RUNS=str(Path(d) / "runs"))
        src = notebook_cell("DATASET_SPEC = ").replace("RUN_ID = None", f'RUN_ID = "{ns["RUN_ID"]}"').replace("MAX_TOKENS = 3000", "MAX_TOKENS = 999")
        with contextlib.redirect_stdout(buf):
            exec(compile(src, "config_cell", "exec"), ns2)
        assert "DIFFERS in ['max_tokens']" in buf.getvalue()


def test_stream_relays_output_and_exit_code():
    ns = run_setup_namespace("/nonexistent")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = ns["stream"]([sys.executable, "-c",
                           "import sys; print('a\\rb\\rc'); sys.stderr.write('err\\n'); sys.exit(3)"])
    assert rc == 3
    assert "a\nb\nc" in buf.getvalue() and "err" in buf.getvalue()


def test_stream_interrupt_kills_process_group():
    ns = run_setup_namespace("/nonexistent")
    child = ("import os, subprocess; p = subprocess.Popen(['sleep', '30']); "
             "print(os.getpid(), p.pid, flush=True); p.wait()")
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
    assert time.time() - t0 < 10, "kill took too long"
    pid, gpid = map(int, buf.getvalue().split("\n")[0].split())
    assert "[interrupted] killed" in buf.getvalue()
    for p in (pid, gpid):
        for _ in range(40):  # the grandchild is reparented to init and reaped asynchronously
            try:
                os.kill(p, 0)
                time.sleep(0.05)
            except ProcessLookupError:
                break
        else:
            raise AssertionError(f"pid {p} still alive after interrupt")


def test_arm_files_status_per_dataset():
    import gen
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "2026-10-15_t"
        ns = run_setup_namespace(out, datasets={"math500": (0, 2), "gsm_hard": (10, 2)})
        rows = ns["arm_files"]("base", "gsm_hard")
        assert [r[0] for r in rows] == ["prompt_off", "env_denied", "tool_on"]
        assert rows[0][1] == out / "gsm_hard" / "base__prompt_off__qwen_math_nocode.jsonl"
        assert rows[1][1].name == "base__env_denied__qwen_math_tir.jsonl"
        assert rows[1][2] == gen.meta_path(rows[1][1]) and rows[1][3] == gen.partial_path(rows[1][1])
        (out / "math500").mkdir(parents=True); (out / "gsm_hard").mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 6  # nothing on disk, two datasets
        m = ns["arm_files"]("base", "math500")
        gen.write_jsonl(m[0][3], [rec("p0", 0), rec("p0", 1), rec("p1", 0)])
        gen.write_jsonl_atomic(m[1][1], [rec("p0", 0, cond="env_denied"), rec("p0", 1, cond="env_denied")])
        m[1][2].write_text(json.dumps(dict(pass1=1.0, n=2, n_problems=1, gen_seconds=60, chunk=100)))
        g = ns["arm_files"]("base", "gsm_hard")
        gen.write_jsonl_atomic(g[2][1], [rec("g0", 0, cond="tool_on"), rec("g0", 1, cond="tool_on")])
        g[2][2].write_text(json.dumps(dict(pass1=None, n=4, n_problems=10, gen_seconds=60, chunk=100)))
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 4  # math500: partial, done, missing; gsm_hard: missing, missing, done
        txt = buf.getvalue()
        assert "base     math500   prompt_off  partial  2 problems checkpointed" in txt
        assert "base     math500   env_denied  done     pass@1=1.000" in txt
        assert "base     gsm_hard  tool_on     done     ungraded  n=4" in txt and "different problems/samples" in txt


def test_stale_files_are_flagged_quarantined_and_excluded():
    import gen
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        folder = out / "math500"; folder.mkdir(parents=True)
        ns = run_setup_namespace(out)
        cur = folder / "base__prompt_off__qwen_math_nocode.jsonl"
        gen.write_jsonl_atomic(cur, [rec("p0", 0), rec("p0", 1)])
        gen.meta_path(cur).write_text(json.dumps(dict(pass1=1.0, n=2, n_problems=1, gen_seconds=1, chunk=100)))
        old = folder / "base__prompt_off__qwen_math_cot.jsonl"  # last week's run, other template
        gen.write_jsonl_atomic(old, [rec("p0", 0), rec("p0", 1)])
        gen.meta_path(old).write_text(json.dumps(dict(pass1=0.5, n=2, n_problems=1, gen_seconds=1)))
        oldv = folder / "base__tool_on__qwen_math_tir.jsonl"    # same name as now, older gen.py
        gen.write_jsonl_atomic(oldv, [rec("p0", 0, cond="tool_on"), rec("p0", 1, cond="tool_on")])
        gen.meta_path(oldv).write_text(json.dumps(dict(pass1=0.5, n=2, n_problems=1, gen_seconds=1)))
        (folder / "passk.png").write_bytes(b"x")

        stale = {f.name for f, _ in ns["stale_files"]("math500")}
        assert stale == {old.name, gen.meta_path(old).name, oldv.name, gen.meta_path(oldv).name}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = ns["load_run"]("math500")
            ns["status"]("base")
        assert len(df) == 2 and set(df.file) == {cur.name}           # the duplicate arm is excluded
        assert "ignoring 2 file(s)" in buf.getvalue() and "stale file(s)" in buf.getvalue()
        assert "older notebook version" in buf.getvalue()
        with contextlib.redirect_stdout(buf):
            ns["move_stale"]()
        assert sorted(p.name for p in (folder / "old").iterdir()) == sorted(stale)
        assert cur.exists() and (folder / "passk.png").exists() and not ns["stale_files"]("math500")
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 2  # tool_on is now missing, so it will be regenerated


def test_status_and_show_handle_ungraded_rows():
    import gen
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"; folder = out / "math500"; folder.mkdir(parents=True)
        ns = run_setup_namespace(out)
        rows = ns["arm_files"]("base", "math500")
        gen.write_jsonl_atomic(rows[0][1], [rec("p0", 0, correct=None), rec("p0", 1, correct=None)])
        rows[0][2].write_text(json.dumps(dict(pass1=None, graded=False, n=2, n_problems=1, gen_seconds=30, chunk=100)))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 2
            ns["show"]("base")
        txt = buf.getvalue()
        assert "done     ungraded" in txt and "rows are ungraded" in txt
        assert "pass1" not in txt.split("rows are ungraded")[1]
        try:
            ns["load_run"]("math500", require_graded=True)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "grade.py" in str(e)


def test_notebook_clones_instead_of_embedding():
    nb = json.load(open(NB))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert not any(c.startswith("%%writefile") for c in cells)
    clone = [c for c in cells if 'REPO = "listevan/239_project"' in c]
    assert len(clone) == 1 and "github_token.txt" in clone[0] and "git" in clone[0]
    assert "print(url" not in clone[0] and "print(tok" not in clone[0]   # the token is never printed

def test_arm_of_unselected_condition_is_kept_and_shown():
    import gen
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"; folder = out / "math500"; folder.mkdir(parents=True)
        ns = run_setup_namespace(out, conditions=("prompt_off", "env_denied", "tool_on"))   # kernel list lacks prose_only
        cur = folder / "base__prompt_off__qwen_math_nocode.jsonl"
        gen.write_jsonl_atomic(cur, [rec("p0", 0), rec("p0", 1)])
        gen.meta_path(cur).write_text(json.dumps(dict(pass1=1.0, n=2, n_problems=1, gen_seconds=1, chunk=100)))
        prose = folder / "base__prose_only__qwen_math_nocode.jsonl"
        gen.write_jsonl_atomic(prose, [rec("p0", 0, cond="prose_only"), rec("p0", 1, cond="prose_only", correct=0)])
        gen.meta_path(prose).write_text(json.dumps(dict(pass1=0.5, n=2, n_problems=1, gen_seconds=1, chunk=100)))
        assert ns["stale_files"]("math500") == []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            df = ns["load_run"]("math500")
            ns["show"]("base")
        assert set(df.condition) == {"prompt_off", "prose_only"} and "ignoring" not in buf.getvalue()
        assert "prose_only" in buf.getvalue()

def test_per_model_conditions():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        ns = run_setup_namespace(out)
        ns["PRESETS"]["t"]["models"]["text_rl"] = dict(model="m/text", native="qwen_math_tir")
        ns["PRESETS"]["t"]["models"]["tool_rl"] = dict(model="m/tool", native="qwen_math_tir")
        ns["CONDITIONS_BY_MODEL"] = {"base": ["tool_on", "no_code"], "text_rl": ["no_code"]}
        assert [r[0] for r in ns["arm_files"]("text_rl", "math500")] == ["no_code"]
        assert [r[0] for r in ns["arm_files"]("base", "math500")] == ["tool_on", "no_code"]
        assert [r[0] for r in ns["arm_files"]("tool_rl", "math500")] == ["prompt_off", "env_denied", "tool_on"]  # fallback
        (out / "math500").mkdir(parents=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("text_rl") == 1

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
