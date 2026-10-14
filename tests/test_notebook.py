"""Tests for the notebook's run-cell helpers (stream / status / arm_files), executed from the
built notebook so the glue between the cells and gen.py is what gets tested."""
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


def run_setup_namespace(out, conditions=("prompt_off", "env_denied", "tool_on"), n=2, n_problems=0):
    ns = dict(
        PRESETS={"t": dict(max_model_len=4096, models={"base": dict(model="m/base", native="qwen_math_tir")})},
        PRESET="t", OUT=str(out), DATASET="math500", CONDITIONS=list(conditions),
        SHARED_TEMPLATE="qwen_math_nocode", N_SAMPLES=n, N_PROBLEMS=n_problems, OVERWRITE=False,
        MAX_TOKENS=64, MAX_MODEL_LEN=4096, TEMPERATURE=0.6, TOP_P=0.95, SEED=0, MAX_CALLS=4,
        GPU_MEM=0.9, CHUNK=2, ALLOW_LONG=False, DENIAL_TEXT=None,
    )
    exec(compile(notebook_cell("def stream(cmd)"), "run_setup_cell", "exec"), ns)
    return ns


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


def test_arm_files_match_gen_naming_and_status_reads_meta():
    import gen
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "run"
        ns = run_setup_namespace(out, n=2)
        rows = ns["arm_files"]("base")
        assert [r[0] for r in rows] == ["prompt_off", "env_denied", "tool_on"]
        assert rows[0][1].name == "base__prompt_off__qwen_math_nocode.jsonl"
        assert rows[1][1].name == "base__env_denied__qwen_math_tir.jsonl"
        assert rows[1][2] == gen.meta_path(rows[1][1]) and rows[1][3] == gen.partial_path(rows[1][1])
        (out / "math500").mkdir(parents=True)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 3  # nothing on disk
        rec = lambda pid, sidx: dict(problem_id=pid, sample_idx=sidx, correct=1)  # noqa: E731
        gen.write_jsonl(rows[0][3], [rec("p0", 0), rec("p0", 1), rec("p1", 0)])
        gen.write_jsonl_atomic(rows[1][1], [rec("p0", 0), rec("p0", 1)])
        rows[1][2].write_text(json.dumps(dict(pass1=1.0, n=2, n_problems=1, gen_seconds=60)))
        with contextlib.redirect_stdout(buf):
            assert ns["status"]("base") == 2  # one done, one partial, one missing
        txt = buf.getvalue()
        assert "partial  2 problems checkpointed" in txt and "done     pass@1=1.000" in txt and "missing" in txt
        ns["N_SAMPLES"] = 4
        with contextlib.redirect_stdout(buf):
            ns["status"]("base")
        assert "different N_SAMPLES" in buf.getvalue()


if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
