"""gen.main() end to end with a stubbed vLLM (no GPU): chunked checkpoints, a crash mid-arm and the
resume, overwrite, several datasets on one model load, and the no_code ban. Run: uv run tests/test_resume.py"""
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen  # noqa: E402

N_PROBLEMS, N = 5, 2


class Out:
    finish_reason, stop_reason = "stop", "<|im_end|>"

    def __init__(self, text):
        self.text, self.token_ids = text, list(range(len(text.split())))


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def get_vocab(self):
        return {"a": 1, "```": 7, "`": 8, "python": 9, "import": 10, "important": 11}

    def convert_tokens_to_string(self, toks):
        return "".join(toks)


class FakeLLM:
    calls = []            # prompts of every generate() call
    biases = []           # logit_bias of each call
    crash_on_call = None  # 1-based index of the call that raises

    def __init__(self, **kw):
        self.kw = kw

    def get_tokenizer(self):
        return FakeTok()

    def generate(self, prompts, params, use_tqdm=False):
        FakeLLM.calls.append(list(prompts))
        first = params[0] if isinstance(params, list) else params
        FakeLLM.biases.append(getattr(first, "logit_bias", None))
        if FakeLLM.crash_on_call == len(FakeLLM.calls):
            raise RuntimeError("simulated disconnect")
        plist = params if isinstance(params, list) else [params] * len(prompts)
        return [types.SimpleNamespace(outputs=[Out(f"So \\boxed{{{p.split('PROBLEM ')[1].split()[0]}}}<|im_end|>")
                                               for _ in range(sp.n)]) for p, sp in zip(prompts, plist)]


class FakeSP:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def install_stubs():
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda: (8, 0))
    vllm = types.ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = FakeLLM, FakeSP
    sys.modules["torch"], sys.modules["vllm"] = torch, vllm
    gen.load_problems = lambda name, n, seed: [
        dict(id=f"p{i}", problem=f"PROBLEM p{i} find x", answer=f"p{i}", subject=name, level="1")
        for i in range(N_PROBLEMS)][:n or N_PROBLEMS]
    FakeLLM.calls, FakeLLM.biases, FakeLLM.crash_on_call = [], [], None


def run(out, *extra, conds="tool_on,no_code"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gen.main(["--model", "m/base", "--tag", "base", "--conditions", conds, "--n", str(N), "--chunk", "2",
                  "--dataset", "aime_all", "--max-tokens", "64", "--out", str(out), *extra])
    return buf.getvalue()


def records(path):
    return [json.loads(line) for line in open(path)]


def arm(out, cond, ds="aime_all"):
    return out / ds / f"base__{cond}__verltool_torl.jsonl"


def test_fresh_run_chunks_ungraded_rows_and_ban():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        log = run(out)
        assert "3 chunk(s) to run" in log and "no_code: 3 backtick + program-keyword tokens banned" in log
        recs = records(arm(out, "tool_on"))
        assert [(r["problem_id"], r["sample_idx"]) for r in recs] == [(f"p{i}", j) for i in range(5) for j in range(2)]
        assert all(r["correct"] is None and r["boxed"] == r["gold"] for r in recs)
        meta = json.load(open(gen.meta_path(arm(out, "no_code"))))
        assert meta["n_records"] == 10 and meta["graded"] is False and meta["n_banned"] == 3 and meta["chunk"] == 2
        assert meta["finished_at"].endswith("+00:00")
        # three chunks per arm; the ban is on the no_code calls only, backticks + import, not "important"
        assert FakeLLM.biases == [None] * 3 + [{7: -100.0, 8: -100.0, 10: -100.0}] * 3
        assert "A conversation between User and Assistant" in FakeLLM.calls[0][0]
        assert "all conditions already done" in run(out)


def test_crash_then_resume():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.crash_on_call = 2
        try:
            run(out, conds="tool_on")
            raise AssertionError("expected the simulated crash")
        except RuntimeError:
            pass
        partial = gen.partial_path(arm(out, "tool_on"))
        assert partial.exists() and not arm(out, "tool_on").exists()
        assert sorted({r["problem_id"] for r in records(partial)}) == ["p0", "p1"]
        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        log = run(out, conds="tool_on")
        assert "resuming, 2 problems already on disk; 2 chunk(s) to run" in log
        assert [len(c) for c in FakeLLM.calls] == [2, 1]
        assert [(r["problem_id"], r["sample_idx"]) for r in records(arm(out, "tool_on"))] == \
            [(f"p{i}", j) for i in range(5) for j in range(2)]
        assert json.load(open(gen.meta_path(arm(out, "tool_on"))))["n_resumed"] == 2 and not partial.exists()


def test_overwrite_conditions_and_multi_dataset():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        run(out)
        m_on = gen.meta_path(arm(out, "tool_on")).stat().st_mtime_ns
        m_nc = gen.meta_path(arm(out, "no_code")).stat().st_mtime_ns
        FakeLLM.calls = []
        assert "todo=aime_all/no_code" in run(out, "--overwrite-conditions", "no_code")
        assert gen.meta_path(arm(out, "tool_on")).stat().st_mtime_ns == m_on
        assert gen.meta_path(arm(out, "no_code")).stat().st_mtime_ns > m_nc
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "no_code", "--n", str(N), "--chunk", "0",
                      "--dataset", "aime_all,olymmath_easy:3:1", "--out", str(out)])
        log = buf.getvalue()
        assert log.count("loaded m/base") == 1 and "todo=olymmath_easy/no_code" in log
        assert len(records(arm(out, "no_code", "olymmath_easy"))) == 3
        for bad in (["--conditions", "prompt_off"], ["--overwrite-conditions", "nope"], ["--dataset", "no_such_set"]):
            try:
                gen.main(["--model", "m", "--tag", "base", "--out", str(out), *bad])
                raise AssertionError(bad)
            except SystemExit as e:
                assert "unknown" in str(e)


def test_resume_never_mixes_settings_and_redo_resumes_after_crash():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.crash_on_call = 2
        try:
            run(out, conds="tool_on")                                          # crash after chunk 1 under T=0.6
            raise AssertionError("expected the simulated crash")
        except RuntimeError:
            pass
        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        log = run(out, "--temperature", "1.0", conds="tool_on")               # resume under other settings
        assert "resuming" not in log and [len(c) for c in FakeLLM.calls] == [2, 2, 1]   # nothing reused
        recs = records(arm(out, "tool_on"))
        assert len({r["fp"] for r in recs}) == 1 and len(recs) == 10
        meta = json.load(open(gen.meta_path(arm(out, "tool_on"))))
        assert meta["temperature"] == 1.0 and "commit" in meta and meta["fp"] == recs[0]["fp"]
        # redo with overwrite, crash midway, rerun without overwrite: the redo resumes, the old arm stays retired
        FakeLLM.calls, FakeLLM.crash_on_call = [], 2
        try:
            run(out, "--temperature", "1.0", "--overwrite-conditions", "tool_on", conds="tool_on")
            raise AssertionError("expected the simulated crash")
        except RuntimeError:
            pass
        assert not arm(out, "tool_on").exists() and Path(str(arm(out, "tool_on")) + ".replaced").exists()
        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        log = run(out, "--temperature", "1.0", conds="tool_on")
        assert "resuming, 2 problems already on disk" in log and arm(out, "tool_on").exists()

def test_plain_arm_uses_cot_prompt_and_ban():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "env_denied,plain", "--template", "torl",
                      "--n", str(N), "--chunk", "0", "--dataset", "amc23", "--out", str(out)])
        assert (out / "amc23" / "base__env_denied__torl.jsonl").exists()
        assert (out / "amc23" / "base__plain__qwen_math_cot.jsonl").exists()
        assert FakeLLM.biases == [None, {7: -100.0, 8: -100.0, 10: -100.0}]
        assert FakeLLM.calls[0][0].endswith("Assistant:") and FakeLLM.calls[1][0].startswith("<|im_start|>system\nPlease reason")
        meta = json.load(open(out / "amc23" / "base__plain__qwen_math_cot.meta.json"))
        assert meta["template_used"] == "qwen_math_cot" and meta["n_banned"] == 3

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
