"""End-to-end test of gen.main() with a stubbed vLLM (no GPU): chunked checkpoints, a crash
mid-arm, and the resume that follows. Run: python tests/test_main_resume.py"""
import contextlib
import io
import json
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
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
    biases = []           # logit_bias of the first SamplingParams of every call
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
        res = []
        for prompt, p in zip(prompts, plist):
            pid = prompt.split("PROBLEM ")[1].split()[0]  # the problem text carries its id
            res.append(types.SimpleNamespace(outputs=[Out(f"So \\boxed{{{pid}}}<|im_end|>") for _ in range(p.n)]))
        return res


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
    gen.make_grader = lambda: ((lambda resp, gold: f"\\boxed{{{gold}}}" in resp), "stub")


def run_main(out, conds="prompt_off,env_denied"):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        gen.main(["--model", "m/base", "--tag", "base", "--conditions", conds, "--n", str(N),
                  "--chunk", "2", "--max-tokens", "64", "--out", str(out)])
    return buf.getvalue()


def records(path):
    return [json.loads(l) for l in open(path)]


def test_fresh_run_writes_every_arm_in_chunks():
    install_stubs()
    FakeLLM.calls, FakeLLM.crash_on_call = [], None
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        log = run_main(out)
        assert "3 chunk(s) to run" in log and "chunk 3/3" in log
        arms = sorted(p.name for p in (out / "math500").glob("*"))
        assert arms == ["base__env_denied__qwen_math_tir.jsonl", "base__env_denied__qwen_math_tir.meta.json",
                        "base__prompt_off__qwen_math_nocode.jsonl", "base__prompt_off__qwen_math_nocode.meta.json"]
        recs = records(out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl")
        assert [(r["problem_id"], r["sample_idx"]) for r in recs] == [(f"p{i}", j) for i in range(5) for j in range(2)]
        assert all(r["correct"] for r in recs)
        meta = json.load(open(out / "math500" / "base__prompt_off__qwen_math_nocode.meta.json"))
        assert meta["n_records"] == 10 and meta["n_resumed"] == 0 and meta["pass1"] == 1.0 and meta["chunk"] == 2
        assert meta["finished_at"].endswith("+00:00") and meta["finished_at"][:4].isdigit()
        assert [len(c) for c in FakeLLM.calls] == [2, 2, 1, 2, 2, 1]  # chunks x conditions, one round each
        assert "all conditions already done" in run_main(out)


def test_crash_mid_arm_then_resume():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.crash_on_call = [], 2  # chunk 1 lands, chunk 2 dies
        try:
            run_main(out)
            raise AssertionError("expected the simulated crash")
        except RuntimeError:
            pass
        arm = out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl"
        partial = gen.partial_path(arm)
        assert partial.exists() and not arm.exists() and not gen.meta_path(arm).exists()
        assert sorted({r["problem_id"] for r in records(partial)}) == ["p0", "p1"]

        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        log = run_main(out)
        assert "resuming, 2 problems already on disk; 2 chunk(s) to run" in log
        assert [len(c) for c in FakeLLM.calls] == [2, 1, 2, 2, 1]  # p2,p3 | p4, then the untouched arm
        assert FakeLLM.calls[0][0].count("PROBLEM p2") == 1 and "PROBLEM p4" in FakeLLM.calls[1][0]
        recs = records(arm)
        assert [(r["problem_id"], r["sample_idx"]) for r in recs] == [(f"p{i}", j) for i in range(5) for j in range(2)]
        meta = json.load(open(gen.meta_path(arm)))
        assert meta["n_resumed"] == 2 and meta["n_records"] == 10
        assert not partial.exists()
        assert "all conditions already done" in run_main(out)


def test_overwrite_discards_partial_and_redoes():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        run_main(out, conds="prompt_off")
        buf = io.StringIO()
        FakeLLM.calls = []
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off", "--n", str(N),
                      "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--overwrite"])
        assert "1 chunk(s) to run" in buf.getvalue() and [len(c) for c in FakeLLM.calls] == [5]


def test_no_grade_then_local_grade():
    install_stubs()
    import grade as gr
    import analyze as an
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.crash_on_call = [], None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off", "--n", str(N),
                      "--chunk", "2", "--max-tokens", "64", "--out", str(out), "--no-grade"])
        log = buf.getvalue()
        assert "grade with src/grade.py" in log and "ungraded" in log and "writing 4 records" in log
        arm = out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl"
        assert all(r["correct"] is None and r["arith_n"] is None for r in records(arm))
        meta = json.load(open(gen.meta_path(arm)))
        assert meta["graded"] is False and meta["pass1"] is None
        try:
            an.load_results(out / "math500")
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "grade.py" in str(e)
        assert len(an.load_results(out / "math500", allow_ungraded=True)) == 10
        gr._GRADER = None  # serial path, stubbed grader
        with contextlib.redirect_stdout(buf):
            rows = gr.grade_dir(out / "math500", workers=0, timeout=5)
        assert rows[0]["graded_now"] == 10 and rows[0]["pass1"] == 1.0
        assert all(r["correct"] is True and r["arith_n"] == 0 for r in records(arm))
        meta = json.load(open(gen.meta_path(arm)))
        assert meta["graded"] is True and meta["pass1"] == 1.0 and meta["grader"] == "stub"
        with contextlib.redirect_stdout(buf):
            assert gr.grade_dir(out / "math500", workers=0)[0]["graded_now"] == 0  # idempotent
        assert an.load_results(out / "math500")["correct"].dtype == bool

def test_multi_dataset_one_model_load():
    install_stubs()
    FakeLLM.calls, FakeLLM.crash_on_call = [], None
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off",
                      "--dataset", "math500,aime24:3:1", "--n", str(N), "--chunk", "0", "--max-tokens", "64",
                      "--out", str(out), "--no-grade"])
        log = buf.getvalue()
        assert log.count("loaded m/base") == 1 and "todo=math500/prompt_off, aime24/prompt_off" in log
        a = records(out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl")
        b = records(out / "aime24" / "base__prompt_off__qwen_math_nocode.jsonl")
        assert len(a) == N_PROBLEMS * N and len(b) == 3 * 1
        assert {r["dataset"] for r in a} == {"math500"} and {r["dataset"] for r in b} == {"aime24"}
        ma = json.load(open(gen.meta_path(out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl")))
        mb = json.load(open(gen.meta_path(out / "aime24" / "base__prompt_off__qwen_math_nocode.jsonl")))
        assert (ma["dataset"], ma["n"], ma["n_problems"]) == ("math500", N, N_PROBLEMS)
        assert (mb["dataset"], mb["n"], mb["n_problems"], mb["n_problems_requested"]) == ("aime24", 1, 3, 3)
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off",
                      "--dataset", "math500,aime24:3:1", "--n", str(N), "--out", str(out), "--no-grade"])
        assert "all conditions already done" in buf.getvalue()

def test_prose_only_bans_backticks_and_uses_shared_prompt():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.biases, FakeLLM.crash_on_call = [], [], None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prose_only,prompt_off", "--n", str(N),
                      "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--no-grade"])
        assert "prose_only: 2 backtick tokens banned" in buf.getvalue()
        assert FakeLLM.biases == [{7: -100.0, 8: -100.0}, None]      # ban on the prose_only arm only
        arm = out / "math500" / "base__prose_only__qwen_math_nocode.jsonl"   # shared no-code template
        recs = records(arm)
        assert len(recs) == N_PROBLEMS * N and all(r["n_calls"] == 0 for r in recs)
        assert json.load(open(gen.meta_path(arm)))["condition"] == "prose_only"

def test_overwrite_conditions_redoes_only_those():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.biases, FakeLLM.crash_on_call = [], [], None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off,env_denied", "--n", str(N),
                      "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--no-grade"])
        po = out / "math500" / "base__prompt_off__qwen_math_nocode.jsonl"
        ed = out / "math500" / "base__env_denied__qwen_math_tir.jsonl"
        m_po, m_ed = gen.meta_path(po).stat().st_mtime_ns, gen.meta_path(ed).stat().st_mtime_ns
        FakeLLM.calls = []
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prompt_off,env_denied", "--n", str(N),
                      "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--no-grade",
                      "--overwrite-conditions", "env_denied"])
        assert "todo=math500/env_denied" in buf.getvalue() and len(FakeLLM.calls) == 1
        assert gen.meta_path(po).stat().st_mtime_ns == m_po and gen.meta_path(ed).stat().st_mtime_ns > m_ed
        try:
            gen.main(["--model", "m/base", "--tag", "base", "--out", str(out), "--overwrite-conditions", "nope"])
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            assert "unknown condition" in str(e)

def test_redirect_and_assist_arms_use_the_right_templates():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.biases, FakeLLM.crash_on_call = [], [], None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "prose_redirect,assist_denied", "--n", str(N),
                      "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--no-grade"])
        assert (out / "math500" / "base__prose_redirect__qwen_math_nocode.jsonl").exists()
        assert (out / "math500" / "base__assist_denied__qwen_math_tir.jsonl").exists()
        assert FakeLLM.biases == [None, None]   # no arm-wide ban: the redirect ban is per sample, after the cap

def test_no_code_arm_bans_backticks_and_uses_native_prompt():
    install_stubs()
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        FakeLLM.calls, FakeLLM.biases, FakeLLM.crash_on_call = [], [], None
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            gen.main(["--model", "m/base", "--tag", "base", "--conditions", "no_code", "--native-template", "verltool_torl",
                      "--n", str(N), "--chunk", "0", "--max-tokens", "64", "--out", str(out), "--no-grade"])
        assert "no_code: 3 backtick + program-keyword tokens banned" in buf.getvalue()
        assert FakeLLM.biases == [{7: -100.0, 8: -100.0, 10: -100.0}]            # backticks + import, not important
        assert (out / "math500" / "base__no_code__verltool_torl.jsonl").exists()
        assert "A conversation between User and Assistant" in FakeLLM.calls[0][0]

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
