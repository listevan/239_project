"""gen.py helpers and the tool loop, no GPU. Run: uv run tests/test_gen.py"""
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen  # noqa: E402


def args(**kw):
    base = dict(max_tokens=400, max_model_len=4096, max_calls=2, exec_timeout=5.0, seed=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_text_helpers():
    assert gen.last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert gen.last_boxed(r"\boxed{a} then \boxed{(3, \frac{\pi}{2})}") == r"(3, \frac{\pi}{2})"
    assert gen.last_boxed("no answer") is None
    t = "text\n```python\nprint(1+1)\n```\n"
    assert gen.last_code_block(t) == "print(1+1)\n" and gen.last_code_block("```\nnot python\n```\n") is None
    assert "print" not in gen.strip_fenced(t)


def test_prompt():
    p = gen.build_prompt("verltool_torl", "What is 1+1?")
    assert p.startswith("<|im_start|>system\nA conversation between User and Assistant.")
    assert p.endswith("\\boxed{}.:\n<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n")


def test_execute():
    out, err = gen.execute("print(6*7)")
    assert out == "42" and not err
    out, err = gen.execute("1/0")
    assert err and "ZeroDivisionError" in out
    out, err = gen.execute("input()")
    assert err                                   # stdin is closed: fails fast
    out, err = gen.execute("while True: pass", timeout=1)
    assert err and out.startswith("Timeout")


def test_tool_on_executes():
    s = gen.Sample(0, 0, "P")
    inj = gen.advance(s, "Compute.\n```python\nprint(2+2)\n```\n", 8, "stop", "```\n", "tool_on", 4,
                      executor=lambda c: ("4", False))
    assert inj == "```output\n4\n```\n" and s.n_calls == 1 and s.n_denied == 0 and not s.done
    gen.advance(s, r"So \boxed{4}", 3, "stop", "<|im_end|>", "tool_on", 4)
    assert s.done and s.finish == "eos" and s.gen.endswith(r"\boxed{4}")


def test_env_denied_denies():
    s = gen.Sample(0, 0, "P")
    inj = gen.advance(s, "```python\nprint(2+2)\n```\n", 5, "stop", "```\n", "env_denied", 4,
                      executor=lambda c: (_ for _ in ()).throw(AssertionError("must not execute")))
    assert gen.DENIAL in inj and s.n_calls == 1 and s.n_denied == 1 and s.n_exec_err == 0


def test_model_opens_output_block_itself():
    s = gen.Sample(0, 0, "P")
    inj = gen.advance(s, "```python\nprint(1)\n```\n```output", 6, "stop", "```output", "tool_on", 4,
                      executor=lambda c: ("1", False))
    assert inj == "\n1\n```\n" and s.n_injected == 1


def test_budget_then_budget_message():
    s = gen.Sample(0, 0, "P")
    for i in range(3):
        gen.advance(s, "```python\nprint(1)\n```\n", 4, "stop", "```\n", "tool_on", 2, executor=lambda c: ("1", False))
    assert s.n_calls == 3 and s.n_denied == 1 and gen.DENIAL_BUDGET in s.gen


def test_fabricated_output_and_truncation():
    s = gen.Sample(0, 0, "P")
    gen.advance(s, "```output\n42\n```\n", 4, "stop", "```\n", "env_denied", 4)
    assert s.n_calls == 0 and not s.done           # a model-written output block is not a call
    gen.advance(s, "and more", 2, "length", None, "env_denied", 4)
    assert s.done and s.truncated and s.finish == "length"
    rec = gen.to_record(s, dict(id="x", answer="1"), "env_denied", "t", "m", "base", "aime_all")
    assert rec["fake_outputs"] == 1 and rec["correct"] is None and rec["arith_n"] is None and rec["dataset"] == "aime_all"


def scripted_generate(script):
    """script[i] answers a context that already holds i output blocks."""
    def generate(prompts, stops, budgets, seeds):
        return [script[min(p.count("```output"), len(script) - 1)] for p in prompts]
    return generate


def test_tool_loop_finishes_and_round_cap():
    s = gen.Sample(0, 0, "P")
    gen.advance(s, "```python\nprint(1)\n```\n", 4, "stop", "```\n", "tool_on", 2, executor=lambda c: ("1", False))
    script = [None, ("```python\nprint(2)\n```\n", 4, "stop", "```\n"), (r"\boxed{2}", 3, "stop", "<|im_end|>")]
    pool = ThreadPoolExecutor(max_workers=1)
    gen.run_tool_loop(scripted_generate(script), lambda t: len(t.split()), [s], "tool_on", args(), pool)
    assert s.done and s.finish == "eos" and s.n_calls == 2
    # a model that never stops calling hits the round cap, then gets one round with text stops only
    s2 = gen.Sample(0, 1, "P")
    seen = []

    def forever(prompts, stops, budgets, seeds):
        seen.append(stops)
        final = stops == gen.TEXT_STOPS
        return [((r"\boxed{1}", 2, "stop", "<|im_end|>") if final else ("```python\nprint(1)\n```\n", 4, "stop", "```\n"))
                for _ in prompts]
    gen.run_tool_loop(forever, lambda t: 1, [s2], "env_denied", args(max_tokens=10_000), pool)
    pool.shutdown()
    assert s2.done and seen[-1] == gen.TEXT_STOPS and s2.rounds == 2 * 2 + 6 + 1


def test_code_token_ids():
    vocab = {"`": 0, " ```": 1, "Ġimport": 2, "import": 3, "important": 4, "importância": 5, "print": 6, "Ġprint(": 7,
             "#": 8, "###": 9, "Ġ==": 10, "=": 11, "lambda": 12, "Ġdef": 13, "Ġnp": 14, "Ġrange": 15, "Ġsympy": 16,
             "Ġfor": 17, "Ġreturn": 18, "Ġ//": 19, "Ġ+=": 20, "elif": 21, "Ġitertools": 22, "Ġnumpy": 23, "Ġsymp": 24}

    class Tok:
        def get_vocab(self):
            return vocab

        def convert_tokens_to_string(self, toks):
            return "".join(toks).replace("Ġ", " ")

    banned = {k for k, i in vocab.items() if i in gen.code_token_ids(Tok())}
    assert banned == {"`", " ```", "Ġimport", "import", "print", "Ġprint(", "#", "Ġ==", "Ġsympy", "Ġ//", "Ġ+=", "elif",
                      "Ġitertools", "Ġnumpy", "Ġsymp"}
    assert gen.code_token_ids(object()) == []


def test_dataset_specs_and_loaders():
    assert gen.parse_dataset_specs("aime_all::64,olymmath_easy:10", 0, 16) == [("aime_all", 0, 64), ("olymmath_easy", 10, 16)]
    for bad in ("nope", "", "aime_all:x"):
        try:
            gen.parse_dataset_specs(bad, 0, 16)
            raise AssertionError(bad)
        except ValueError:
            pass
    fake = {("AI-MO/aimo-validation-aime", None, "train"): [dict(id=60, problem="P60", answer=204, url="https://aops/2024_AIME_I/1"),
                                                            dict(id=1, problem="P1", answer="7", url="https://aops/2022_AIME_II/3")],
            ("math-ai/aime25", None, "test"): [dict(id=0, problem="Q0", answer="70")],
            ("RUC-AIBOX/OlymMATH", "en-easy", "test"): [dict(unique_id="e1", problem="O1", answer="\\frac{1}{2}", subject="Algebra")]}
    mod = types.ModuleType("datasets")
    mod.load_dataset = lambda name, cfg=None, split=None: fake[(name, cfg, split)]
    saved = sys.modules.get("datasets")
    sys.modules["datasets"] = mod
    try:
        a = gen.load_problems("aime_all", 0, 0)
        assert [x["id"] for x in a] == ["aime2024-60", "aime2022-1", "aime2025-0"] and a[0]["answer"] == "204"
        assert {x["subject"] for x in a} == {"aime2024", "aime2022", "aime2025"}
        o = gen.load_problems("olymmath_easy", 0, 0)
        assert o == [dict(id="olymmath-e1", problem="O1", answer="\\frac{1}{2}", subject="Algebra", level="olymmath-easy")]
        assert len(gen.load_problems("aime_all", 2, 0)) == 2
    finally:
        if saved is None:
            del sys.modules["datasets"]
        else:
            sys.modules["datasets"] = saved


def test_checkpoint_helpers():
    import json, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "base__no_code__verltool_torl.jsonl"
        rec = lambda pid, k: dict(problem_id=pid, sample_idx=k)  # noqa: E731
        gen.write_jsonl(gen.partial_path(path), [rec("p0", 0), rec("p0", 1), rec("p1", 0)])
        with open(gen.partial_path(path), "a") as f:
            f.write('{"problem_id": "p2", "sample_')          # torn line from a crash
        gen.write_jsonl(gen.partial_path(path), [rec("p1", 1)])
        kept, done = gen.read_partial(path, n=2)
        assert done == {"p0", "p1"} and len(kept) == 4
        items = [dict(id=f"p{i}") for i in range(5)]
        assert [[i for i, _ in c] for c in gen.plan_chunks(items, done, 2)] == [[2, 3], [4]]
        gen.write_jsonl_atomic(path, kept)
        assert len(open(path).read().splitlines()) == 4 and gen.meta_path(path).name.endswith(".meta.json")
        assert json.loads(open(path).readline())["problem_id"] in ("p0", "p1")


def test_audit_fixes_in_helpers():
    assert gen.last_boxed(r"so \boxed{116} then \boxed{11") == "116"           # cut-off final box: earlier one counts
    out, err = gen.execute("import sys; sys.stdout.buffer.write(b'ok \\xff'); sys.stdout.flush()")
    assert out.startswith("ok") and not err                                   # non-UTF-8 output no longer crashes
    seeds = {gen.round_seed(0, p, k, r) for p in range(30) for k in range(128) for r in range(1, 4)}
    assert len(seeds) == 30 * 128 * 3                                         # no collisions, even for n > 64
    # a python block closed and then end-of-sequence is a call like any other
    s = gen.Sample(0, 0, "P")
    inj = gen.advance(s, "Check:\n```python\nprint(3)\n```", 6, "stop", None, "env_denied", 4)
    assert inj is not None and gen.DENIAL in inj and s.n_eos_code == 1 and s.n_denied == 1 and not s.done
    s2 = gen.Sample(0, 0, "P")
    assert gen.advance(s2, "plain ending", 2, "stop", None, "env_denied", 4) is None and s2.finish == "eos"


def test_settings_fingerprint_and_arm_done():
    import json, tempfile
    a = SimpleNamespace(model="m", template="verltool_torl", temperature=0.6, top_p=0.95, max_tokens=3000,
                        max_model_len=4096, max_calls=4, exec_timeout=10.0, seed=0)
    fp = gen.settings_fp(a, 16)
    assert fp == gen.settings_fp(a, 16) and fp != gen.settings_fp(a, 64)
    assert fp != gen.settings_fp(SimpleNamespace(**{**vars(a), "temperature": 1.0}), 16)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.jsonl"
        gen.write_jsonl(gen.partial_path(path), [dict(problem_id="p0", sample_idx=k, fp="OTHER") for k in range(2)]
                        + [dict(problem_id="p1", sample_idx=k, fp=fp) for k in range(2)])
        kept, done = gen.read_partial(path, 2, fp)
        assert done == {"p1"}                                                  # rows from other settings dropped
        meta = gen.meta_path(path)
        meta.write_text(json.dumps(dict(n=4, n_problems=6, n_problems_requested=6)))
        assert not gen.arm_done(meta, 16, 0) and not gen.arm_done(meta, 4, 0)   # smoke-sized arm is not done
        meta.write_text(json.dumps(dict(n=64, n_problems=120, n_problems_requested=0)))
        assert gen.arm_done(meta, 16, 0) and gen.arm_done(meta, 64, 0) and not gen.arm_done(meta, 128, 0)
        path.write_text("{}\n")
        gen.retire(path)
        assert not path.exists() and not meta.exists() and Path(str(path) + ".replaced").exists()

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
