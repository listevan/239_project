"""Laptop tests for gen.py helpers (no GPU, no vllm). Run: python -m pytest -q tests
or simply: python tests/test_gen.py"""
import sys
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gen  # noqa: E402


def args(**kw):
    base = dict(max_tokens=400, max_model_len=4096, max_calls=2, exec_timeout=5.0, seed=0,
                exec_workers=2, max_output_chars=200)
    base.update(kw)
    return SimpleNamespace(**base)


def test_last_boxed():
    assert gen.last_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert gen.last_boxed(r"\boxed{a} then \boxed{(3, \frac{\pi}{2})}") == r"(3, \frac{\pi}{2})"
    assert gen.last_boxed("no answer") is None


def test_code_block_and_fence():
    t = "text\n```python\nprint(1+1)\n```\n"
    assert gen.last_code_block(t) == "print(1+1)\n"
    assert gen.last_code_block("```\nnot python\n```\n") is None
    assert "print" not in gen.strip_fenced(t)


def test_execute():
    out, err = gen.execute("print(6*7)")
    assert out == "42" and not err
    out, err = gen.execute("1/0")
    assert err and "ZeroDivisionError" in out
    out, err = gen.execute("import time; time.sleep(3)", timeout=0.5)
    assert err and "Timeout" in out


def test_grader():
    grade, name = gen.make_grader()
    print("grader:", name)
    assert grade(r"The answer is \boxed{\frac{1}{2}}.", r"\frac12")
    assert grade(r"\boxed{0.5}", r"\frac{1}{2}")
    assert grade(r"so \boxed{204}", r"\boxed{204}")
    assert not grade(r"\boxed{205}", "204")
    assert not grade("no box at all 204", "204") or name == "math_verify"  # math_verify may accept bare numbers
    if name == "math_verify":
        assert grade(r"\boxed{\left( 3, \frac{\pi}{2} \right)}", r"\left( 3, \frac{\pi}{2} \right)")


def scripted_generate(script):
    """script: {context_suffix_key: (text, finish, stop)} chosen by the number of
    ```output blocks already in the context (i.e. the round index)."""
    def generate(prompts, stops, budgets, seeds=None, biases=None):
        outs = []
        for p in prompts:
            k = p.count("```output")
            text, fin, stop = script[min(k, len(script) - 1)]
            outs.append((text, len(text.split()), fin, stop))
        return outs
    return generate


def run(cond, script, a):
    s = gen.Sample(0, 0, "PROMPT\n")
    pool = ThreadPoolExecutor(max_workers=2)
    text, fin, stop = script[0]
    gen.splice_round([(s, text, len(text.split()), fin, stop)], cond, a, pool)
    gen.run_tool_loop(scripted_generate(script), lambda t: len(t.split()), [s], cond, a, pool)
    pool.shutdown()
    return s


def test_tool_on_executes_and_finishes():
    script = {0: ("Let me compute.\n```python\nprint(17*23)\n```\n", "stop", gen.STOP_FENCE),
              1: ("So the answer is \\boxed{391}.", "stop", None)}
    s = run("tool_on", script, args())
    assert s.done and s.finish == "eos" and not s.truncated
    assert s.n_calls == 1 and s.n_denied == 0 and s.n_injected == 1
    assert "```output\n391\n```" in s.gen and s.gen.endswith("\\boxed{391}.")


def test_env_denied_injects_message():
    script = {0: ("```python\nprint(2)\n```\n", "stop", gen.STOP_FENCE),
              1: ("fine, 1+1 = 2 so \\boxed{2}", "stop", None)}
    s = run("env_denied", script, args())
    assert s.n_calls == 1 and s.n_denied == 1
    assert gen.DENIAL in s.gen and "```output\n" + gen.DENIAL in s.gen


def test_prompt_off_intercepts_code():
    script = {0: ("I will just compute.\n```python\nprint(2+2)\n```\n", "stop", gen.STOP_FENCE),
              1: ("2 + 2 = 4 so \\boxed{4}", "stop", None)}
    s = run("prompt_off", script, args())
    assert s.n_calls == 1 and s.n_denied == 1 and s.n_exec_err == 0
    assert gen.DENIAL_UNAVAILABLE in s.gen and "```output\n4" not in s.gen
    assert "Do not write code" in gen.build_prompt("qwen_math_nocode", "Q")


def test_custom_denial_text():
    script = {0: ("```python\nprint(1)\n```\n", "stop", gen.STOP_FENCE), 1: ("\\boxed{1}", "stop", None)}
    s = run("env_denied", script, args(denial_text="NOPE"))
    assert "```output\nNOPE\n```" in s.gen


def test_model_opens_output_block_itself():
    script = {0: ("```python\nprint(5)\n```\n```output", "stop", gen.STOP_OUTPUT),
              1: ("\nThus \\boxed{5}", "stop", None)}
    s = run("tool_on", script, args())
    assert "```output\n5\n```\n" in s.gen and s.n_injected == 1


def test_budget_cap_then_denial():
    script = {0: ("```python\nprint(1)\n```\n", "stop", gen.STOP_FENCE),
              1: ("```python\nprint(2)\n```\n", "stop", gen.STOP_FENCE),
              2: ("```python\nprint(3)\n```\n", "stop", gen.STOP_FENCE),
              3: ("done \\boxed{3}", "stop", None)}
    s = run("tool_on", script, args(max_calls=2))
    assert s.n_calls == 3 and s.n_denied == 1 and s.n_exec_err == 0
    assert s.gen.count("```output") == 3 and gen.DENIAL_BUDGET in s.gen


def test_fabricated_output_and_truncation():
    script = {0: ("```output\n99\n```\n", "stop", gen.STOP_FENCE),   # fabricated, no code
              1: ("and on and on", "length", None)}
    s = run("env_denied", script, args())
    assert s.truncated and s.n_injected == 0 and s.n_calls == 0
    rec = gen.to_record(s, dict(id="x", answer="1"), "env_denied", "torl", "m", "tool_rl", lambda t, g: False)
    assert rec["fake_outputs"] == 1 and rec["truncated"] and not rec["boxed_present"]


def test_round_cap_forces_final_round():
    script = {k: ("```\nplain fence\n```\n", "stop", gen.STOP_FENCE) for k in range(20)}
    a = args(max_calls=1)
    s = gen.Sample(0, 0, "P")
    pool = ThreadPoolExecutor(max_workers=1)
    calls = {"n": 0}

    def generate(prompts, stops, budgets, seeds=None, biases=None):
        calls["n"] += 1
        if gen.STOP_FENCE not in stops:   # final round: no tool stops
            return [("final \\boxed{1}", 3, "stop", None) for _ in prompts]
        return [("```\nplain fence\n```\n", 5, "stop", gen.STOP_FENCE) for _ in prompts]
    gen.run_tool_loop(generate, lambda t: 1, [s], "env_denied", a, pool)
    pool.shutdown()
    assert s.done and s.gen.endswith("\\boxed{1}") and calls["n"] <= 2 * a.max_calls + 8


def test_arith_record():
    s = gen.Sample(0, 0, "P")
    s.gen = "We have 12 * 13 = 156 and 17 * 23 = 381.\n```python\nprint(1+1 == 3)\n```\n\\boxed{156}"
    s.done = True
    rec = gen.to_record(s, dict(id="x", answer="156"), "prompt_off", "qwen_math_cot", "m", "base", lambda t, g: True)
    assert rec["arith_n"] == 2 and rec["arith_wrong"] == 1 and rec["code_emitted"]


def test_prompt_build():
    p = gen.build_prompt("torl", "What is 1+1?")
    assert p.endswith("What is 1+1?\nAssistant:") and "\\boxed{}" in p
    assert gen.build_prompt("qwen_math_cot", "Q").startswith("<|im_start|>system")


def test_partial_resume_roundtrip():
    import json, os, tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "base__env_denied__qwen_math_tir.jsonl"
        rec = lambda pid, sidx, ok: dict(problem_id=pid, sample_idx=sidx, correct=ok)  # noqa: E731
        # chunk 1 complete (p0, p1), chunk 2 torn: p2 has 1 of 2 samples and a cut-off line
        gen.write_jsonl(gen.partial_path(path), [rec("p0", 0, 1), rec("p0", 1, 0), rec("p1", 0, 1), rec("p1", 1, 1)])
        gen.write_jsonl(gen.partial_path(path), [rec("p2", 0, 1)])
        with open(gen.partial_path(path), "a") as f:
            f.write('{"problem_id": "p3", "sample_')
        kept, done = gen.read_partial(path, n=2)
        assert done == {"p0", "p1"} and len(kept) == 4
        items = [dict(id=f"p{i}") for i in range(5)]
        chunks = gen.plan_chunks(items, done, chunk=2)
        assert [[i for i, _ in c] for c in chunks] == [[2, 3], [4]]  # global indices survive
        assert len(gen.plan_chunks(items, done, chunk=0)) == 1
        assert gen.plan_chunks(items, {"p0", "p1", "p2", "p3", "p4"}, chunk=2) == []
        # redo of p2 appends a full set; last write per sample_idx wins, no duplicates
        gen.write_jsonl(gen.partial_path(path), [rec("p2", 0, 0), rec("p2", 1, 0)])
        kept, done = gen.read_partial(path, n=2)
        assert done == {"p0", "p1", "p2"} and len(kept) == 6
        assert [r["correct"] for r in kept if r["problem_id"] == "p2"] == [0, 0]
        gen.write_jsonl_atomic(path, kept)
        assert len(open(path).read().splitlines()) == 6 and not os.path.exists(str(path) + ".tmp")
        assert gen.meta_path(path).name == "base__env_denied__qwen_math_tir.meta.json"
        assert not gen.partial_path(path).name.endswith(".jsonl")  # analyze.load_results globs *.jsonl

def test_to_record_ungraded():
    s = gen.Sample(0, 1, "p"); s.gen = r"so 2 + 2 = 5 and \boxed{156}"; s.done = True
    rec = gen.to_record(s, dict(id="x", answer="156"), "prompt_off", "qwen_math_cot", "m", "base")
    assert rec["correct"] is None and rec["arith_n"] is None and rec["arith_wrong"] is None
    assert rec["boxed"] == "156" and rec["boxed_present"] and rec["gold"] == "156" and rec["text"] == s.gen

def test_dataset_specs_and_gsm_gold():
    assert gen.parse_dataset_specs("math500", 0, 16) == [("math500", 0, 16)]
    assert gen.parse_dataset_specs("aime_all::64,olymmath_hard:10", 0, 16) == [("aime_all", 0, 64), ("olymmath_hard", 10, 16)]
    assert gen.parse_dataset_specs("math500:20:4, aime24::64,gsm_hard:300,math500_l5", 0, 16) == \
        [("math500", 20, 4), ("aime24", 0, 64), ("gsm_hard", 300, 16), ("math500_l5", 0, 16)]
    for bad in ("nope", "", "math500:x"):
        try:
            gen.parse_dataset_specs(bad, 0, 16)
            raise AssertionError(bad)
        except ValueError:
            pass
    assert gen.gsm_gold(-9867630.0) == "-9867630" and gen.gsm_gold(3431580.0) == "3431580" and gen.gsm_gold(7) == "7"
    assert gen.gsm_gold(2.5) is None and gen.gsm_gold(float("nan")) is None and gen.gsm_gold("x") is None
    assert gen.gsm_gold(1e16) is None

def test_fence_token_ids():
    class Tok:
        def get_vocab(self):
            return {"a": 0, "```": 1, "``": 2, "`python": 3, "Ġ`": 4, "x": 5, "```Ċ": 6}
    assert gen.fence_token_ids(Tok()) == [1, 2, 3, 4, 6]
    assert gen.fence_token_ids(object()) == []
    assert "prose_only" in gen.CONDITIONS and "prose_only" in gen.LOOP_CONDITIONS

def test_prose_redirect_cuts_fence_and_bans_after_cap():
    a = args(max_calls=1)
    calls = []

    def generate(prompts, stops, budgets, seeds=None, biases=None):
        calls.append((stops, biases))
        outs = []
        per = stops if isinstance(stops[0], list) else [stops] * len(prompts)
        for p, st in zip(prompts, per):
            if p.count(gen.REDIRECT_TEXT) == 0:
                outs.append(("Think.\n```python\n", 3, "stop", "```python\n"))          # tries to code
            elif "9" not in p and "```\n" in st:
                outs.append(("So 3 * 3 = 9. ```\n", 6, "stop", "```\n"))              # tries again: bare fence
            else:
                outs.append(("So 3 * 3 = 9. Hence \\boxed{9}<|im_end|>", 8, "stop", "<|im_end|>"))
        return outs

    s = gen.Sample(0, 0, "P")
    gen.advance(s, "Think.\n```python\n", 3, "stop", "```python\n", "prose_redirect", a.max_calls)
    assert s.gen == "Think.\n" + gen.REDIRECT_TEXT and s.n_calls == 1 and not s.done
    pool = ThreadPoolExecutor(max_workers=1)
    gen.run_tool_loop(generate, lambda t: len(t.split()), [s], "prose_redirect", a, pool, fence_bias={7: -100.0})
    pool.shutdown()
    assert s.finish == "eos" and "```" not in s.gen and gen.last_boxed(s.gen) == "9"
    # after max_calls=1 redirects every later round had plain text stops and the backtick ban
    assert all(st == [gen.TEXT_STOPS] and bi == [{7: -100.0}] for st, bi in calls)
    assert s.gen.count(gen.REDIRECT_TEXT) == 1   # the bare fence in round 2 was not a redirect: the ban round has no fence stops
    # a fresh sample with a higher cap gets fence stops and no ban while redirects remain
    calls.clear()
    s2 = gen.Sample(0, 1, "P")
    gen.advance(s2, "Think.\n```python\n", 3, "stop", "```python\n", "prose_redirect", 4)
    a4 = args(max_calls=4)
    pool = ThreadPoolExecutor(max_workers=1)
    gen.run_tool_loop(generate, lambda t: len(t.split()), [s2], "prose_redirect", a4, pool, fence_bias={7: -100.0})
    pool.shutdown()
    assert calls[0] == ([gen.TEXT_STOPS + list(gen.FENCE_OPEN_STOPS)], [None]) and s2.n_calls == 2
    assert "```" not in s2.gen and s2.gen.count(gen.REDIRECT_TEXT) == 2 and gen.last_boxed(s2.gen) == "9"


def test_assist_denied_uses_its_own_message():
    s = gen.Sample(0, 0, "P")
    gen.advance(s, "```python\nprint(2+2)\n```\n", 5, "stop", "```\n", "assist_denied", 4, denial="custom")
    assert gen.DENIAL_ASSIST in s.gen and "custom" not in s.gen and s.n_denied == 1
    s = gen.Sample(0, 0, "P")
    gen.advance(s, "```python\nprint(2+2)\n```\n", 5, "stop", "```\n", "env_denied", 4, denial="custom")
    assert "custom" in s.gen and gen.DENIAL_ASSIST not in s.gen

def test_verltool_template_and_boxed_grading():
    p = gen.build_prompt("verltool_torl", "What is 1+1?")
    assert p.startswith("<|im_start|>system\nA conversation between User and Assistant.")
    assert "\\boxed{}.:\n<|im_end|>\n<|im_start|>user\nWhat is 1+1?<|im_end|>\n<|im_start|>assistant\n" in p
    grade, _ = gen.make_grader()
    assert gen.grade_boxed(grade, r"so \boxed{2}", "2") is True
    assert gen.grade_boxed(grade, "the answer is 2", "2") is False
    s = gen.Sample(0, 0, "P"); s.gen = r"1+1 = 2 so \boxed{2}"; s.done = True
    rec = gen.to_record(s, dict(id="x", answer="2"), "prompt_off", "t", "m", "base", grade)
    assert rec["correct"] and rec["correct_boxed"]

def test_code_token_ids_bans_program_tokens_only():
    vocab = {"`": 0, " ```": 1, "Ġimport": 2, "import": 3, "important": 4, "importância": 5, "print": 6, "Ġprint(": 7,
             "#": 8, "###": 9, "Ġ==": 10, "=": 11, "lambda": 12, "Ġdef": 13, "Ġnp": 14, "Ġrange": 15, "Ġsympy": 16,
             "Ġfor": 17, "Ġreturn": 18, "Ġ//": 19, "Ġ+=": 20, "elif": 21, "Ġitertools": 22, "Ġnumpy": 23, "Ġsymp": 24}

    class Tok:
        def get_vocab(self):
            return vocab

        def convert_tokens_to_string(self, toks):
            return "".join(toks).replace("Ġ", " ")

    ids = gen.code_token_ids(Tok())
    banned = {k for k, i in vocab.items() if i in ids}
    assert banned == {"`", " ```", "Ġimport", "import", "print", "Ġprint(", "#", "Ġ==", "Ġsympy", "Ġ//", "Ġ+=", "elif",
                      "Ġitertools", "Ġnumpy", "Ġsymp"}
    # math prose must stay writable: \lambda, digits named "def", np (binomial mean), range, for, return, ###
    assert not banned & {"important", "importância", "###", "=", "lambda", "Ġdef", "Ġnp", "Ġrange", "Ġfor", "Ġreturn"}
    assert "no_code" in gen.CONDITIONS and "no_code" not in gen.SHARED_PROMPT_CONDITIONS
    assert "no_code" not in gen.REDIRECT_STOPS        # nothing is inserted under no_code

if __name__ == "__main__":
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
