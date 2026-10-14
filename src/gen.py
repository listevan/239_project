#!/usr/bin/env python
"""Sample one model under one or more tool conditions with vLLM and write JSONL.

Every model gets the same grid. Conditions:
  prompt_off   shared no-code prompt ("do not write code"). If the model writes a
               code block anyway, the harness intercepts it and injects the denial
               message, so nothing is ever executed and no output is fabricated.
               n_calls on these rows = non-compliance count.
  env_denied   the model's tool prompt (asks for code). Every code block is
               intercepted and answered with the denial message.
  tool_on      the model's tool prompt. Code is executed with `python -I` in a
               temp dir (NOT a sandbox: fine for Colab, not for a shared box).
               Calls past --max-calls get the budget message instead.
  plain_cot    optional. Shared plain CoT prompt, one pass, no interception at
               all: the condition prior work (CIR) used. Code the model writes is
               just text and it will fabricate its own ```output block.

Only `main()` imports vllm, so the helpers can be unit-tested on a laptop.

Example
  python gen.py --model GAIR/ToRL-1.5B --tag tool_rl \
      --conditions prompt_off,env_denied,tool_on --native-template torl \
      --dataset math500 --n-problems 100 --n 16 --max-tokens 3000 \
      --max-model-len 4096 --out results/math-1.5b
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arith_check  # noqa: E402

# --------------------------------------------------------------------------- prompts
# Verified against the sources on 2026-10-13:
#   qwen_math_cot  Oat-Zero model card (sail/Qwen2.5-Math-*-Oat-Zero), also the
#                  Qwen2.5-Math CoT prompt. Shared tool-off prompt for every model.
#   qwen_math_tir  Qwen2.5-Math TIR prompt (same words ToRL uses).
#   torl           ToRL paper Fig. 3, verbatim, raw User/Assistant format.
#   cir            CIR paper (2505.24480) Fig. 5 text, verbatim. The paper does not
#                  say whether it sits in the system or user turn; system assumed.
TEMPLATES = {
    "qwen_math_cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "qwen_math_nocode": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}. Do not write code: solve the problem in natural language and "
        "arithmetic only.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "qwen_math_tir": (
        "<|im_start|>system\nPlease integrate natural language reasoning with programs to "
        "solve the problem above, and put your final answer within \\boxed{}.<|im_end|>\n"
        "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "torl": (
        "A conversation between User and Assistant. The user asks a question, and the "
        "Assistant solves it.\nUser: Please integrate natural language reasoning with "
        "programs to solve the problem above, and put your final answer within \\boxed{}.\n"
        "{q}\nAssistant:"
    ),
    "cir": (
        "<|im_start|>system\nPlease solve the following problem step by step. During your "
        "reasoning process, if needed, you can choose to write python code to enhance your "
        "reasoning. The code executor will run your code and provide the execution results "
        "back to you to support your reasoning process. Please put the final answer within "
        "\\boxed{}.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
}

# Injected in place of execution results. AdaTIR wording for the over-budget case;
# an honest "no interpreter" wording for prompt_off / env_denied (override: --denial-text).
DENIAL_BUDGET = ("Tool call budget exceeded; the code was not executed. "
                 "Please continue reasoning and give the final answer.")
DENIAL_UNAVAILABLE = ("No interpreter is available; the code was not executed. "
                      "Continue reasoning without running code and give the final answer.")
DENIAL = DENIAL_UNAVAILABLE

STOP_FENCE = "```\n"       # end of any fenced block
STOP_OUTPUT = "```output"  # model opens the output block itself
TEXT_STOPS = ["<|im_end|>", "<|endoftext|>", "\nUser:"]

CODE_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)
FENCED_RE = re.compile(r"```.*?```", re.S)
CONDITIONS = ("prompt_off", "env_denied", "tool_on", "plain_cot")
LOOP_CONDITIONS = ("prompt_off", "env_denied", "tool_on")


def build_prompt(template: str, question: str) -> str:
    return TEMPLATES[template].replace("{q}", question)


def last_code_block(text: str):
    """Last closed ```python block in `text`, or None."""
    blocks = CODE_RE.findall(text)
    return blocks[-1] if blocks else None


def last_boxed(text: str):
    """Content of the last \\boxed{...} with balanced braces, or None."""
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def strip_fenced(text: str) -> str:
    return FENCED_RE.sub(" ", text)


# --------------------------------------------------------------------------- grading
def make_grader():
    try:
        from math_verify import parse, verify  # type: ignore

        def grade(response: str, gold: str) -> bool:
            try:
                g = parse(gold if "\\boxed" in gold else f"${gold}$")
                p = parse(response)
                return bool(verify(g, p))
            except Exception:
                return False
        return grade, "math_verify"
    except ImportError:
        def norm(s):
            return re.sub(r"\s+|\\left|\\right|\\!|\\,|\$", "", s or "").replace("dfrac", "frac")

        def grade(response: str, gold: str) -> bool:
            b = last_boxed(response)
            return b is not None and norm(b) == norm(last_boxed(gold) or gold)
        return grade, "string"


# --------------------------------------------------------------------------- data
def load_problems(name: str, n_problems: int, seed: int):
    from datasets import load_dataset  # type: ignore
    if name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [dict(id=r["unique_id"], problem=r["problem"], answer=r["answer"],
                      subject=r["subject"], level=str(r["level"])) for r in ds]
    elif name == "aime24":
        ds = load_dataset("math-ai/aime24", split="test")
        items = [dict(id=f"aime24-{r['id']}", problem=r["problem"],
                      answer=last_boxed(r["solution"]) or r["solution"],
                      subject="aime24", level="aime") for r in ds]
    elif name == "aime25":
        ds = load_dataset("math-ai/aime25", split="test")
        items = [dict(id=f"aime25-{r['id']}", problem=r["problem"], answer=str(r["answer"]),
                      subject="aime25", level="aime") for r in ds]
    else:
        raise ValueError(f"unknown dataset {name}")
    if n_problems and n_problems < len(items):
        idx = sorted(random.Random(seed).sample(range(len(items)), n_problems))
        items = [items[i] for i in idx]
    return items


# --------------------------------------------------------------------------- executor
def execute(code: str, timeout: float = 10.0, max_chars: int = 2000):
    """Run `code` with python -I in a temp dir. Returns (output_text, errored)."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "snippet.py"
        path.write_text(code)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True,
                               stdin=subprocess.DEVNULL, text=True, timeout=timeout)
            out, err = r.stdout, r.returncode != 0
            if err:
                tail = r.stderr.strip().splitlines()
                out = (out + "\n" + tail[-1]) if tail else out
        except subprocess.TimeoutExpired:
            out, err = f"Timeout: code ran longer than {timeout:.0f}s", True
    out = out.strip()
    if len(out) > max_chars:
        out = out[:max_chars] + "...[truncated]"
    return out, err


# --------------------------------------------------------------------------- tool loop
class Sample:
    __slots__ = ("pidx", "sidx", "prompt", "gen", "gen_tokens", "n_calls", "n_denied",
                 "n_exec_err", "n_injected", "rounds", "done", "truncated", "finish")

    def __init__(self, pidx, sidx, prompt):
        self.pidx, self.sidx, self.prompt = pidx, sidx, prompt
        self.gen = ""
        self.gen_tokens = 0
        self.n_calls = self.n_denied = self.n_exec_err = self.n_injected = self.rounds = 0
        self.done = self.truncated = False
        self.finish = None

    @property
    def context(self):
        return self.prompt + self.gen


def advance(s: Sample, text: str, n_tokens: int, finish_reason, stop_reason, cond: str,
            max_calls: int, executor=execute, denial=None):
    """Apply one generation chunk to sample `s`. Returns the text to inject (or None)."""
    s.gen += text
    s.gen_tokens += n_tokens
    s.rounds += 1
    if finish_reason == "length":
        s.done, s.truncated, s.finish = True, True, "length"
        return None
    if stop_reason not in (STOP_FENCE, STOP_OUTPUT):
        s.done, s.finish = True, "eos"
        return None
    code = last_code_block(text)
    if code is None:
        return None  # a non-python fence or a fabricated ```output: keep generating
    s.n_calls += 1
    if cond == "tool_on" and s.n_calls <= max_calls:
        result, err = executor(code)
        s.n_exec_err += int(err)
    else:
        result = DENIAL_BUDGET if cond == "tool_on" else (denial or DENIAL_UNAVAILABLE)
        s.n_denied += 1
    inj = ("\n" + result + "\n```\n") if stop_reason == STOP_OUTPUT else ("```output\n" + result + "\n```\n")
    s.gen += inj
    s.n_injected += 1
    return inj


TOOL_STOPS = (STOP_FENCE, STOP_OUTPUT)


def splice_round(pending, cond, args, pool):
    """pending: list of (sample, text, n_tokens, finish_reason, stop_reason) for one round.
    Executes tool_on code calls in parallel, then applies `advance` to every sample."""
    futures = {}
    for s, text, ntok, fin, stop in pending:
        if (cond == "tool_on" and stop in TOOL_STOPS and fin != "length"
                and s.n_calls + 1 <= args.max_calls):
            code = last_code_block(text)
            if code is not None:
                futures[id(s)] = pool.submit(execute, code, args.exec_timeout, args.max_output_chars)
    t0 = time.time()
    if futures:
        print(f"  executing {len(futures)} code blocks ({args.exec_workers} workers, "
              f"{args.exec_timeout:.0f}s timeout each)...", flush=True)
    for s, text, ntok, fin, stop in pending:
        fut = futures.get(id(s))
        if fut is not None:
            ex = lambda code, f=fut: f.result()  # noqa: E731
        else:
            ex = lambda code: execute(code, args.exec_timeout, args.max_output_chars)  # noqa: E731
        advance(s, text, ntok, fin, stop, cond, args.max_calls, executor=ex,
                denial=getattr(args, "denial_text", None))
    if futures:
        n_err = sum(1 for s, *_ in pending if id(s) in futures and futures[id(s)].result()[1])
        print(f"  executed in {time.time() - t0:.0f}s, {n_err} errored or timed out", flush=True)


def run_tool_loop(generate, tokenize_len, samples, cond, args, pool):
    """Rounds 2..N. generate(prompts, stops, budgets, seeds) -> [(text, n_tokens, finish, stop_reason)]."""
    max_rounds = 2 * args.max_calls + 6
    rnd, t0 = 1, time.time()
    while True:
        rnd += 1
        keep, budgets = [], []
        for s in samples:
            if s.done:
                continue
            remaining = min(args.max_tokens - s.gen_tokens,
                            args.max_model_len - tokenize_len(s.context) - 1)
            if remaining < 16:
                s.done, s.truncated, s.finish = True, True, "length"
            else:
                keep.append(s)
                budgets.append(remaining)
        if not keep:
            break
        final = any(s.rounds >= max_rounds for s in keep)
        stops = TEXT_STOPS if final else TEXT_STOPS + list(TOOL_STOPS)
        print(f"  round {rnd}: {len(keep)} of {len(samples)} samples still generating"
              f"{' (final round, no more tool stops)' if final else ''}, {time.time() - t0:.0f}s into the loop",
              flush=True)
        seeds = [(args.seed * 1_000_003 + s.pidx * 4096 + s.sidx * 64 + s.rounds) % (2 ** 31) for s in keep]
        outs = generate([s.context for s in keep], stops, budgets, seeds)
        splice_round([(s,) + tuple(o) for s, o in zip(keep, outs)], cond, args, pool)


# --------------------------------------------------------------------------- records
def to_record(s: Sample, item, cond, template, model, tag, grade=None):
    """One JSONL row. With grade=None (gen.py --no-grade) `correct`, `arith_n` and `arith_wrong` are
    left null for src/grade.py to fill on another machine; everything else is generation-time data."""
    text = s.gen
    fake_outputs = max(0, text.count(STOP_OUTPUT) - s.n_injected)
    if grade is None:
        correct = a_n = a_wrong = None
    else:
        correct = bool(grade(text, item["answer"]))
        a_n, a_wrong, _ = arith_check.check(strip_fenced(text))
    return dict(
        model=model, tag=tag, condition=cond, template=template,
        problem_id=item["id"], subject=item.get("subject"), level=item.get("level"),
        sample_idx=s.sidx, correct=correct, gold=item["answer"],
        boxed=last_boxed(text), boxed_present=last_boxed(text) is not None,
        truncated=s.truncated, finish=s.finish, gen_tokens=s.gen_tokens, rounds=s.rounds,
        code_emitted="```python" in text or "```py" in text, fake_outputs=fake_outputs,
        n_calls=s.n_calls, n_denied=s.n_denied, n_exec_err=s.n_exec_err,
        arith_n=a_n, arith_wrong=a_wrong, text=text,
    )


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default=None)
    p.add_argument("--tag", required=True, help="role label: base | tool_rl | text_rl | ...")
    p.add_argument("--conditions", default="prompt_off")
    p.add_argument("--shared-template", default="qwen_math_nocode",
                   help="no-code prompt used for prompt_off, same for every model")
    p.add_argument("--native-template", default="qwen_math_tir",
                   help="the model's tool prompt, used for env_denied and tool_on")
    p.add_argument("--plain-template", default="qwen_math_cot", help="prompt for plain_cot")
    p.add_argument("--denial-text", default=None,
                   help="override the message injected for prompt_off / env_denied")
    p.add_argument("--dataset", default="math500", choices=["math500", "aime24", "aime25"])
    p.add_argument("--n-problems", type=int, default=0, help="0 = all")
    p.add_argument("--n", type=int, default=16, help="samples per problem")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--allow-long", action="store_true",
                   help="override max_position_embeddings so max-model-len can exceed the config")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-calls", type=int, default=4)
    p.add_argument("--exec-timeout", type=float, default=10.0)
    p.add_argument("--exec-workers", type=int, default=8)
    p.add_argument("--max-output-chars", type=int, default=2000)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--out", required=True, help="results directory")
    p.add_argument("--overwrite", action="store_true", help="redo finished arms and discard partial files")
    p.add_argument("--no-grade", action="store_true",
                   help="write raw trajectories only (correct/arith fields null); grade later with src/grade.py")
    p.add_argument("--chunk", type=int, default=100,
                   help="problems per checkpoint; records are appended to <arm>.jsonl.partial after each "
                        "chunk and a restart resumes from there (0 = whole arm in one go)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- checkpointing
# An arm is "done" when its .meta.json exists (written last). While an arm runs, each chunk of
# problems is appended to <arm>.jsonl.partial, so a disconnect loses at most one chunk. The
# final .jsonl is written atomically (tmp + rename) and the partial file is removed.
def partial_path(path: Path) -> Path:
    return Path(str(path) + ".partial")


def meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def read_partial(path: Path, n: int):
    """Records already on disk for this arm. Keeps only problems with all n sample_idx present
    (last write wins), so a torn chunk is simply redone. Returns (records, finished problem ids)."""
    pp = partial_path(path)
    if not pp.exists():
        return [], set()
    by_pid = {}
    with open(pp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from a crash mid-write
            by_pid.setdefault(r["problem_id"], {})[r["sample_idx"]] = r
    done = {pid for pid, d in by_pid.items() if len(d) == n}
    recs = [by_pid[pid][k] for pid in done for k in sorted(by_pid[pid])]
    return recs, done


def plan_chunks(items, done_ids, chunk):
    """Remaining (global_index, item) pairs, grouped into chunks. Global indices are kept so
    per-sample seeds do not depend on chunking or on what was already finished."""
    remaining = [(i, it) for i, it in enumerate(items) if it["id"] not in done_ids]
    size = chunk if chunk > 0 else max(1, len(remaining))
    return [remaining[s:s + size] for s in range(0, len(remaining), size)]


def write_jsonl(path: Path, recs, mode="a"):
    # A crash mid-write leaves a torn last line with no newline; start the append on a fresh line
    # so the torn line stays one (ignorable) bad record instead of swallowing the next good one.
    if mode == "a" and path.exists() and path.stat().st_size > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b"\n"
        if needs_nl:
            with open(path, "a") as f:
                f.write("\n")
    with open(path, mode) as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # FUSE mounts such as Google Drive may not support fsync


def write_jsonl_atomic(path: Path, recs):
    tmp = Path(str(path) + ".tmp")
    write_jsonl(tmp, recs, mode="w")
    os.replace(tmp, path)


def main(argv=None):
    args = parse_args(argv)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conds:
        if c not in CONDITIONS:
            sys.exit(f"unknown condition {c}; choose from {CONDITIONS}")
    out_dir = Path(args.out) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    def out_path(cond, template):
        return out_dir / f"{args.tag}__{cond}__{template}.jsonl"

    def template_for(cond):
        if cond == "prompt_off":
            return args.shared_template
        if cond == "plain_cot":
            return args.plain_template
        return args.native_template

    todo = [(c, template_for(c)) for c in conds
            if args.overwrite or not meta_path(out_path(c, template_for(c))).exists()]
    if not todo:
        print("all conditions already done, nothing to do (use --overwrite to redo)")
        return
    print(f"gen.py starting: {args.tag} {args.model}; todo={todo}", flush=True)
    print(f"loading {args.dataset} from the HF hub...", flush=True)
    items = load_problems(args.dataset, args.n_problems, args.seed)
    if args.no_grade:
        grade, grader_name = None, "none (grade with src/grade.py)"
    else:
        print("loading grader (math_verify)...", flush=True)
        grade, grader_name = make_grader()
    print(f"{len(items)} problems from {args.dataset}; grader={grader_name}", flush=True)
    print("importing torch + vllm (first time: a minute or two)...", flush=True)

    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # The FlashInfer sampler JIT-builds a kernel on first use (about two minutes on a fresh Colab
    # runtime, every time, since the cache dies with the VM). The torch sampler needs no build.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.allow_long:
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    import torch  # type: ignore
    if not torch.cuda.is_available():
        sys.exit("no CUDA device is visible: this runtime has no GPU. Colab: Runtime > Change runtime type > GPU "
                 "(VS Code: reconnect choosing a GPU runtime), then run again from the first cell.")
    from vllm import LLM, SamplingParams  # type: ignore
    dtype = args.dtype
    if dtype == "auto":
        major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 8
        dtype = "bfloat16" if major >= 8 else "float16"
    kw = dict(model=args.model, dtype=dtype, max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True)
    if args.revision:
        kw["revision"] = args.revision
    if args.allow_long:
        kw["hf_overrides"] = {"max_position_embeddings": args.max_model_len}
    t0 = time.time()
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    print(f"loaded {args.model} ({dtype}) in {time.time() - t0:.0f}s", flush=True)

    def tokenize_len(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def sp(n, max_tokens, stops, seed=None):
        return SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p,
                              seed=args.seed if seed is None else seed,
                              max_tokens=max_tokens, stop=stops, include_stop_str_in_output=True)

    def generate(prompts, stops, budgets, seeds):
        params = [sp(1, b, stops, seed) for b, seed in zip(budgets, seeds)]
        res = llm.generate(prompts, params, use_tqdm=True)
        outs = []
        for r in res:
            o = r.outputs[0]
            outs.append((o.text, len(o.token_ids), o.finish_reason, o.stop_reason))
        return outs

    def run_arm(part, cond, template):
        """Generate every sample for the (global_index, item) pairs in `part`; returns Samples."""
        prompts = [build_prompt(template, it["problem"]) for _, it in part]
        samples = []
        if cond == "plain_cot":
            res = llm.generate(prompts, sp(args.n, args.max_tokens, TEXT_STOPS), use_tqdm=True)
            for (pidx, _), prompt, r in zip(part, prompts, res):
                for sidx, o in enumerate(r.outputs):
                    s = Sample(pidx, sidx, prompt)
                    advance(s, o.text, len(o.token_ids), o.finish_reason, None, cond, 0)
                    if o.finish_reason == "length":
                        s.truncated, s.finish = True, "length"
                    samples.append(s)
            return samples
        res = llm.generate(prompts, sp(args.n, args.max_tokens, TEXT_STOPS + list(TOOL_STOPS)), use_tqdm=True)
        first = []
        for (pidx, _), prompt, r in zip(part, prompts, res):
            for sidx, o in enumerate(r.outputs):
                s = Sample(pidx, sidx, prompt)
                samples.append(s)
                first.append((s, o.text, len(o.token_ids), o.finish_reason, o.stop_reason))
        pool = ThreadPoolExecutor(max_workers=args.exec_workers)
        splice_round(first, cond, args, pool)
        run_tool_loop(generate, tokenize_len, samples, cond, args, pool)
        pool.shutdown(wait=True)
        return samples

    order = {it["id"]: i for i, it in enumerate(items)}
    for cond, template in todo:
        t1 = time.time()
        path = out_path(cond, template)
        if args.overwrite:
            partial_path(path).unlink(missing_ok=True)
        kept, _ = read_partial(path, args.n)
        kept = [r for r in kept if r["problem_id"] in order]
        done_ids = {r["problem_id"] for r in kept}
        chunks = plan_chunks(items, done_ids, args.chunk)
        print(f"== {args.tag} / {cond} / {template}: {len(items)} problems x {args.n} samples"
              + (f"; resuming, {len(done_ids)} problems already on disk" if done_ids else "")
              + f"; {len(chunks)} chunk(s) to run", flush=True)
        new_recs = []
        for ci, part in enumerate(chunks, 1):
            tc = time.time()
            samples = run_arm(part, cond, template)
            tg = time.time()
            if grade is not None:
                print(f"   grading {len(samples)} samples (math_verify, serial)...", flush=True)
            recs = []
            for i, smp in enumerate(samples, 1):
                recs.append(to_record(smp, items[smp.pidx], cond, template, args.model, args.tag, grade))
                if grade is not None and i % 40 == 0:
                    print(f"   graded {i}/{len(samples)}, {time.time() - tg:.0f}s", flush=True)
            print(f"   writing {len(recs)} records to {partial_path(path).name}...", flush=True)
            tw = time.time()
            write_jsonl(partial_path(path), recs)
            new_recs += recs
            acc_txt = (f"pass@1={sum(r['correct'] for r in recs) / max(1, len(recs)):.3f}" if grade is not None
                       else "ungraded")
            print(f"   chunk {ci}/{len(chunks)}: {len(part)} problems, {acc_txt}, "
                  f"{(time.time() - tc) / 60:.1f} min (write {time.time() - tw:.0f}s), checkpointed", flush=True)
        gen_s = time.time() - t1
        recs = kept + new_recs
        recs.sort(key=lambda r: (order[r["problem_id"]], r["sample_idx"]))
        print(f"   writing {path.name} + meta...", flush=True)
        write_jsonl_atomic(path, recs)
        graded = [r["correct"] for r in recs if r.get("correct") is not None]
        acc = sum(graded) / len(graded) if graded and len(graded) == len(recs) else None
        meta = dict(vars(args), condition=cond, template=template, dtype=dtype, grader=grader_name,
                    graded=acc is not None, n_problems=len(items), n_records=len(recs), n_resumed=len(done_ids),
                    pass1=acc, gen_seconds=gen_s, total_gen_tokens=sum(r["gen_tokens"] for r in recs),
                    finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        meta_path(path).write_text(json.dumps(meta, indent=1))
        partial_path(path).unlink(missing_ok=True)
        print(f"   {'pass@1=%.3f' % acc if acc is not None else 'ungraded'}  "
              f"truncated={sum(r['truncated'] for r in recs) / len(recs):.2%}  "
              f"tokens={meta['total_gen_tokens'] / 1e6:.2f}M  {gen_s / 60:.1f} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
