#!/usr/bin/env python
"""Sample one model under the pilot's conditions with vLLM and write ungraded JSONL.

Three checkpoints from one base model, three conditions, one prompt (the RL models'
training prompt) everywhere; only what happens to code differs:

  tool_on     code runs: each closed ```python block is executed with `python -I` in a temp
              dir (10 s timeout, --max-calls budget, then a budget message). Positive control.
  env_denied  the tool is off: each closed ```python block gets the reply "No interpreter is
              available; the code was not executed. ..." and generation continues.
  no_code     code cannot be written: a logit ban on every backtick token and on the
              program-only keywords (docs/no_code_ban.md). Nothing is intercepted or inserted.
  plain       the best case for tool-free reasoning: the plain step-by-step prompt (qwen_math_cot,
              which does not invite code) plus the same ban. Same prompt for every model.

Rows are written with correct = null; src/grade.py grades them on any machine. Only main()
imports vllm and torch, so everything else runs in the laptop tests.

  python src/gen.py --model <hf id> --tag tool_rl --conditions tool_on,env_denied,no_code \
      --dataset aime_all::64,olymmath_easy::64 --out <runs root>/<run id>
"""
import argparse
import hashlib
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

# --------------------------------------------------------------------------- prompt
# The prompt both VerlTool ToRL-recipe checkpoints were trained on (tool and no-tool runs alike),
# read from their released training rollouts (step_records) on 2026-10-15: VerlTool's system
# prompt v1 as it stood in May 2025, Qwen chat format, system content ending in ".:\n".
TEMPLATES = {
    # verified 2026-10-13: the Qwen2.5-Math chain-of-thought prompt (also Oat-Zero's), the Qwen2.5-Math
    # tool-integrated prompt, and ToRL's own raw prompt (ToRL paper Fig. 3, verbatim)
    "qwen_math_cot": (
        "<|im_start|>system\nPlease reason step by step, and put your final answer "
        "within \\boxed{}.<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
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
    "verltool_torl": (
        "<|im_start|>system\nA conversation between User and Assistant. The user asks a question, and the "
        "Assistant solves it. The assistant first thinks about the reasoning process in the mind and then "
        "provides the user with the answer. User: Please integrate natural language reasoning with programs "
        "to solve the problem above, and put your final answer within \\boxed{}.:\n<|im_end|>\n"
        "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    ),
}

CONDITIONS = ("tool_on", "env_denied", "no_code", "plain")
BANNED = ("no_code", "plain")                 # arms run under the code-token ban
COND_TEMPLATE = {"plain": "qwen_math_cot"}    # conditions with a fixed prompt; the rest use --template


def template_for(cond, template):
    return COND_TEMPLATE.get(cond, template)
DENIAL = ("No interpreter is available; the code was not executed. "
          "Continue reasoning without running code and give the final answer.")
DENIAL_BUDGET = ("Tool call budget exceeded; the code was not executed. "
                 "Please continue reasoning and give the final answer.")

STOP_FENCE = "```\n"       # end of any fenced block
STOP_OUTPUT = "```output"  # the model opens the output block itself
TOOL_STOPS = (STOP_FENCE, STOP_OUTPUT)
TEXT_STOPS = ["<|im_end|>", "<|endoftext|>", "\nUser:"]
CODE_RE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.S)
FENCED_RE = re.compile(r"```.*?```", re.S)

# no_code ban, chosen and checked in docs/no_code_ban.md. Words match a whole token (after stripping
# whitespace) or a token continuing with punctuation, never a longer word ("import" yes, "important"
# no); symbols match exactly ("#" yes, the "###" markdown heading no). lambda, def, np, return, range
# stay allowed: math prose needs them (\lambda tokenizes as backslash + "lambda").
CODE_BAN_WORDS = ("print", "import", "elif", "sympy", "symp", "numpy", "itertools")
CODE_BAN_SYMBOLS = ("==", "+=", "-=", "*=", "/=", "!=", "//", "#")
_CODE_WORD_RE = re.compile(r"^(?:%s)(?!\w)" % "|".join(sorted(CODE_BAN_WORDS, key=len, reverse=True)))
BAN_BIAS = -100.0

EXEC_WORKERS = 8
MAX_OUTPUT_CHARS = 2000


def build_prompt(template: str, question: str) -> str:
    return TEMPLATES[template].replace("{q}", question)


def code_token_ids(tok):
    """Ids banned under no_code: every token containing a backtick plus the program-only keywords.
    Matching is on the decoded token text, so byte-level forms (leading space, newline, tab) count."""
    try:
        vocab = tok.get_vocab()
    except AttributeError:
        return []
    out = []
    for raw, i in vocab.items():
        try:
            d = tok.convert_tokens_to_string([raw])
        except Exception:  # noqa: BLE001
            d = tok.decode([i])
        st = d.strip()
        if "`" in d or st in CODE_BAN_SYMBOLS or (st and _CODE_WORD_RE.match(st)):
            out.append(i)
    return sorted(out)


def last_code_block(text: str):
    """Last closed ```python block in `text`, or None."""
    blocks = CODE_RE.findall(text)
    return blocks[-1] if blocks else None


def last_boxed(text: str):
    """Content of the last complete \\boxed{...} (balanced braces), or None. A final box cut off by the
    token cap does not hide an earlier complete one."""
    end = len(text)
    while True:
        i = text.rfind("\\boxed", 0, end)
        if i < 0:
            return None
        j = text.find("{", i)
        if 0 <= j:
            depth = 0
            for k in range(j, len(text)):
                if text[k] == "{":
                    depth += 1
                elif text[k] == "}":
                    depth -= 1
                    if depth == 0:
                        return text[j + 1:k]
        end = i


def strip_fenced(text: str) -> str:
    return FENCED_RE.sub(" ", text)


# --------------------------------------------------------------------------- data
def _load_aime_all():
    """AIME 2022-2025, 120 problems: AI-MO's validation set (2022, 2023, 2024) plus math-ai/aime25."""
    from datasets import load_dataset  # type: ignore
    items = []
    for r in load_dataset("AI-MO/aimo-validation-aime", split="train"):
        year = next((y for y in ("2022", "2023", "2024") if y in str(r.get("url", ""))), "20xx")
        items.append(dict(id=f"aime{year}-{r['id']}", problem=r["problem"], answer=str(r["answer"]).strip(),
                          subject=f"aime{year}", level="aime"))
    for r in load_dataset("math-ai/aime25", split="test"):
        items.append(dict(id=f"aime2025-{r['id']}", problem=r["problem"], answer=str(r["answer"]).strip(),
                          subject="aime2025", level="aime"))
    return items


def _load_olymmath_easy():
    """OlymMATH en-easy (RUC-AIBOX), 100 problems: AIME-level, sourced from printed publications,
    released 2025 (after Qwen2.5-Math). The hard split floors at 1.5B (1.5% pass@1 for a stronger
    1.5B model in the OlymMATH paper), so the easy split is the informative one here."""
    from datasets import load_dataset  # type: ignore
    return [dict(id=f"olymmath-{r['unique_id']}", problem=r["problem"], answer=str(r["answer"]).strip(),
                 subject=r.get("subject"), level="olymmath-easy")
            for r in load_dataset("RUC-AIBOX/OlymMATH", "en-easy", split="test")]


def _load_math500(levels=None):
    """MATH-500 (HuggingFaceH4/MATH-500); levels=(3, 4, 5) keeps the multi-step ones (about 370)."""
    from datasets import load_dataset  # type: ignore
    return [dict(id=r["unique_id"], problem=r["problem"], answer=str(r["answer"]).strip(), subject=r["subject"],
                 level=str(r["level"]))
            for r in load_dataset("HuggingFaceH4/MATH-500", split="test") if levels is None or int(r["level"]) in levels]


def _load_amc23():
    """AMC 2023 (math-ai/amc23), 40 problems, integer answers: competition problems where computation
    helps but insight is still needed (between MATH and AIME)."""
    from datasets import load_dataset  # type: ignore
    return [dict(id=f"amc23-{r['id']}", problem=r["question"], answer=str(r["answer"]).strip(), subject="amc23",
                 level="amc") for r in load_dataset("math-ai/amc23", split="test")]


def _load_amc24():
    """AMC 12 2024, contests A and B (rawsh/2024_AMC12), 45 problems with the answer as a value; the one
    stored only as a letter is skipped. Held in November 2024, after Qwen2.5-Math was released, so the base
    model cannot have seen them."""
    from datasets import load_dataset  # type: ignore
    out = []
    for r in load_dataset("rawsh/2024_AMC12", split="train"):
        ans = str(r["answer"]).strip()
        if re.fullmatch(r"[A-E]", ans):
            continue
        exam = "A" if r["exam"].strip().endswith("A") else "B"
        out.append(dict(id=f"amc24{exam}-{r['problem_number']}", problem=r["problem"], answer=ans,
                        subject=f"amc24{exam}", level="amc"))
    return out


DATASETS = {"amc24": _load_amc24, "amc23": _load_amc23, "aime_all": _load_aime_all, "olymmath_easy": _load_olymmath_easy,
            "math500": _load_math500, "math500_l35": lambda: _load_math500((3, 4, 5))}


def parse_dataset_specs(spec: str, n_problems: int, n: int):
    """'aime_all::64,olymmath_easy:20' -> [(name, n_problems, n_samples), ...]; each item is
    name[:n_problems[:n_samples]], an empty field takes the --n-problems / --n default."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        f = part.split(":")
        if f[0] not in DATASETS:
            raise ValueError(f"unknown dataset {f[0]!r}; choose from {list(DATASETS)}")
        out.append((f[0], int(f[1]) if len(f) > 1 and f[1] else n_problems, int(f[2]) if len(f) > 2 and f[2] else n))
    if not out:
        raise ValueError("no dataset given")
    return out


def load_problems(name: str, n_problems: int, seed: int):
    items = DATASETS[name]()
    if n_problems and n_problems < len(items):
        idx = sorted(random.Random(seed).sample(range(len(items)), n_problems))
        items = [items[i] for i in idx]
    return items


# --------------------------------------------------------------------------- executor
EXEC_MEMORY_BYTES = 4 * 1024 ** 3   # per snippet; the VM also hosts vLLM


def _limit_memory():
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (EXEC_MEMORY_BYTES, EXEC_MEMORY_BYTES))
    except (ImportError, ValueError, OSError):
        pass


def execute(code: str, timeout: float = 10.0, max_chars: int = MAX_OUTPUT_CHARS):
    """Run `code` with python -I in a temp dir (memory capped on Linux). Returns (output_text, errored).
    Output is decoded with errors="replace", so a stray non-UTF-8 byte cannot crash the run."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "snippet.py"
        path.write_text(code)
        try:
            r = subprocess.run([sys.executable, "-I", str(path)], cwd=d, capture_output=True,
                               stdin=subprocess.DEVNULL, text=True, errors="replace", timeout=timeout,
                               preexec_fn=_limit_memory if sys.platform.startswith("linux") else None)
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
                 "n_exec_err", "n_injected", "rounds", "done", "truncated", "finish", "n_eos_code")

    def __init__(self, pidx, sidx, prompt):
        self.pidx, self.sidx, self.prompt = pidx, sidx, prompt
        self.gen = ""
        self.gen_tokens = 0
        self.n_calls = self.n_denied = self.n_exec_err = self.n_injected = self.rounds = self.n_eos_code = 0
        self.done = self.truncated = False
        self.finish = None

    @property
    def context(self):
        return self.prompt + self.gen


def advance(s: Sample, text: str, n_tokens: int, finish_reason, stop_reason, cond: str,
            max_calls: int, executor=execute):
    """Apply one generation chunk to sample `s`. Returns the text injected (or None)."""
    s.gen += text
    s.gen_tokens += n_tokens
    s.rounds += 1
    if finish_reason == "length":
        s.done, s.truncated, s.finish = True, True, "length"
        return None
    if stop_reason not in TOOL_STOPS:
        # A python block closed with ``` and then end-of-sequence never reaches the fence stop (no
        # newline after it). Treat it as a call like any other, so a model that ends this way (base does
        # in about 5% of tool-condition samples) is not silently left without its result.
        tail = text.rstrip()
        blocks = list(CODE_RE.finditer(tail))
        if not (blocks and blocks[-1].end() == len(tail)):
            s.done, s.finish = True, "eos"
            return None
        s.n_eos_code += 1
        stop_reason = None
    code = last_code_block(text)
    if code is None:
        return None  # a non-python fence or a model-written ```output: keep generating
    s.n_calls += 1
    if cond == "tool_on" and s.n_calls <= max_calls:
        result, err = executor(code)
        s.n_exec_err += int(err)
    else:
        result = DENIAL_BUDGET if cond == "tool_on" else DENIAL
        s.n_denied += 1
    if stop_reason == STOP_OUTPUT:
        inj = "\n" + result + "\n```\n"
    else:
        inj = ("" if s.gen.endswith("\n") else "\n") + "```output\n" + result + "\n```\n"
    s.gen += inj
    s.n_injected += 1
    return inj


def splice_round(pending, cond, args, pool):
    """pending: [(sample, text, n_tokens, finish_reason, stop_reason)] for one round. Runs the
    tool_on code calls in parallel, then applies `advance` to every sample."""
    futures = {}
    for s, text, ntok, fin, stop in pending:
        if cond == "tool_on" and stop in TOOL_STOPS and fin != "length" and s.n_calls + 1 <= args.max_calls:
            code = last_code_block(text)
            if code is not None:
                futures[id(s)] = pool.submit(execute, code, args.exec_timeout)
    t0 = time.time()
    if futures:
        print(f"  executing {len(futures)} code blocks ({EXEC_WORKERS} workers, "
              f"{args.exec_timeout:.0f}s timeout each)...", flush=True)
    for s, text, ntok, fin, stop in pending:
        fut = futures.get(id(s))
        ex = (lambda code, f=fut: f.result()) if fut is not None else (lambda code: execute(code, args.exec_timeout))
        advance(s, text, ntok, fin, stop, cond, args.max_calls, executor=ex)
    if futures:
        n_err = sum(1 for s, *_ in pending if id(s) in futures and futures[id(s)].result()[1])
        print(f"  executed in {time.time() - t0:.0f}s, {n_err} errored or timed out", flush=True)


def round_seed(seed, pidx, sidx, rnd):
    """Seed for one sample's later round: a hash, so no two (problem, sample, round) collide."""
    return int.from_bytes(hashlib.blake2b(f"{seed}/{pidx}/{sidx}/{rnd}".encode(), digest_size=4).digest(), "big") % (2 ** 31)


def run_tool_loop(generate, tokenize_len, samples, cond, args, pool):
    """Rounds 2..N for samples that stopped at a code block.
    generate(prompts, stops, budgets, seeds) -> [(text, n_tokens, finish_reason, stop_reason)]."""
    max_rounds = 2 * args.max_calls + 6
    rnd, t0 = 1, time.time()
    while True:
        rnd += 1
        keep, budgets = [], []
        for s in samples:
            if s.done:
                continue
            remaining = min(args.max_tokens - s.gen_tokens, args.max_model_len - tokenize_len(s.context) - 1)
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
        seeds = [round_seed(args.seed, s.pidx, s.sidx, s.rounds) for s in keep]
        outs = generate([s.context for s in keep], stops, budgets, seeds)
        splice_round([(s,) + tuple(o) for s, o in zip(keep, outs)], cond, args, pool)


# --------------------------------------------------------------------------- records
def to_record(s: Sample, item, cond, template, model, tag, dataset, fp=None):
    """One ungraded JSONL row; correct / correct_boxed / arith_* are filled by src/grade.py."""
    text = s.gen
    boxed = last_boxed(text)
    return dict(
        model=model, tag=tag, condition=cond, template=template, dataset=dataset,
        problem_id=item["id"], subject=item.get("subject"), level=item.get("level"), sample_idx=s.sidx,
        correct=None, correct_boxed=None, correct_any=None, gold=item["answer"], boxed=boxed,
        boxed_present=boxed is not None,
        truncated=s.truncated, finish=s.finish, gen_tokens=s.gen_tokens, rounds=s.rounds,
        code_emitted="```python" in text or "```py" in text,
        fake_outputs=max(0, text.count(STOP_OUTPUT) - s.n_injected),
        n_calls=s.n_calls, n_denied=s.n_denied, n_exec_err=s.n_exec_err, n_eos_code=s.n_eos_code,
        arith_n=None, arith_wrong=None, fp=fp, text=text,
    )


# --------------------------------------------------------------------------- checkpointing
# An arm is done when its .meta.json exists (written last). While it runs, each chunk of problems is
# appended to <arm>.jsonl.partial, so a disconnect loses at most one chunk; the final .jsonl is written
# atomically (tmp + rename) and the partial file removed.
def partial_path(path: Path) -> Path:
    return Path(str(path) + ".partial")


def meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def settings_fp(args, n, template=None):
    """Short fingerprint of everything that shapes a sample; checkpoints only resume under the same one."""
    keys = dict(model=args.model, template=template or args.template, temperature=args.temperature, top_p=args.top_p,
                max_tokens=args.max_tokens, max_model_len=args.max_model_len, max_calls=args.max_calls,
                exec_timeout=args.exec_timeout, seed=args.seed, n=n)
    return hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:10]


def git_commit():
    try:
        r = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def arm_done(meta_file: Path, n: int, n_problems: int):
    """A finished arm counts as done if it has at least the samples and problems asked for (more is
    fine). An arm with fewer, e.g. from a smoke run, is redone."""
    if not meta_file.exists():
        return False
    m = json.loads(meta_file.read_text())
    if m.get("n", 0) < n:
        return False
    if n_problems == 0:
        return m.get("n_problems_requested", 0) == 0
    return m.get("n_problems", 0) >= n_problems


def retire(path: Path):
    """Move a finished arm aside (<arm>.jsonl.replaced, <arm>.meta.json.replaced) before it is redone."""
    for f in (path, meta_path(path)):
        if f.exists():
            os.replace(f, Path(str(f) + ".replaced"))
    partial_path(path).unlink(missing_ok=True)


def read_partial(path: Path, n: int, fp=None):
    """Records already checkpointed for this arm, keeping only problems with all n samples (last write
    wins) and, if fp is given, only rows written under the same settings, so a torn chunk is redone and
    a resume can never mix models, temperatures or budgets. Returns (records, finished problem ids)."""
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
            if fp is not None and r.get("fp") != fp:
                continue  # written under other settings
            by_pid.setdefault(r["problem_id"], {})[r["sample_idx"]] = r
    done = {pid for pid, d in by_pid.items() if len(d) == n}
    return [by_pid[pid][k] for pid in done for k in sorted(by_pid[pid])], done


def plan_chunks(items, done_ids, chunk):
    """Remaining (global_index, item) pairs grouped into chunks. Global indices keep per-sample seeds
    independent of chunking and of what was already finished."""
    remaining = [(i, it) for i, it in enumerate(items) if it["id"] not in done_ids]
    size = chunk if chunk > 0 else max(1, len(remaining))
    return [remaining[k:k + size] for k in range(0, len(remaining), size)]


def write_jsonl(path: Path, recs, mode="a"):
    # A crash mid-write leaves a torn last line with no newline; start the append on a fresh line so the
    # torn line stays one ignorable bad record instead of swallowing the next good one.
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


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True, help="role label: base | tool_rl | text_rl")
    p.add_argument("--conditions", default=",".join(CONDITIONS), help=f"comma list from {', '.join(CONDITIONS)}")
    p.add_argument("--template", default="verltool_torl", choices=sorted(TEMPLATES))
    p.add_argument("--dataset", default="aime_all",
                   help=f"comma list of name[:n_problems[:n_samples]]; names: {', '.join(DATASETS)}. "
                        "The model is loaded once for all of them.")
    p.add_argument("--n-problems", type=int, default=0, help="default problems per dataset, 0 = all")
    p.add_argument("--n", type=int, default=64, help="default samples per problem")
    p.add_argument("--max-tokens", type=int, default=3000)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-calls", type=int, default=4)
    p.add_argument("--exec-timeout", type=float, default=10.0)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--chunk", type=int, default=100,
                   help="problems per checkpoint (appended to <arm>.jsonl.partial); 0 = whole arm at once")
    p.add_argument("--out", required=True, help="run folder; arms go to <out>/<dataset>/")
    p.add_argument("--overwrite", action="store_true", help="redo every finished arm")
    p.add_argument("--overwrite-conditions", default="", help="comma list of conditions to redo, others kept")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    redo = {c.strip() for c in args.overwrite_conditions.split(",") if c.strip()}
    bad = sorted(set(conds) - set(CONDITIONS)) + sorted(redo - set(CONDITIONS))
    if bad:
        sys.exit(f"unknown condition(s) {bad}; choose from {list(CONDITIONS)}")
    try:
        specs = parse_dataset_specs(args.dataset, args.n_problems, args.n)
    except ValueError as e:
        sys.exit(str(e))

    def out_path(ds, cond):
        return Path(args.out) / ds / f"{args.tag}__{cond}__{template_for(cond, args.template)}.jsonl"

    def force(cond):
        return args.overwrite or cond in redo

    plan = {ds: [c for c in conds if force(c) or not arm_done(meta_path(out_path(ds, c)), ns, np_)]
            for ds, np_, ns in specs}
    plan = {ds: cs for ds, cs in plan.items() if cs}
    commit = git_commit()
    if not plan:
        print("all conditions already done, nothing to do (use --overwrite to redo)")
        return
    print(f"gen.py starting: {args.tag} {args.model}; todo="
          + ", ".join(f"{ds}/{c}" for ds, cs in plan.items() for c in cs), flush=True)
    print("importing torch + vllm (first time: a minute or two)...", flush=True)
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    # The FlashInfer sampler JIT-builds a kernel on first use (about two minutes on a fresh Colab VM,
    # every time, since the cache dies with the VM). The torch sampler needs no build.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    import torch  # type: ignore
    if not torch.cuda.is_available():
        sys.exit("no CUDA device is visible: this runtime has no GPU. Colab: Runtime > Change runtime type > GPU "
                 "(VS Code: reconnect choosing a GPU runtime), then run again from the first cell.")
    from vllm import LLM, SamplingParams  # type: ignore
    dtype = "bfloat16" if torch.cuda.get_device_capability()[0] >= 8 else "float16"
    t0 = time.time()
    llm = LLM(model=args.model, dtype=dtype, max_model_len=args.max_model_len, seed=args.seed,
              gpu_memory_utilization=args.gpu_mem, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    print(f"loaded {args.model} ({dtype}) in {time.time() - t0:.0f}s", flush=True)
    ban = {i: BAN_BIAS for i in code_token_ids(tok)}
    state = {"bias": None}   # the no_code ban while that arm runs, else None

    def tokenize_len(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def sp(n, max_tokens, stops, seed=None):
        kw = dict(n=n, temperature=args.temperature, top_p=args.top_p, seed=args.seed if seed is None else seed,
                  max_tokens=max_tokens, stop=stops, include_stop_str_in_output=True)
        if state["bias"]:
            kw["logit_bias"] = state["bias"]
        return SamplingParams(**kw)

    def generate(prompts, stops, budgets, seeds):
        res = llm.generate(prompts, [sp(1, b, stops, seed) for b, seed in zip(budgets, seeds)], use_tqdm=True)
        return [(r.outputs[0].text, len(r.outputs[0].token_ids), r.outputs[0].finish_reason,
                 r.outputs[0].stop_reason) for r in res]

    def run_part(part, cond, n):
        """Generate n samples for every (global_index, item) in `part`; returns Samples."""
        prompts = [build_prompt(template_for(cond, args.template), it["problem"]) for _, it in part]
        res = llm.generate(prompts, sp(n, args.max_tokens, TEXT_STOPS + list(TOOL_STOPS)), use_tqdm=True)
        samples, first = [], []
        for (pidx, _), prompt, r in zip(part, prompts, res):
            for sidx, o in enumerate(r.outputs):
                s = Sample(pidx, sidx, prompt)
                samples.append(s)
                first.append((s, o.text, len(o.token_ids), o.finish_reason, o.stop_reason))
        pool = ThreadPoolExecutor(max_workers=EXEC_WORKERS)
        splice_round(first, cond, args, pool)
        run_tool_loop(generate, tokenize_len, samples, cond, args, pool)
        pool.shutdown(wait=True)
        return samples

    for ds, np_, ns in specs:
        if ds not in plan:
            continue
        print(f"loading {ds} from the HF hub...", flush=True)
        items = load_problems(ds, np_, args.seed)
        order = {it["id"]: i for i, it in enumerate(items)}
        (Path(args.out) / ds).mkdir(parents=True, exist_ok=True)
        print(f"{len(items)} problems from {ds}, {ns} samples each", flush=True)
        for cond in plan[ds]:
            t1 = time.time()
            path = out_path(ds, cond)
            state["bias"] = ban if cond in BANNED else None
            if cond in BANNED:
                print(f"   {cond}: {len(ban)} backtick + program-keyword tokens banned"
                      + ("" if ban else "  WARNING: the tokenizer exposes no vocab, nothing is banned"), flush=True)
            fp = settings_fp(args, ns, template_for(cond, args.template))
            if meta_path(path).exists():
                # redo: forced, or the finished arm has fewer samples/problems than asked; set the old one
                # aside first, so a crash during the redo resumes the redo instead of keeping the old arm
                print(f"   retiring the finished {path.name} (kept as .replaced) before redoing it", flush=True)
                retire(path)
            kept, _ = read_partial(path, ns, fp)
            kept = [r for r in kept if r["problem_id"] in order]
            done_ids = {r["problem_id"] for r in kept}
            chunks = plan_chunks(items, done_ids, args.chunk)
            print(f"== {args.tag} / {ds} / {cond}: {len(items)} problems x {ns} samples"
                  + (f"; resuming, {len(done_ids)} problems already on disk" if done_ids else "")
                  + f"; {len(chunks)} chunk(s) to run", flush=True)
            new_recs = []
            for ci, part in enumerate(chunks, 1):
                tc = time.time()
                recs = [to_record(s, items[s.pidx], cond, template_for(cond, args.template), args.model, args.tag, ds, fp)
                        for s in run_part(part, cond, ns)]
                print(f"   writing {len(recs)} records to {partial_path(path).name}...", flush=True)
                write_jsonl(partial_path(path), recs)
                new_recs += recs
                print(f"   chunk {ci}/{len(chunks)}: {len(part)} problems, {(time.time() - tc) / 60:.1f} min, "
                      "checkpointed", flush=True)
            recs = sorted(kept + new_recs, key=lambda r: (order[r["problem_id"]], r["sample_idx"]))
            write_jsonl_atomic(path, recs)
            meta = dict(vars(args), dataset=ds, n=ns, n_problems_requested=np_, condition=cond, dtype=dtype,
                        fp=fp, commit=commit,
                        n_banned=len(ban) if cond in BANNED else 0, template_used=template_for(cond, args.template),
                        graded=False, pass1=None,
                        n_problems=len(items), n_records=len(recs), n_resumed=len(done_ids),
                        gen_seconds=time.time() - t1, total_gen_tokens=sum(r["gen_tokens"] for r in recs),
                        finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            meta_path(path).write_text(json.dumps(meta, indent=1))
            partial_path(path).unlink(missing_ok=True)
            print(f"   done: truncated={sum(r['truncated'] for r in recs) / len(recs):.1%}  "
                  f"called={sum(r['n_calls'] > 0 for r in recs) / len(recs):.1%}  "
                  f"tokens={meta['total_gen_tokens'] / 1e6:.2f}M  {meta['gen_seconds'] / 60:.1f} min -> {path}",
                  flush=True)


if __name__ == "__main__":
    main()
