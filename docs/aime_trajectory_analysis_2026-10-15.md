# AIME trajectory analysis, run 2026-10-15_math-1.5b

Four independent reads of the 34,560 AIME24/25 trajectories (30 problems x 64
samples x 9 arms), plus a harness audit. Per-sample classifications and scripts
are in the run folder on Drive under `analysis/trajectories_2026-10-15/`.

## Headline numbers (pass@1 / pass@64)

| arm | AIME24 | AIME25 |
|---|---|---|
| tool_rl env_denied | 0.13 / 0.43 | 0.06 / 0.40 |
| text_rl env_denied | 0.17 / 0.50 | 0.13 / 0.47 |
| base env_denied | 0.07 / 0.40 | 0.03 / 0.40 |
| tool_rl tool_on | 0.27 / 0.60 | 0.21 / 0.63 |
| text_rl tool_on | 0.22 / 0.53 | 0.17 / 0.43 |

Taking the tool away costs ToRL 14 to 15 points at pass@1. With the tool denied
ToRL is below Oat-Zero at pass@1 on both sets (paired bootstrap excludes zero)
and 6.7 points below at pass@64 (interval includes zero).

## 1. Where the gap sits (per-problem view)

- Tool-dependent problems (ToRL tool_on >= 8/64, env_denied <= 1/64): 4 per set.
  Code there: 6 brute-force enumerations (10^5 to 10^6 cases), 1 numeric
  root-of-unity product, 1 verification. Five of the eight are solved by no model
  without the tool, Oat-Zero and base included: a free pass, not a lost skill.
- Real atrophy candidates: aime24-82 (Oat-Zero 10/64 by hand, ToRL 0) and
  aime25-2 (39/64 vs 1). ToRL writes correct enumeration code, is denied, then
  boxes a one-line guess (post-denial text under 300 chars in 50 to 90 percent of
  those samples, accuracy 0).
- Oat-Zero's pass@1 edge is concentrated: on AIME25 three problems carry it
  (removing them leaves +0.7 points); on AIME24 it is spread over about 7.
- Coverage under env_denied (solved at least once): 13 vs 15 problems on AIME24,
  12 vs 14 on AIME25, same core; base covers nothing the RL models miss. ToRL
  solves the same problems under prompt_off and env_denied (agreement 80 to 90
  percent); the prompt moves rates, not solvability.

## 2. Why ToRL fails when denied (60 wrong trajectories per model)

| primary failure | ToRL | Oat-Zero |
|---|---|---|
| gave up or fabricated after the denial | 31 | 17 |
| wrong plan | 17 | 27 |
| arithmetic slip | 9 | 13 |
| truncated or no box | 3 | 2 |

- Fabrication is always a prose claim in the voice of reading program output
  ("The solution to the system is a = 13/24, b = 5/24, c = 3/8"), never a fake
  output fence.
- Conditional on a denial the two models derail alike (52 percent of ToRL's
  denied wrong samples, 77 percent of Oat-Zero's). The difference is exposure:
  ToRL writes code in over 90 percent of samples, including trivial ones (a 3x3
  linear system, one binomial); Oat-Zero in 33 to 40 percent of its wrong samples
  and 3 to 25 percent of its correct ones.
- 16 of 60 ToRL failures had a sound plan whose code would have produced the
  right answer (Oat-Zero: 5 of 60). About half of those are brute forces no model
  could finish by hand; the rest are small computations it did not attempt.

## 3. The code habit

- ToRL defers the computation to code in about 68 percent of trajectories. In a
  hand-judged sample of 20 denied trajectories, 16 skip the computation after the
  denial and 4 compute by hand; when it computes by hand it succeeds.
- Derailment is rare: under 5 percent retry code, 2 to 3 percent truncate after
  the denial. Code is about 19 percent of the trajectory, so token cost is not
  the mechanism.
- Counterfactual, every ToRL sample at its own no-code accuracy: closes about two
  thirds of the AIME25 deficit and none of the AIME24 one. Low to moderate
  confidence; ToRL's no-code samples are a degenerate tail (long, often truncated).
- Oat-Zero handles a denial worse than ToRL (41 percent quick fabricated boxes vs
  25 percent); its advantage comes entirely from the two thirds of samples that
  never write code.

## 4. Hand-checked arithmetic (120 trajectories, tool denied)

| arm | steps checked | wrong | rate | 95% (cluster bootstrap) |
|---|---|---|---|---|
| ToRL | 190 | 11 | 5.8% | 1.8 to 9.9 |
| Oat-Zero | 256 | 21 | 8.2% | 4.2 to 11.9 |
| base | 242 | 30 | 12.4% | 6.5 to 20.4 |

No evidence that ToRL makes more slips than Oat-Zero; if anything fewer.
Among wrong trajectories, an arithmetic slip explains 1 of 36 ToRL failures and
3 of 34 Oat-Zero failures; "asserted a result after the denial" explains 20 of 36
and 12 of 34. Sample is small; a 5-point true difference cannot be excluded.

## 5. Harness audit (all 34,560 rows)

- Nothing executed under prompt_off or env_denied: 16,636 python blocks
  intercepted, 16,636 denials, 0 executions. The budget message never appears
  under denial; the denial text never appears under tool_on.
- Non-python fences (5 markdown, 3 matlab, 2 javascript, 1 each c, java, asy)
  never ran in any condition.
- 17 output fences under denial contain model-written content ("Done", "0",
  "43", one "SyntaxError"); fabricated, not leaked, 14 of them from base.
- Executed code (tool_on, 10,457 blocks) has no os, subprocess, network, file
  write, input, eval or environment access. 448 blocks hit the 10-second timeout
  on `while True` loops; 27 percent of calls error or time out.

## 6. The automatic arithmetic checker is under-covering

The checker finds about 1,300 equations per arm but reaches only 22 to 26
percent of samples (0.7 equations per sample against about 24 equals signs).
Root cause: `\[`, `\]`, `\(`, `\)` count as glue, so the first and last side of
nearly every displayed equation is discarded; a second bug deletes semicolons.
Fixes, in order of gain: strip math delimiters and treat `\quad`, `\text{}`,
`\boxed`, `\\`, `&`, `\begin`/`\end` as separators (26 to 71 percent of samples
covered); `\sqrt`; implicit multiplication; `\binom` and factorials; nested
fractions; `\pmod`. With all rules about 7,000 equations per arm, 75 percent of
samples, enough to detect a 3-point difference. Also deduplicate identical
equations within a sample: one Oat-Zero trajectory repeats a false line 129
times and alone produces the 20.8 percent AIME24 figure.

## Reading

With the tool denied, ToRL loses to the text-only control, but the trajectories
say this is not an eroded ability to compute. Its hand arithmetic is at least as
good as Oat-Zero's, and when it computes by hand after a denial it succeeds. What
RL with the tool changed is behaviour: it delegates computation to code in about
90 percent of solutions, including trivial ones, and after a denial it asserts
the result the code would have printed instead of deriving it. Half of the
tool-on advantage is brute-force problems no 1.5B model solves by reasoning. The
atrophy that remains is narrow and real: a failure to fall back to hand
computation on enumeration-shaped problems, concentrated in about three problems
per set. Two caveats stand: Oat-Zero and ToRL are different recipes, not the
same recipe minus the tool, and 60 problems is thin at pass@64.

## Next

1. Fix the checker (section 6) and regrade; the arithmetic metric then has power.
2. Separate propensity from ability: a condition that forbids the code fence at
   decoding time, so ToRL must compute by hand without a denial to derail it.
3. More problems at the hard end (AIME 2022 and 2023, HMMT) for pass@64 power.
4. The 7B CIR checkpoint, which makes several calls per problem, for the scaling
   read.

## Addendum, 15:18: the backtick ban (prose_only) is not an ability test

| prose_only | AIME24 pass@1 / @64 | AIME25 | truncated | no box |
|---|---|---|---|---|
| ToRL | 0.04 / 0.27 | 0.03 / 0.27 | 69% | 66% |
| Oat-Zero | 0.14 / 0.40 | 0.11 / 0.50 | 12 to 20% | 12 to 19% |
| base | 0.04 / 0.27 | 0.03 / 0.37 | 67 to 70% | 60% |

The ban held (zero backticks in 11,520 rows) but ToRL and the base answered it
with unfenced Python: `print(sum(A))` repeated until the budget, repetitive tails
in 53 to 60 percent of truncated samples, code mentioned in about 90 percent of
samples. Banning a character does not remove the plan. Two replacements are in
the code: `prose_redirect` cuts the fence the moment it opens and splices in "Let
me do this computation by hand instead", with the ban only after four redirects;
`assist_denied` keeps env_denied's setup and tells the model, in the denial, that
no output exists and to compute by hand step by step.

## Second read, 2026-10-15 evening (three Opus 5.5 agents)

Per-sample files: `analysis/trajectories_2026-10-15_opus/` in the run folder
(`prose/` anatomy of prose_only, `matched/` like-for-like ability comparison,
`falsify/` adversarial review).

### Corrections to sections 3 and 4 above

- The 5.8% vs 8.2% hand arithmetic comparison used different problems for each
  model. On the 25 problems both samples share it is 8.7% (ToRL) vs 6.7%
  (Oat-Zero), within noise either way. Over about 3,000 automatically checked
  steps per arm, single-operation error rates are 0.5 to 1.0% for both and
  indistinguishable. "If anything fewer slips" is withdrawn.
- "When ToRL computes by hand after a denial it succeeds" rested on 4 samples,
  2 correct. Withdrawn.

### What holds

- **No erosion below base.** ToRL with the tool off is never significantly below
  Qwen2.5-Math-1.5B on any problem. Gain over base: +4.8 points env_denied
  [+1.7, +8.3], +2.5 prompt_off [+0.1, +5.3], -0.2 prose_only. Oat-Zero: +9.7,
  +6.4, +8.8.
- **No template confound.** ToRL pass@1 is 0.097 under its raw training template
  (env_denied) and 0.097 under the chat template (prompt_off); one role-token
  leak in 3,840 rows.
- **prose_only is an artifact of the ban.** ToRL writes or announces code in 90%
  of prose_only samples, at the same place (median 1,698 vs 1,564 characters),
  with the same announcement (99%) and the same program (identifier overlap 0.33,
  the same as two prompt_off samples of one problem). Where the fence would go it
  emits a bare `python`, `\[`, `####` or junk tokens. Nothing ends the program,
  so 40 to 42% of samples are code-then-loop. Oat-Zero loses 1 point under the
  ban (0.93x of prompt_off), ToRL 6 points (0.38x).

### What does not hold, and the reconciled estimate

How ToRL compares with Oat-Zero when it actually works by hand depends on what
counts as hand work:

| definition | problems | ToRL minus Oat-Zero, pass@1 |
|---|---|---|
| no code at all, no aborted-code tells, pooled tool-off (strict, k=8) | 23 | +0.4 [-5.2, +6.2] |
| no code at all, pooled (k=3) | 45 | -3.4 [-7.7, +0.6] |
| GENUINE_PROSE under prose_only (>= 5 samples each) | 18 | +0.1 [-7.3, +8.4] |
| finished pure prose under prose_only | 20 | -11.3 [-22, -1] |
| no-code plus post-denial continuations, env_denied | 28 | -10.4 [-19, -3] |

The large negative rows mix populations: most of ToRL's hand work comes after a
denial, while most of Oat-Zero's is prose from the start. Within post-denial
continuations ToRL does better than Oat-Zero (9.4% vs 1.6% accuracy). ToRL's
code-free samples are a 4 to 12% tail chosen by the model itself, toward problems
it codes on least (rank correlation -0.90). The data rule out a deficit above
about 8 points and cannot rule out 3 to 5.

Where ToRL's hand work does fail, it is strategy and enumeration, not arithmetic:
on aime25-2 its multinomials are right but it lists all three cases in 1 of 14
hand attempts (Oat-Zero solves it in about 70% of attempts); on aime25-16 it
reaches "n + 2 divides 39" in 14 to 25 of 64 samples (Oat-Zero 51) and then tries
n = 1, 2, 3.

### The gap is partly the recipe, not the tool

On 10 problems ToRL trails Oat-Zero even with the tool on (aime25-5 28 vs 48 of
64, aime25-15 5 vs 21, aime24-84 19 vs 35). Those problems carry 59% of the
env_denied gap, 33% of prompt_off's and 54% of prose_only's.

### Revised reading

Tool RL left ToRL's tool-free reasoning near the base model's level, while text
RL improved Oat-Zero's by 6 to 10 points. With these two checkpoints that cannot
be called atrophy (lost) as opposed to never gained, because the recipes differ
in data and algorithm as well as the tool. The behavioural findings stand: ToRL
reaches for code in about 90% of solutions, keeps the same plan when the syntax
is banned, and often asserts a result after a denial.

### Grading note

math-verify credits answers that are not boxed: 30 of ToRL's 140 correct
prose_only rows have no `\boxed`. This affects every arm; a boxed-only accuracy
column would make the strictness explicit.

### Next experiments, revised

1. prose_redirect and assist_denied (implemented, not yet run): cheapest way to
   raise ToRL's hand-work rate and tighten the interval.
2. Forced-plan completion: give all three models the same verified prose
   solution up to the point where the computation or enumeration starts, ban
   fences after that prefix, score about 100 items. Holds plan and habit fixed;
   with base as the floor it separates lost from never gained.
3. More problems: 90 to 135 AIME-level problems for 80% power at a 5-point
   difference.
4. A matched pair from one recipe (same data and algorithm, tool on vs off),
   which is what "atrophy" strictly needs.
