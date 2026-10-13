# Two class project ideas (2026-10-10)

## 1. Tool atrophy vs internalization

**Question.** When a math model is RL-trained to call a code interpreter, does it lose the ability to compute without it, or does it internalize what the tool taught it?

**Why it is open.** The effect appears in side figures of other papers but nobody has studied it. Fig 6 of arXiv 2505.24480 (Qwen2.5-Math-7B, tool RL, tool removed at eval, vs base text-only): AIME24 pass@1 0.10 vs 0.07, OlymMATH-Hard pass@1024 0.37 vs 0.45. Sharper at low k, less coverage at high k on hard problems. Trice (2605.06326) Table 7 and AdaTIR (2601.14696) show scattered no-tool numbers with no base comparison. Nobody frames atrophy vs internalization, measures per-step arithmetic error rate, or does dose-response over RL checkpoints.

**Pilot (no training).** ToRL-1.5B/7B and SimpleTIR-7B with tools disabled, vs their base models, vs SimpleRL-Zoo (text-only RL from the same base). pass@k to 64 on MATH-500 and AIME. Regex-extracted arithmetic-step error rate. Dose-response over released step checkpoints (VerlTool ToRL 320/360/430, SimpleTIR step20-180).

**Kill.** Tool-RL and text-RL models match within noise at every k and in step error rate.

**Possible extension.** Mix tool-free rollouts into tool RL and test whether the high-k loss disappears.

## 2. Repeat, switch, or quit after a failed test

**Question.** After the k-th failed test run, do coding agents retry the same fix, change approach, or give up, and does RL training make them more stubborn than their base?

**Why it is open.** Loop rates (SWE-Gym 29% repeat an action 3x, SWE-smith length-10 repetition implies 89% failure), give-up rates (Reveree 25% of failures, Codex 98%), and early-failure monitors (FailFast-RestartSmart 66.6 to 71.8 on SWE-bench Verified) all exist. Nobody measures the conditional hazard after the k-th failure or compares RL-trained vs base persistence.

**Pilot.** Trace analysis on existing SWE-bench trajectories. Segment by failed-test events, classify next action as repeat / switch / quit, estimate solve probability conditional on k and action. Compare an RL-trained agent and its base model on identical tasks.

**Kill.** Hazard is flat in k, or SWE-Protege (2602.22124) already covers the RL comparison: it reports RL drives long repetitions to near zero with no SWE-bench gain.

**Caveat.** Descriptive unless paired with an intervention. Ranked below idea 1.
