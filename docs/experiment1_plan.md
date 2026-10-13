# Tool atrophy vs internalization: pilot plan (2026-10-11)

Doc with full tables: https://claude.ai/code/artifact/90f6d2fe-7671-4108-85d9-b3aaa086c6f3

## Question

Does RL-training a math model to call a code interpreter erode or improve its tool-free reasoning? Sharp prediction from CIR (arXiv 2505.24480, Fig 6, Qwen2.5-Math-7B, 7B only): tool-RL model with tool off beats base at pass@1 on all four benchmarks, but falls below base at pass@1024 on OlymMATH-Hard (0.37 vs 0.45, crossover near k=128). AIME24/25 and OlymMATH-Easy show no crossover. Nobody has interpreted this or run it at two sizes.

CIR's own reading (Sec 4.1): code integration expands the capability boundary; Base_code has the highest pass@k at large k. Their one sentence on RL_text compares it to RL_code, not to Base_text ("inherently relies on code-integrated reasoning"). No text-RL curve in Fig 6 (Oat-Zero and SimpleRL-Zero are in their Table 1 but only with tool off at avg@16), no statement of how execution was disabled for a model that writes code in ~100% of responses (their Fig 4), one size.

Kill: tool-RL (tool off) and text-only RL from the same base match within bootstrap CI at every k and in arithmetic error rate.

## Checkpoints (all verified on HF 2026-10-11)

| Model | Base | Size | Steps | License |
| --- | --- | --- | --- | --- |
| htxu91/zero-tir-7b-{250,550}step | Qwen2.5-7B base | 7B | 250, 550 | MIT |
| RUC-AIBOX/Qwen-Math-7B-CIR | Qwen2.5-Math-7B | 7B | final | none (the Fig 6 model) |
| VerlTool/torl-* (~30 repos) | Qwen2.5-Math-1.5B/7B | 1.5B, 7B | 140/280 same run; 310-650 across configs | none, no cards |
| GAIR/ToRL-1.5B, ToRL-7B | Qwen2.5-Math-1.5B/7B | 1.5B, 7B | final | none |
| ZhenghaiXue/Qwen2.5-{7B,32B}-SimpleTIR | Qwen2.5-7B/32B base | 7B, 32B | final | Apache-2.0 |
| Heng1999/Qwen3-8B-TIR-DAPO | Qwen3-8B (post-trained) | 8B | final | Apache-2.0 |
| JoeYing/ReTool-Qwen-32B | Qwen2.5-32B-Instruct | 32B | final | Apache-2.0 |
| CajZella/TRICE-{4B,30B} | Qwen3-4B-Thinking-2507 | 4B, 30B-A3B | final | Apache-2.0 |

Start: ZeroTIR + CIR at 7B, VerlTool 1.5B for dose-response. No public RL checkpoint: OTC, AdaTIR, START, TIR-Judge, official rStar2-Agent.

Text-only RL controls (same base, zero-RL, verified on HF 2026-10-11):

| Base | Tool-RL | Text-RL control | Match |
| --- | --- | --- | --- |
| Qwen2.5-Math-7B | CIR, ToRL-7B | sail/Qwen2.5-Math-7B-Oat-Zero (Dr.GRPO, 400 steps, Apache-2.0); hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo | clean |
| Qwen2.5-Math-1.5B | ToRL-1.5B, VerlTool | sail/Qwen2.5-Math-1.5B-Oat-Zero (288 steps); relex-rlvr/RLVR-Qwen2.5-Math-1.5B (branches step_1..step_499, the only per-step text-RL series) | clean |
| Qwen2.5-7B | ZeroTIR, SimpleTIR-7B | hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo (same data family); zwhe99/DeepMath-Zero-7B | good, response cap 8K vs 16-24K |
| Qwen2.5-32B | SimpleTIR-32B | hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo; BytedTsinghua-SIA/DAPO-Qwen-32B | good |
| Qwen2.5-32B-Instruct | ReTool | none; ReTool-Qwen-32B-SFT is the only pre-RL control | weak |
| Qwen3-4B-Thinking-2507 | TRICE | none | no control |

Cautions: evaluate each checkpoint with its own template plus one shared template (Oat-Zero, SimpleRL-Zoo, ORZ trained on different prompts). Skip the LUFFY family (rope_theta and max_pos changed). hkust-nlp SimpleRL-Zoo for Math-1.5B does not exist.

## First tests (two weeks)

1. Reproduce Fig 6 on CIR-7B vs Qwen2.5-Math-7B, tool off. AIME24+25 pooled n=256, MATH-500 n=64. If no crossover on AIME, add OlymMATH-Hard.
2. Two tool-off conditions, both reported. A prompt-off: plain CoT prompt identical for all models, no execution; log code-emission and fabricated-output rates. B env-denied: model's own training prompt, stop at the output delimiter, inject "Tool call budget exceeded; the code was not executed. Please continue reasoning and give the final answer." (AdaTIR), cap 4. Effect under A but not B = prompt mismatch, not lost ability. Only 1 of 7 prior papers stated its mechanism.
3. Control: Oat-Zero for Qwen2.5-Math-7B (SimpleRL-Zoo second), SimpleRL-Zoo-7B for ZeroTIR. One decoding setting for all: T 0.6, top-p 0.95, 16,384 max tokens (Yue et al. 2504.13837; CIR).
4. Arithmetic error rate: arith_check.py in this folder (regex "a op b = c" incl. LaTeX, exact Fractions, report wrong/extracted + coverage, never default to 0). Split by final-answer correctness.
5. Also first-class: truncation rate, no-boxed rate (Trice saw 44% truncation after tool-only SFT, fabricated outputs, 80-cycle loops).
6. Dose-response: VerlTool 1.5B steps 140/280 then 310-650; ZeroTIR 250/550.

Decision: tool-RL off below text-RL at high k, or higher arithmetic error than base while text-RL is not = atrophy, go. Tool-RL off above base everywhere with lower error = internalization, go. Both match = stop.

Compute: ~110M tokens per condition at 7B, ~6 h on one H100. 5 conditions at 7B + 4 at 1.5B = 35-40 H100-hours; half with n=64 on AIME.

## Scaling: what past work says

No tool-removed pass@k at two sizes exists. Evidence:
- Trice (2605.06326) Table 7: tool-off vs base, 4B 4/5 benchmarks up (AIME25 -3.3), 30B-A3B 4/5 up with larger gains (HMMT25 +6.1, BeyondAIME +5.1). Internalization, bigger at 30B. But recipe is mixed tool/text SFT then RL.
- Trice SFT stage, 4B: BeyondAIME 54.3 -> 34.1 after tool-only SFT, 49.0 after mixed. Atrophy is real, data-mix driven, repairable.
- Tool-Overuse Illusion (2604.19749): ReTool-32B makes 3.09 calls on simple GSM8K vs 1.19 at 7B; tool gain on hard items smaller at 32B. Bigger models offload more.
- Invisible Leash (2507.14843): text-RL support preservation 0.93 (1.5B), 0.97 (7B), 0.99 (14B). Coverage loss shrinks, never zero.
- Ramasesh et al. ICLR 2022: forgetting falls monotonically with scale (supervised, to 8.6B).

Prediction: tool-off pass@1 advantage persists or grows with scale (moderate confidence). Large-k deficit shrinks but does not vanish (low confidence; all large-k data is 14B or below). No published sign flip at pass@1. Biggest confound is recipe, not size: hold it fixed, SimpleTIR 7B vs 32B is the one verified same-recipe pair. ReTool-32B, SimpleTIR-32B, rStar2-Agent-14B never had a tool-off eval on competition math.

## Open questions

- Hardware and hours (sets n, 32B in scope or not).
- Solo or team.
- Does the class accept a diagnostic result, or does it need the fix (study 3, needs training)?
- Math only is fine? All open tool-RL checkpoints are math models.
- Base family: Qwen2.5-Math (cleanest controls, Fig 6 reproduction) vs Qwen3-8B (newer, base already post-trained).

## Beyond math: coding agents (verified HF 2026-10-11, none has step checkpoints)

| Base | Agent-RL model | Stages | Tools | SWE-bench V | Tool-off eval by authors |
| --- | --- | --- | --- | --- | --- |
| Qwen3-32B | agentica-org/DeepSWE-Preview (MIT); NovaSky-AI/SA-SWE-32B | RL only | bash, editor, search | 42.2; 39.4 | none; third party LiveCodeBench 61.0 -> 60.1 |
| Qwen2.5-Coder-32B-Instruct | RUC-AIBOX/SWE-Master-32B-SFT -> -RL | SFT -> RL | R2E-Gym bash, editor | 57.8 -> 61.4 | none |
| Qwen3-8B | Kwai-Klear/Klear-AgentForge-8B(-SFT); KDEGroup/SWE-AGILE-(SFT|RL)-8B | SFT -> RL | bash, str_replace | 38.2 -> 39.4; 21.5 -> 24.1 | none |
| Qwen3-4B-Instruct-2507 | RUC-AIBOX/SWE-Master-4B-SFT -> -RL | SFT -> RL | R2E-Gym | 27.6 -> 33.4 | none; cheapest pilot |

Tool-off analog: CRUXEval-O (predict code output without running it) is the arithmetic analog; single-shot LiveCodeBench / HumanEval+ for generation without scaffold.
Prior: SWE-RL 70B (text-only RL, no tools) CRUXEval-O 61.9 -> 75.5. CWM joint RL: CRUXEval-O direct 67.8 -> 66.6, CoT 83.3 -> 94.3. 2608.13566: trajectory SFT hurts tool-off coding (SWE-agent-LM-7B LCB 15.2 -> 6.7), pure RL flat (DeepSWE -0.9); 18/28 OOD comparisons degraded. Nobody has run CRUXEval on a tool-executing RL agent vs its base.
Cautions: hold Qwen3 thinking mode and chat template fixed; no step checkpoints anywhere, so dose-response means re-running DeepSWE/rLLM or SWE-AGILE recipe.

## Beyond math: search tools (verified HF 2026-10-11)

Cleanest control of all three domains: PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-{3b,3b-it,7b,7b-it}-em-ppo-v0.2 (Wikipedia retriever) vs PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-{3b,3b-it,7b,7b-it}-em-ppo-v0.2 (text-only RL twin, same author/data/algo/code). Four complete triplets, 3B vs 7B free. Use v0.2 on both sides (v0.3 has no R1 twin). Recipe variants on same base and data: Alibaba-NLP/ZeroSearch_wiki_V2_*, hzy/ikea-qwen2.5-*, hank0316/AdaSearch-*. Only search-RL repo with step checkpoints: wckwan/Search-R1-Qwen3-8B-SDAR (step_20..400), no text-RL twin. No-retriever RL arm: TsinghuaC3I/*-SSRL.

Prior: no paper evaluates a released search-RL checkpoint with retrieval off. Text-RL raises closed-book EM 7-15 pts (Search-R1 Table 2: 7B direct 0.181 -> R1 0.276). Search-R1 7B searches on 100% of queries (AdaSearch re-eval). SAAS: answerable-without-search share rose 12.7% -> 24.3% during training. 2607.20090 (Qwen3.5-4B search agent): closed-book HotpotQA 18.01 -> 17.98 flat. 2505.15117 Table 4: noise-retriever run ends at 0.241 vs R1 0.276 (weak hint of loss).

Confound: format collapse (emits <search> or stalls). Three protocols: (i) tool removed, search tags banned; (ii) tool present, returns "search unavailable"; (iii) avg@16 solve rate per question. Deficit surviving (iii) = real atrophy. Benchmarks: Search-R1's seven test sets, normalized EM; PopQA bucketed by popularity with correct/wrong/abstain; behavioral metrics: search-tag emission rate and abstention rate with tool off. Expected: base < search-RL(no tool) <= text-RL.

## Domain ranking for a pilot
1. Math (CIR/ZeroTIR): hook already published, controls exist, cheapest.
2. Search (Search-R1 vs R1): cleanest control, 3B/7B, but expected effect is small and behavioral.
3. Coding agents (DeepSWE, SWE-Master): open and interesting (CRUXEval-O), but 32B inference and no step checkpoints.
