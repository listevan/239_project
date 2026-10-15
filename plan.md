# Tool atrophy pilot: plan, abstract, memo

Copied from the Claude Doc https://claude.ai/code/artifact/90f6d2fe-7671-4108-85d9-b3aaa086c6f3 on 2026-10-14 (tabs Plan, Abstract, Memo, in that order; headings demoted one level). The doc is the editable original; this file is the repo copy.

## Tool Atrophy Pilot Plan

Oct 11, 2026 · @Megan

Does RL-training a math model to call a code interpreter erode or improve its tool-free reasoning? We test this on released checkpoints before training anything ourselves.

### Question and hypothesis

The sharp prediction: a tool-RL model with its interpreter removed beats its base at pass@1 but falls below it at high k on hard problems. Figure 6 of [Code-Integrated Reasoning](https://arxiv.org/html/2505.24480) already shows this shape for Qwen2.5-Math-7B, uninterpreted.

| Benchmark | pass@1, base vs tool-RL no tool | pass@1024 |
| --- | --- | --- |
| AIME 2024 | 0.07 vs 0.10 | 0.73 vs 0.73 |
| OlymMATH-Hard | 0.03 vs 0.04 | 0.45 vs 0.37 |

Two readings fit. Atrophy: the model offloaded arithmetic to the tool and lost text-only coverage. Sharpening: any RL narrows the distribution, and the tool is incidental. The text-only RL control from the same base separates them.

What CIR itself concludes from Figure 6, in Section 4.1: code integration expands the capability boundary, and the base model with a code prompt has the highest pass@k at large k. Their one sentence on the tool-off curve compares it to the with-tool curve, not to the base: the drop shows the model "inherently relies on code-integrated reasoning." They never compare the tool-off model to the base, never plot a text-only RL model on the same axes even though Oat-Zero and SimpleRL-Zero sit in their Table 1, and never state how execution was turned off for a model that emits code in nearly 100% of responses by the end of training (their Figure 4).

Kill criterion: tool-RL and text-RL models match within noise at every k and in per-step arithmetic error rate. Then there is no tool-specific effect and we stop.

### Checkpoints to use

Enough released tool-RL checkpoints exist to run the whole first study without training. All repos below returned HTTP 200 on the Hugging Face API on 2026-10-11.

| Model | Base | Size | Step checkpoints | License | Note |
| --- | --- | --- | --- | --- | --- |
| [ZeroTIR](https://huggingface.co/htxu91/zero-tir-7b-550step) | Qwen2.5-7B base | 7B | 250, 550 | MIT | Cleanest zero-RL setup. Paper gives same-setup no-tool numbers. 250-step repo has no card |
| [CIR](https://huggingface.co/RUC-AIBOX/Qwen-Math-7B-CIR) | Qwen2.5-Math-7B base | 7B | final | none | The exact model behind Figure 6. Lets us reproduce the pass@k shape first |
| [VerlTool ToRL](https://huggingface.co/VerlTool) | Qwen2.5-Math-1.5B, 7B | 1.5B, 7B | 140 and 280 same run; 310 to 650 across configs | none | About 30 repos, no cards. Pseudo dose-response, mixed run settings |
| [ToRL](https://huggingface.co/GAIR/ToRL-7B) | Qwen2.5-Math-1.5B, 7B base | 1.5B, 7B | final | none | Official. The paper never evaluates its own model tool-off, so this would be a first |
| [SimpleTIR](https://huggingface.co/ZhenghaiXue/Qwen2.5-7B-SimpleTIR) | Qwen2.5-7B, 32B base | 7B, 32B | final | Apache-2.0 | Official. An unofficial step 20 to 720 series exists but uses a modified algorithm |
| [Qwen3-8B-TIR-DAPO](https://huggingface.co/Heng1999/Qwen3-8B-TIR-DAPO) | Qwen3-8B | 8B | final | Apache-2.0 | Pure RL, no SFT. Its paper already compares TIR vs text pass@k |
| [ReTool-Qwen-32B](https://huggingface.co/JoeYing/ReTool-Qwen-32B) | Qwen2.5-32B-Instruct | 32B | final | Apache-2.0 | Largest clean official release. Needs 2 GPUs for pass@k |
| [TRICE](https://huggingface.co/CajZella/TRICE-4B) | Qwen3-4B-Thinking-2507 | 4B, 30B-A3B | final | Apache-2.0 | Claims no-tool ability preserved. A direct test of that claim |

Start with ZeroTIR and CIR at 7B, and the VerlTool 1.5B series for dose-response. ToRL, Tool-Star, ARPO, and AutoTIR have SFT cold starts or search tools, which muddy the comparison. OTC, AdaTIR, START, TIR-Judge, and the official rStar2-Agent release have no public RL checkpoint.

Text-only RL controls from the same base weights, all zero-RL with no SFT stage, all verified on the Hugging Face API on 2026-10-11.

| Base | Tool-RL model | Text-only RL control | Match |
| --- | --- | --- | --- |
| Qwen2.5-Math-7B | CIR, ToRL-7B | [Oat-Zero](https://huggingface.co/sail/Qwen2.5-Math-7B-Oat-Zero), Dr.GRPO, 400 steps, Apache-2.0. Second: [SimpleRL-Zoo](https://huggingface.co/hkust-nlp/Qwen-2.5-Math-7B-SimpleRL-Zoo) | Clean |
| Qwen2.5-Math-1.5B | ToRL-1.5B, VerlTool | [Oat-Zero 1.5B](https://huggingface.co/sail/Qwen2.5-Math-1.5B-Oat-Zero), 288 steps. [RLVR-Qwen2.5-Math-1.5B](https://huggingface.co/relex-rlvr/RLVR-Qwen2.5-Math-1.5B) has every step 1 to 499 as branches | Clean. The only per-step text-RL series |
| Qwen2.5-7B | ZeroTIR, SimpleTIR-7B | [SimpleRL-Zoo-7B](https://huggingface.co/hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo), same data family. Or [DeepMath-Zero-7B](https://huggingface.co/zwhe99/DeepMath-Zero-7B) | Good. Response cap 8K vs 16K to 24K |
| Qwen2.5-32B | SimpleTIR-32B | [SimpleRL-Zoo-32B](https://huggingface.co/hkust-nlp/Qwen-2.5-32B-SimpleRL-Zoo) or [DAPO-Qwen-32B](https://huggingface.co/BytedTsinghua-SIA/DAPO-Qwen-32B) | Good |
| Qwen2.5-32B-Instruct | ReTool | None exists. ReTool-Qwen-32B-SFT is the only pre-RL control | Weak |
| Qwen3-4B-Thinking-2507 | TRICE | None | No control |

Two cautions. Evaluate every checkpoint with its own template plus one shared template, since Oat-Zero, SimpleRL-Zoo, and ORZ all trained on different prompts. Skip the LUFFY family, which changed rope\_theta and max position in the config.

### First tests

Week one reproduces the Figure 6 shape on the exact CIR model. Week two adds the text-only RL control and the arithmetic error metric. If the tool-RL and text-RL curves overlap everywhere, we stop.

1. **Reproduce.** CIR 7B vs Qwen2.5-Math-7B, tool off. AIME24 and AIME25 pooled, 60 problems, n=256. MATH-500, n=64. Expect pass@1 above base and a crossover near k=128 on the hardest set. If AIME shows no crossover, add OlymMATH-Hard, where the original one was.
2. **Two tool-off conditions, both reported.** Only one of seven prior papers stated how it disabled the tool. Condition A, prompt-off: a plain chain-of-thought prompt identical for all models, no execution, and log how often the model still emits code or fabricates an output block. Condition B, env-denied: the model's own training prompt, stop at the output delimiter, inject the AdaTIR message "Tool call budget exceeded; the code was not executed. Please continue reasoning and give the final answer," cap at 4 injections. An effect under A but not B is prompt mismatch, not lost ability.
3. **Control.** Add a text-only RL model from the same base, Oat-Zero for Qwen2.5-Math-7B, SimpleRL-Zoo as the canonical second, and SimpleRL-Zoo-7B for ZeroTIR. One decoding setting for every model: temperature 0.6, top-p 0.95, 16,384 max tokens, as Yue et al. and CIR used.
4. **Arithmetic error rate.** Extract every "a op b = c" equality, including LaTeX forms, evaluate exactly with fractions, report wrong over extracted per response plus coverage. Never default to zero when nothing parses. Split by final-answer correctness. A tested checker is already written.
5. **Also first-class.** Truncation rate and no-boxed-answer rate. Trice saw 44% truncation text-only after tool-only SFT, with fabricated outputs and 80-cycle import loops.
6. **Dose-response.** VerlTool 1.5B at steps 140 and 280 from one run, then 310 to 650 across configs. ZeroTIR 7B at 250 and 550.

| Outcome | Call |
| --- | --- |
| Tool-RL off below text-RL at high k, or higher arithmetic error than base while text-RL is not | Go. Tool-specific atrophy |
| Tool-RL off above base everywhere, lower arithmetic error | Go. Internalization, write it up as such |
| Tool-RL and text-RL within bootstrap CI at every k and in error rate | Stop |

Compute: about 110M generated tokens per condition at 7B, roughly 6 hours on one H100. Five conditions at 7B and four at 1.5B come to 35 to 40 H100-hours, half that with n=64 on AIME. Verify throughput before committing. A 24 to 48 GB GPU runs 2 to 4 times slower.

### Scaling: what past work says

No paper has run a tool-removed pass@k comparison at two or more model sizes. The only multi-size tool-off evidence is at the pass@1 level, and it points to internalization growing with scale, with one sharp warning about training data mix.

| Paper | Sizes | Tool-off result | Read |
| --- | --- | --- | --- |
| [Trice](https://arxiv.org/abs/2605.06326) Table 7 | 4B, 30B-A3B | 4B: 4 of 5 benchmarks up vs base, AIME25 down 3.3. 30B: 4 of 5 up with larger gains, HMMT25 +6.1, BeyondAIME +5.1 | Internalization, larger at 30B. Recipe is mixed tool and text SFT, then RL |
| Trice, SFT stage only | 4B | BeyondAIME 54.3 to 34.1 after tool-only SFT, 49.0 after mixed SFT | Atrophy is real, caused by data mix, and mostly repairable |
| [CIR](https://arxiv.org/html/2505.24480v1) Fig 6 | 7B only | pass@1 above base on all 4 benchmarks. pass@1024 below base only on OlymMATH-Hard, 0.37 vs 0.45, crossover near k=128 | The single large-k tool-off curve in print. Deficit is hard-problem specific |
| [Tool-Overuse Illusion](https://arxiv.org/html/2604.19749v1) | ReTool 7B, 32B | 32B makes 3.09 calls on simple GSM8K items vs 1.19 at 7B. Tool gain on hard items smaller at 32B | Larger tool-RL models offload more, not less |
| [Invisible Leash](https://arxiv.org/abs/2507.14843) | 1.5B, 7B, 14B text RL | Support preservation 0.93, 0.97, 0.99 | RL coverage loss shrinks with size but never reaches zero |
| [Ramasesh et al. 2022](https://iclr.cc/virtual/2022/poster/6452) | LMs to 8.6B, supervised | Forgetting falls monotonically with scale | Generic prior, not RL |

Prediction from the literature: the tool-off pass@1 advantage persists or grows with scale, moderate confidence. The large-k coverage deficit shrinks but does not vanish, low confidence, since every large-k data point is 14B or below. Nothing published predicts a sign flip at pass@1.

The biggest confound is recipe, not size. Mixed versus tool-only training data moved Trice's numbers more than 4B versus 30B did. Our scaling test must hold the recipe fixed: SimpleTIR 7B vs 32B is the one verified pair with the same algorithm and data. ReTool-Qwen-32B, SimpleTIR-32B, and rStar2-Agent-14B have never had a tool-off eval on competition math.

### Beyond math: coding agents and search

The coding-agent version is open and feasible. Clean base-to-agent pairs exist at 8B and 32B, the tool-off metric exists, and nobody has combined them. All repos verified on the Hugging Face API on 2026-10-11. None ships intermediate checkpoints.

| Base | Agent-RL model | Stages | Tools in RL | SWE-bench Verified | Tool-off eval by authors |
| --- | --- | --- | --- | --- | --- |
| Qwen3-32B | [DeepSWE-Preview](https://huggingface.co/agentica-org/DeepSWE-Preview), MIT. Second: [SA-SWE-32B](https://huggingface.co/NovaSky-AI/SA-SWE-32B) | RL only, no SFT | bash, file editor, search | 42.2; 39.4 | None. A third party measured LiveCodeBench 61.0 to 60.1 |
| Qwen2.5-Coder-32B-Instruct | [SWE-Master-32B-SFT](https://huggingface.co/RUC-AIBOX/SWE-Master-32B-SFT) then [SWE-Master-32B-RL](https://huggingface.co/RUC-AIBOX/SWE-Master-32B-RL) | SFT then RL | R2E-Gym bash, editor | 57.8 to 61.4 | None |
| Qwen3-8B | [Klear-AgentForge-8B](https://huggingface.co/Kwai-Klear/Klear-AgentForge-8B) and [SWE-AGILE-RL-8B](https://huggingface.co/KDEGroup/SWE-AGILE-RL-8B), each with an SFT sibling | SFT then RL | bash, str\_replace edit | 38.2 to 39.4; 21.5 to 24.1 | None |
| Qwen3-4B-Instruct-2507 | [SWE-Master-4B-RL](https://huggingface.co/RUC-AIBOX/SWE-Master-4B-RL) with SFT sibling | SFT then RL | R2E-Gym | 27.6 to 33.4 | None. Cheapest pilot |

The tool-off analog of arithmetic is predicting what code does without running it. CRUXEval-O asks exactly that, and the shell does exactly that for the agent. Single-shot LiveCodeBench and HumanEval+ cover code generation without a scaffold.

What prior work shows, none of it on a tool-executing RL agent against its base:

- [SWE-RL](https://arxiv.org/abs/2502.18449), 70B, text-only RL that never called a tool: CRUXEval-O rose 61.9 to 75.5. So RL without a tool improves execution reasoning. Whether RL with a shell does too is the open question.
- [Code World Model](https://arxiv.org/abs/2510.02387): joint RL including agentic SWE moved CRUXEval-O direct mode 67.8 to 66.6 and chain-of-thought mode 83.3 to 94.3. Mixed, and the RL is not isolated.
- [Benchmark-oriented optimization study](https://arxiv.org/abs/2608.13566): trajectory SFT hurts tool-off coding, SWE-agent-LM-7B fell from 15.2 to 6.7 on LiveCodeBench, while pure RL was flat, DeepSWE 61.0 to 60.1. Eighteen of 28 out-of-distribution comparisons degraded.

Two cautions. Hold Qwen3 thinking mode and the chat template fixed across base and agent, since a 7B agent model can look collapsed just from losing its scaffold. Dose-response needs re-running the open DeepSWE or SWE-AGILE recipe, because no repo has step checkpoints.

**Search tools.** The search version is open too, and it has the cleanest control of the three domains. No paper has evaluated a released search-RL checkpoint with retrieval disabled. The tool-off analog is closed-book question answering.

| Base | Search-RL model | Text-only RL twin | Why it is clean |
| --- | --- | --- | --- |
| Qwen2.5-3B and 7B, base and Instruct | [Search-R1 v0.2 PPO](https://huggingface.co/PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.2), Wikipedia retriever | [R1 v0.2 PPO](https://huggingface.co/PeterJinGo/R1-nq_hotpotqa_train-qwen2.5-7b-em-ppo-v0.2), same author, data, algorithm, and code | Only the tool differs. Four complete triplets, so 3B vs 7B comes free |
| Same bases | [ZeroSearch](https://huggingface.co/Alibaba-NLP/ZeroSearch_wiki_V2_Qwen2.5_7B), [IKEA](https://huggingface.co/hzy/ikea-qwen2.5-7b), [AdaSearch](https://huggingface.co/hank0316/AdaSearch-Qwen2.5-7B-Instruct) | Same R1 twins | Recipe variants on identical data. IKEA and AdaSearch train the model to search less, so they predict less atrophy |
| Qwen3-8B | [Search-R1-Qwen3-8B-SDAR](https://huggingface.co/wckwan/Search-R1-Qwen3-8B-SDAR), steps 20 to 400 public | None | The only search-RL repo with step checkpoints |

What prior work shows:

- Text-only RL on the same QA data raises closed-book exact match by 7 to 15 points. Search-R1's own Table 2: Qwen2.5-7B direct 0.181, text-RL 0.276. So the control is expected to go up, and the question is whether the search-trained model follows it.
- Search-RL models stop answering from memory. AdaSearch re-ran the official Search-R1 7B checkpoint and found it searches on 100% of queries. Nobody checked whether the knowledge is still there.
- Two indirect hints say it is. [SAAS](https://arxiv.org/abs/2605.29796) found the share of questions the model could answer without searching rose from 12.7% to 24.3% during search RL. A Qwen3.5-4B search agent in [2607.20090](https://arxiv.org/abs/2607.20090) kept closed-book HotpotQA flat, 18.01 to 17.98.
- One weak hint the other way. Search-R1's [follow-up study](https://arxiv.org/abs/2505.15117) trained with a noise retriever: the model learned to stop searching and ended at 0.241, below the 0.276 text-RL run.

The confound is format collapse. A model that searches on every query will emit a search tag or stall when the tool is gone, which looks like lost knowledge and is not. So three protocols: tool removed and search tags banned; tool present but every call returns "search unavailable"; and avg@16 solve rate per question, which ignores formatting. A deficit that survives the third is real atrophy. Benchmarks are Search-R1's seven test sets with its normalized exact match, so the published direct and text-RL rows anchor us, plus PopQA bucketed by entity popularity with correct, wrong, and abstained graded separately. Expected: base below search-RL below or equal to text-RL.

### Open questions for you

- **Hardware.** What GPUs, and for how many hours? This sets n on AIME, whether 32B is in scope, and whether the dose-response series fits.
- **Solo or team?** The plan above is sized for one person over about four weeks of compute and analysis.
- **Does the class accept a diagnostic result?** Studies one and two produce a finding, not a method. Study three, the fix, needs training and may not fit.
- **Math only?** Every open tool-RL checkpoint is a math model. A coding-agent version would require training our own tool-RL model first.
- **Which base family?** Qwen2.5-Math gives the cleanest controls and the Figure 6 reproduction. Qwen3-8B gives a newer model but its base is already post-trained.

### Sources

Papers: [CIR 2505.24480](https://arxiv.org/html/2505.24480v1), [Trice 2605.06326](https://arxiv.org/abs/2605.06326), [AdaTIR 2601.14696](https://arxiv.org/abs/2601.14696), [Tool-Overuse Illusion 2604.19749](https://arxiv.org/html/2604.19749v1), [TIR-Judge 2510.23038](https://arxiv.org/abs/2510.23038), [ToRL 2503.23383](https://arxiv.org/abs/2503.23383), [SimpleTIR 2509.02479](https://arxiv.org/abs/2509.02479), [ZeroTIR 2505.07773](https://arxiv.org/abs/2505.07773), [ReTool 2504.11536](https://arxiv.org/abs/2504.11536), [Yue et al. pass@k 2504.13837](https://arxiv.org/abs/2504.13837), [Invisible Leash 2507.14843](https://arxiv.org/abs/2507.14843), [Ramasesh et al. ICLR 2022](https://iclr.cc/virtual/2022/poster/6452), [Dr.GRPO 2503.20783](https://arxiv.org/abs/2503.20783), [SimpleRL-Zoo 2503.18892](https://arxiv.org/abs/2503.18892), [arithmetic checker precedent 2603.21558](https://arxiv.org/abs/2603.21558).

Checkpoints: [ZeroTIR](https://huggingface.co/htxu91/zero-tir-7b-550step), [CIR](https://huggingface.co/RUC-AIBOX/Qwen-Math-7B-CIR), [VerlTool](https://huggingface.co/VerlTool), [ToRL](https://huggingface.co/GAIR/ToRL-7B), [SimpleTIR](https://huggingface.co/ZhenghaiXue/Qwen2.5-7B-SimpleTIR), [Qwen3-8B-TIR-DAPO](https://huggingface.co/Heng1999/Qwen3-8B-TIR-DAPO), [ReTool-Qwen-32B](https://huggingface.co/JoeYing/ReTool-Qwen-32B), [TRICE](https://huggingface.co/CajZella/TRICE-4B). Controls are linked in their table above.

Local: the plan and the tested arithmetic checker are in \~/projects/research-ideas/.

---

## Does Learning to Use a Code Interpreter Make a Model Worse at Reasoning Without One?

Oct 11, 2026 · @Megan

Reinforcement learning with a code interpreter is now the standard way to train math reasoning models. The model learns when to hand a calculation to Python and reads the result back, and papers such as ToRL, ReTool, and SimpleTIR report large gains with the tool on. Almost nobody evaluates these models with the tool off. The few that do disagree, and the one careful figure shows a strange shape: the tool-trained model beats its base at pass@1 but falls below it at pass@1024 on hard problems. Two stories fit. The model may have internalized what the interpreter taught it and simply become more decisive. Or it may have offloaded arithmetic to the tool, lost the skill, and kept only the confident habits. I want to study which story is true, and whether any loss is specific to tool training or a general cost of RL.

I compare three models that share one base: the base itself, a tool-RL model with its interpreter disabled at evaluation, and a text-only RL model trained on the same base with no tool. The text-only control separates tool-specific atrophy from the distribution sharpening that any outcome-reward RL causes. All three exist as released checkpoints, so the first two studies need no training.

The first study measures what is lost. I will run pass@k from 1 to at least 64 on MATH-500 and AIME with the interpreter off, and read coverage at high k against the base. I will also extract every arithmetic equality in each response and check it with Python, giving a per-step arithmetic error rate. Atrophy predicts a higher error rate than the base and lower coverage at high k. Internalization predicts a lower error rate and coverage at least as good. The text-only RL model tells me how much of either effect RL alone would have produced.

The second study asks how the effect grows. Several groups release intermediate RL checkpoints, so I will trace the tool-off deficit across training steps and against the model's tool-call rate at each step. Atrophy predicts the deficit grows with steps and tracks tool use. I will repeat the comparison at 1.5B and 7B, and at 32B if the checkpoints run, to see whether the deficit shrinks with scale, since larger models forget less under fine-tuning but also call tools more often, and the two forces pull in opposite directions.

The third study, if the first two find real atrophy, tests the cheapest fix. Mixing tool-free rollouts into tool RL, or penalizing tool calls as OTC and AdaTIR do, should keep the skill alive. I will check whether released models trained with tool-call penalties show a smaller deficit before attempting any training myself.

Together the three studies test whether tool training removes a skill the model already had, whether the loss belongs to the tool or to RL, and whether it grows with training or fades with scale.

---

## Memo: tool atrophy

Oct 11, 2026 · @Megan

**Question.** When a model is RL-trained to call a tool, what happens to the skill the tool replaces? Does the model absorb it, or hand it off and let it decay?

**The test in one line.** Base, tool-RL with the tool off, and text-only RL, all from one base, same prompt, pass@k to 256 plus arithmetic error rate. Tool-RL below text-RL at high k or in error rate is atrophy. Tool-RL above both is internalization. Overlap means the tool did nothing beyond RL, and we stop.

**Why it matters.** Tool RL is now how frontier models are trained for math, code, and search. One published figure shows the atrophy shape at 7B and reads it as something else. Nobody has run the control.

**Open issue 1: scaling up.** CIR measured tool-off only at 7B, and no paper has run it at two sizes of one recipe. Trice found bigger models keep more without the tool, while Tool-Overuse Illusion found bigger models call the tool more, so the direction is open.

one possible direction: debugger, give solution

**Open issue 2: which tool to test for coding agents.** Math has one tool, so tool-off is one clean condition, and that is where we start. A coding agent has a shell, edits, search, and tests, so we have not settled what tool-off means there. CRUXEval-O, predicting code output without running it, is the leading candidate.
