# no_code: the token ban, and how it was checked (2026-10-16)

`no_code` gives the model its own training prompt and bans, through vLLM
`logit_bias = -100`, every token that programs need and math prose does not.
Nothing is inserted or redirected. Implemented in `gen.code_token_ids`.

## The list

- every vocabulary token containing a backtick (no code fence can open)
- words, matched as a whole token or a token continuing with punctuation, never
  a longer word: `print`, `import`, `elif`, `sympy` / `symp`, `numpy`, `itertools`
- symbols, exact match only: `==`, `+=`, `-=`, `*=`, `/=`, `!=`, `//`, `#`

156 token ids on the Qwen2.5-Math tokenizer, identical for Qwen2.5-Math-1.5B
and both VerlTool checkpoints (same vocabulary and merges).

## Kept allowed on purpose

| token | why it stays allowed |
|---|---|
| `lambda` | `\lambda` tokenizes as backslash + `lambda`: banning it blocks λ |
| `def` | an AIME problem names its digits `abc` and `def` |
| `np` | the binomial mean is written `np` |
| `return` | word problems say "return to its original position" |
| `for`, `in`, `if`, `range`, `while` | ordinary English in math prose |
| `###` | markdown headings in prose (only the single `#` is banned) |
| indentation, `len`, `True`, `False` | add no coverage, touch 6 to 13% of prose |

## Verification

Corpora: 12,000 python-tagged code blocks and 12,000 code-free prose samples, from
VerlTool's released training rollouts (the models we run) and the 2026-10-15 AIME
trajectories (ToRL, Oat-Zero, base). Same tokenizer as the run.

| check | result |
|---|---|
| code blocks containing a banned token | 11,999 / 12,000 (99.99%); the one leak is the block `2` |
| first banned token within the first token / 3 / 10 tokens of the block | 77% / 93% / 96% |
| code-free prose samples containing a banned token | 54 / 12,000 (0.45%), 43 of them base-model text that copies Asymptote code (`i+=30`) or announces "Python and sympy" |
| ToRL outputs from the old backtick-only ban (unfenced code) containing a banned token | 3,166 / 3,840 (82%) |
| `gen.code_token_ids` through Hugging Face `AutoTokenizer` (what vLLM uses) | 156 ids, equal to the validated list, equal across the three models |

Spot checks through the real tokenizer: `\lambda`, "digits abc and def", "the mean
is np", "return to the start", "### Step 1", "range of values", "important" are
allowed; `import sympy as sp`, `print(x)`, `if a == b:`, `count += 1`, `x // 2`,
`# comment`, `from itertools import ...`, `\text{import numpy}` are blocked.
Known cost: LaTeX `\#` ("\# of ways") is blocked with the comment marker.

## Not verifiable offline

How the models respond when these tokens are unavailable: whether they switch to
prose, write keyword-free code-like lines, or loop. Check in the smoke run:
truncation of the no_code arms (the backtick-only ban gave ToRL 69%), and a read
of a few tool_rl no_code rows.
