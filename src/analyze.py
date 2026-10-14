"""Load gen.py results, compute pass@k, arithmetic error rates, and plot.

Usage in a notebook:
    import analyze as an
    df = an.load_results("results/math-1.5b/math500")
    an.summary(df)                 # one row per (tag, condition)
    an.plot(df, ["tool_rl prompt_off", "text_rl prompt_off", "base prompt_off"])   # one panel
    an.plot(df, {"base": ["base prompt_off", "base tool_on"], "tool_rl": [...]})    # one panel per key
    an.plot_passk(df)              # older view: one panel per condition, models as lines
    an.verdict(df)                 # atrophy / internalization / overlap readout
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_ORDER = ["base", "text_rl", "tool_rl"]
# dataviz reference palette, categorical slots 1-3 (CVD-validated): blue, orange, aqua
ROLE_COLOR = {"base": "#2a78d6", "tool_rl": "#eb6834", "text_rl": "#1baf7a"}
COND_ORDER = ["prompt_off", "env_denied", "tool_on", "plain_cot"]
COND_STYLE = {"prompt_off": "-", "env_denied": "--", "tool_on": ":", "plain_cot": "-."}
COND_MARK = {"prompt_off": "o", "env_denied": "s", "tool_on": "^", "plain_cot": "D"}
# conditions get hues of their own (reference palette slots 7, 8, 4, 6, validated as a set) so a
# per-model panel never reuses a model color
COND_COLOR = {"prompt_off": "#6250d6", "env_denied": "#e34948", "tool_on": "#eda100", "plain_cot": "#008300"}
ROLE_STYLE = {"base": "-", "text_rl": "--", "tool_rl": ":"}
GRAY = "#52514e"
# default arms of the headline figure: tool-RL with the tool off against the control and the base
MAIN_ARMS = [("tool_rl", "prompt_off"),
             ("text_rl", "prompt_off"), ("text_rl", "tool_on"),
             ("base", "prompt_off"), ("base", "tool_on")]


def load_results(results_dir, allow_ungraded=False) -> pd.DataFrame:
    """Every *.jsonl in the folder as one DataFrame. Raises if any record is still ungraded
    (gen.py --no-grade) unless allow_ungraded, since pass@k would silently be wrong."""
    paths = sorted(Path(results_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl under {results_dir}")
    frames = []
    for p in paths:
        rows = [json.loads(line) for line in open(p)]
        f = pd.DataFrame(rows)
        f["file"] = p.name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["tag"] + " / " + df["condition"]
    n_un = int(df["correct"].isna().sum()) if "correct" in df else len(df)
    if n_un and not allow_ungraded:
        raise ValueError(f"{n_un} of {len(df)} records in {results_dir} are ungraded: "
                         f"run  python src/grade.py {results_dir}  first")
    if not n_un:
        df["correct"] = df["correct"].astype(bool)
    return df


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator from Chen et al. 2021."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    return 1.0 - math.exp(math.lgamma(n - c + 1) - math.lgamma(n - c - k + 1)
                          - math.lgamma(n + 1) + math.lgamma(n - k + 1))


def ks_for(n: int):
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    return [k for k in ks if k <= n]


def per_problem(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["tag", "condition", "problem_id"])["correct"]
    out = g.agg(n="count", c="sum").reset_index()
    return out


def passk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rows: (tag, condition); columns: pass@k means over problems."""
    pp = per_problem(df)
    rows = []
    for (tag, cond), grp in pp.groupby(["tag", "condition"]):
        n = int(grp["n"].min())
        row = dict(tag=tag, condition=cond, n_problems=len(grp), n_samples=n)
        for k in ks_for(n):
            row[f"pass@{k}"] = float(np.mean([pass_at_k(int(r.n), int(r.c), k) for r in grp.itertuples()]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "tag"]).reset_index(drop=True)


def bootstrap_passk(df: pd.DataFrame, tag: str, cond: str, k: int, iters: int = 1000, seed: int = 0):
    """Percentile bootstrap over problems. Returns (mean, lo, hi)."""
    pp = per_problem(df)
    grp = pp[(pp.tag == tag) & (pp.condition == cond)]
    vals = np.array([pass_at_k(int(r.n), int(r.c), k) for r in grp.itertuples()])
    if len(vals) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = rng.choice(vals, size=(iters, len(vals)), replace=True).mean(axis=1)
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def paired_diff(df: pd.DataFrame, a: tuple, b: tuple, k: int, iters: int = 1000, seed: int = 0):
    """Bootstrap CI of pass@k(a) - pass@k(b), paired by problem. a, b = (tag, condition)."""
    pp = per_problem(df)
    A = pp[(pp.tag == a[0]) & (pp.condition == a[1])].set_index("problem_id")
    B = pp[(pp.tag == b[0]) & (pp.condition == b[1])].set_index("problem_id")
    ids = A.index.intersection(B.index)
    if len(ids) == 0:
        return (float("nan"),) * 3
    da = np.array([pass_at_k(int(A.loc[i, "n"]), int(A.loc[i, "c"]), k) for i in ids])
    db = np.array([pass_at_k(int(B.loc[i, "n"]), int(B.loc[i, "c"]), k) for i in ids])
    d = da - db
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(iters, len(d)), replace=True).mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: accuracy, pass@kmax, failure modes, arithmetic error rate."""
    pk = passk_table(df)
    rows = []
    for (tag, cond), g in df.groupby(["tag", "condition"]):
        n = int(g.groupby("problem_id").size().min())
        kmax = ks_for(n)[-1]
        pkrow = pk[(pk.tag == tag) & (pk.condition == cond)].iloc[0]
        arith_n = int(g["arith_n"].sum())
        rows.append(dict(
            tag=tag, condition=cond, template=g["template"].iloc[0],
            problems=g["problem_id"].nunique(), samples=n,
            **{"pass@1": round(pkrow["pass@1"], 3), f"pass@{kmax}": round(pkrow[f"pass@{kmax}"], 3)},
            trunc_pct=round(100 * g["truncated"].mean(), 1),
            no_boxed_pct=round(100 * (~g["boxed_present"]).mean(), 1),
            code_pct=round(100 * g["code_emitted"].mean(), 1),
            fake_out_pct=round(100 * (g["fake_outputs"] > 0).mean(), 1),
            calls_mean=round(g["n_calls"].mean(), 2), denied_mean=round(g["n_denied"].mean(), 2),
            exec_err_mean=round(g["n_exec_err"].mean(), 2),
            arith_err_pct=round(100 * g["arith_wrong"].sum() / arith_n, 2) if arith_n else float("nan"),
            arith_checked=arith_n, arith_coverage_pct=round(100 * (g["arith_n"] > 0).mean(), 1),
            tokens_mean=round(g["gen_tokens"].mean()),
        ))
    out = pd.DataFrame(rows)
    out["_r"] = out["tag"].map({t: i for i, t in enumerate(ROLE_ORDER)}).fillna(9)
    return out.sort_values(["condition", "_r"]).drop(columns="_r").reset_index(drop=True)


def _plot_lines(df, pk, ax, ci, iters, label_cond=True):
    for _, r in pk.iterrows():
        tag, cond = r["tag"], r["condition"]
        ks = ks_for(int(r["n_samples"]))
        ys = [r[f"pass@{k}"] for k in ks]
        color = ROLE_COLOR.get(tag, "#52514e")
        ax.plot(ks, ys, COND_STYLE.get(cond, "-"), color=color, lw=2,
                marker=COND_MARK.get(cond, "o"), ms=6,
                label=f"{tag} / {cond}" if label_cond else tag)
        if ci:
            lo, hi = [], []
            for k in ks:
                _, l, h = bootstrap_passk(df, tag, cond, k, iters=iters)
                lo.append(l); hi.append(h)
            ax.fill_between(ks, lo, hi, color=color, alpha=0.12, lw=0)
    kmax = ks_for(int(pk["n_samples"].max()))
    ax.set_xscale("log", base=2)
    ax.set_xticks(kmax)
    ax.set_xticklabels([str(k) for k in kmax])
    ax.set_xlabel("k (samples)")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def plot_passk(df: pd.DataFrame, title=None, ci=True, iters=300, facet=True):
    """One panel per condition (three models per panel), or one panel with every arm."""
    import matplotlib.pyplot as plt
    pk = passk_table(df)
    conds = [c for c in COND_ORDER if c in set(pk.condition)] + \
            sorted(set(pk.condition) - set(COND_ORDER))
    if not facet or len(conds) == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        _plot_lines(df, pk, ax, ci, iters)
        ax.set_ylabel("pass@k")
        ax.set_title(title or f"pass@k, {df['problem_id'].nunique()} problems", loc="left")
        return fig
    fig, axes = plt.subplots(1, len(conds), figsize=(4.6 * len(conds), 4.2), dpi=120, sharey=True)
    for ax, cond in zip(axes, conds):
        sub = pk[pk.condition == cond].copy()
        sub["_r"] = sub["tag"].map({t: i for i, t in enumerate(ROLE_ORDER)}).fillna(9)
        _plot_lines(df[df.condition == cond], sub.sort_values("_r"), ax, ci, iters, label_cond=False)
        ax.set_title(cond, loc="left")
    axes[0].set_ylabel("pass@k")
    fig.suptitle(title or f"pass@k, {df['problem_id'].nunique()} problems", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def _format_axis(ax, n_max):
    ks = ks_for(int(n_max))
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (samples)")
    ax.set_ylim(0, 1)
    ax.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_arm(ax, df, pk, tag, cond, color, style, label, ci, iters):
    """One pass@k curve (plus bootstrap band) for arm (tag, cond). Returns False if absent."""
    row = pk[(pk.tag == tag) & (pk.condition == cond)]
    if row.empty:
        return False
    r = row.iloc[0]
    ks = ks_for(int(r["n_samples"]))
    ax.plot(ks, [r[f"pass@{k}"] for k in ks], style, color=color, lw=2,
            marker=COND_MARK.get(cond, "o"), ms=6, label=label)
    if ci:
        lo, hi = zip(*[bootstrap_passk(df, tag, cond, k, iters=iters)[1:] for k in ks])
        ax.fill_between(ks, lo, hi, color=color, alpha=0.12, lw=0)
    return True


def _arm(line):
    """'tool_rl prompt_off', 'tool_rl/prompt_off' or ('tool_rl', 'prompt_off') -> ('tool_rl', 'prompt_off')."""
    if isinstance(line, str):
        parts = line.replace("/", " ").split()
        if len(parts) != 2:
            raise ValueError(f"line must be 'model condition', got {line!r}")
        return parts[0], parts[1]
    tag, cond = line
    return tag, cond


def plot(df: pd.DataFrame, lines, title=None, color="model", ci=True, iters=300, save=None):
    """Draw pass@k curves.

    lines  a list of "model condition" strings for one panel, or a dict {panel title: [lines]} for
           several panels sharing the y axis. Models: base, text_rl, tool_rl. Conditions: prompt_off,
           env_denied, tool_on, plain_cot. "tool_rl/prompt_off" and ("tool_rl", "prompt_off") also work.
    color  "model" or "condition": what the color encodes. The other one sets the line style, but only
           when it varies within a panel.
    ci     shade a 95% bootstrap band over problems.
    save   path of a PNG to write.
    Lines whose arm is not in df are listed and skipped.
    """
    import matplotlib.pyplot as plt
    panels = lines if isinstance(lines, dict) else {None: lines}
    pk = passk_table(df)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 if n == 1 else 4.6 * n, 4.4), dpi=120, sharey=True, squeeze=False)
    missing = []
    for ax, (name, arms) in zip(axes[0], panels.items()):
        arms = [_arm(a) for a in arms]
        vary_tag, vary_cond = len({t for t, _ in arms}) > 1, len({c for _, c in arms}) > 1
        for tag, cond in arms:
            if color == "condition":
                col = COND_COLOR.get(cond, GRAY)
                style = ROLE_STYLE.get(tag, "-") if vary_tag else "-"
            else:
                col = ROLE_COLOR.get(tag, GRAY)
                style = COND_STYLE.get(cond, "-") if vary_cond else "-"
            if not _draw_arm(ax, df, pk, tag, cond, col, style, f"{tag} {cond}", ci, iters):
                missing.append(f"{tag} {cond}")
        _format_axis(ax, pk["n_samples"].max())
        if name:
            ax.set_title(name, loc="left")
        if ax.get_lines():
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("pass@k")
    if title:
        if n == 1:
            axes[0][0].set_title(title, loc="left")
        else:
            fig.suptitle(title, x=0.01, ha="left")
    if missing:
        print("not in results, skipped:", ", ".join(missing))
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


def plot_passk_by_model(df: pd.DataFrame, title=None, ci=True, iters=300):
    """One panel per model, its conditions as lines. Same as plot(df, {model: [...]}, color="condition")."""
    pk = passk_table(df)
    tags = [t for t in ROLE_ORDER if t in set(pk.tag)] + sorted(set(pk.tag) - set(ROLE_ORDER))
    panels = {t: [f"{t} {c}" for c in COND_ORDER if c in set(pk[pk.tag == t].condition)] for t in tags}
    return plot(df, panels, title=title or f"pass@k within each model, {df['problem_id'].nunique()} problems",
                color="condition", ci=ci, iters=iters)


def plot_passk_main(df: pd.DataFrame, arms=MAIN_ARMS, title=None, ci=True, iters=300):
    """The headline arms on one axis. Same as plot(df, [...])."""
    return plot(df, list(arms), title=title or f"pass@k, {df['problem_id'].nunique()} problems", ci=ci, iters=iters)


def verdict(df: pd.DataFrame, cond: str = "env_denied", iters: int = 1000):
    """Print the three-way readout from the plan for one condition (same prompt, all models)."""
    pk = passk_table(df)
    sub = pk[pk.condition == cond]
    tags = set(sub.tag)
    need = {"base", "text_rl", "tool_rl"}
    if not need <= tags:
        print(f"[{cond}] need base, text_rl, tool_rl; have {sorted(tags)}")
        return
    n = int(sub.n_samples.min())
    kmax = ks_for(n)[-1]
    print(f"[{cond}] {sub.n_problems.min()} problems, n={n}, kmax={kmax}")
    for k in (1, kmax):
        for a, b in ((("tool_rl", cond), ("text_rl", cond)), (("tool_rl", cond), ("base", cond)),
                     (("text_rl", cond), ("base", cond))):
            m, lo, hi = paired_diff(df, a, b, k, iters=iters)
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  pass@{k:<4} {a[0]:>7} - {b[0]:<7} = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    g = df[df.condition == cond].groupby("tag")
    er = (g["arith_wrong"].sum() / g["arith_n"].sum() * 100).round(2)
    print("  arithmetic error rate (%):", er.to_dict())
    d1 = paired_diff(df, ("tool_rl", cond), ("text_rl", cond), 1, iters=iters)
    dk = paired_diff(df, ("tool_rl", cond), ("text_rl", cond), kmax, iters=iters)
    if dk[2] < 0:
        print("  -> tool_rl below text_rl at high k: ATROPHY signature")
    elif d1[1] > 0 and dk[1] > 0:
        print("  -> tool_rl above text_rl at pass@1 and high k: INTERNALIZATION signature")
    elif d1[1] <= 0 <= d1[2] and dk[1] <= 0 <= dk[2]:
        print("  -> tool_rl and text_rl overlap at every k: tool added nothing beyond RL (kill criterion)")
    else:
        print("  -> mixed: pass@1 gain without a high-k deficit, or vice versa; look at the curve")
