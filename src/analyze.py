"""Load gen.py results, compute pass@k, arithmetic error rates, and plot.

Usage in a notebook:
    import analyze as an
    df = an.load_results("results/math-1.5b/math500")
    an.summary(df)                 # one row per (tag, condition)
    an.calls_table(df)             # tool-call behaviour: how many called, how often, retries after a denial
    an.plot(df, ["tool_rl prompt_off", "text_rl prompt_off", "base prompt_off"])   # one panel
    an.plot(df, {"base": ["base prompt_off", "base tool_on"], "tool_rl": [...]})    # one panel per key
    an.plot_passk(df)              # older view: one panel per condition, models as lines
    an.verdict(df)                 # atrophy / internalization / overlap readout
"""
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_ORDER = ["base", "text_rl", "tool_rl"]
# dataviz reference palette, categorical slots 1-3 (CVD-validated): blue, orange, aqua
ROLE_COLOR = {"base": "#2a78d6", "tool_rl": "#eb6834", "text_rl": "#1baf7a"}
COND_ORDER = ["tool_on", "env_denied", "no_code", "prompt_off", "assist_denied", "prose_only", "prose_redirect",
              "plain_cot"]
COND_STYLE = {"prompt_off": "-", "env_denied": "--", "assist_denied": (0, (6, 2)), "prose_only": "-.",
              "prose_redirect": (0, (3, 1, 1, 1)), "no_code": "-", "tool_on": ":", "plain_cot": (0, (5, 1))}
COND_MARK = {"prompt_off": "o", "env_denied": "s", "assist_denied": "P", "prose_only": "v", "prose_redirect": "X",
             "no_code": "D", "tool_on": "^", "plain_cot": "*"}
# conditions get hues of their own (reference palette slots 7, 8, 4, 6, validated as a set) so a
# per-model panel never reuses a model color
COND_COLOR = {"prompt_off": "#6250d6", "env_denied": "#e34948", "assist_denied": "#1c5cab", "prose_only": "#e87ba4",
              "prose_redirect": "#52514e", "no_code": "#008300", "tool_on": "#eda100", "plain_cot": "#9a9890"}
ROLE_STYLE = {"base": "-", "text_rl": "--", "tool_rl": ":"}
GRAY = "#52514e"
# default arms of the headline figure: every model asked for code and denied it, plus the base with the tool
MAIN_ARMS = [("tool_rl", "env_denied"), ("text_rl", "env_denied"), ("base", "env_denied"), ("base", "tool_on")]


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
    if "dataset" not in df.columns:
        df["dataset"] = Path(results_dir).name
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
            pass1_boxed=(round(float(g["correct_boxed"].astype(float).mean()), 3)
                         if "correct_boxed" in g and g["correct_boxed"].notna().all() else float("nan")),
            trunc_pct=round(100 * g["truncated"].mean(), 1),
            no_boxed_pct=round(100 * (~g["boxed_present"]).mean(), 1),
            code_pct=round(100 * g["code_emitted"].mean(), 1),
            fake_out_pct=round(100 * (g["fake_outputs"] > 0).mean(), 1),
            called_pct=round(100 * (g["n_calls"] > 0).mean(), 1),
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


def calls_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tool-call behaviour per arm.

    called_pct          % of samples with at least one intercepted python block
    calls_total         all calls in the arm (sum over samples)
    calls_per_caller    mean calls among the samples that called at all
    denied_total        calls answered with a denial message
    retry_after_denial  mean further calls after the first denial, among samples denied at least once.
                        Once a call is denied every later call in that sample is denied too (prompt_off
                        and env_denied deny all, tool_on denies everything past the budget), so this is
                        n_denied - 1.
    retry_pct           % of denied samples that called again anyway
    calls=k             number of samples that made exactly k calls (5 and more pooled)
    """
    rows = []
    for (tag, cond), g in df.groupby(["tag", "condition"]):
        callers = g[g["n_calls"] > 0]
        denied = g[g["n_denied"] > 0]
        row = dict(tag=tag, condition=cond, samples=len(g),
                   called_pct=round(100 * (g["n_calls"] > 0).mean(), 1),
                   calls_total=int(g["n_calls"].sum()),
                   calls_per_caller=round(float(callers["n_calls"].mean()), 2) if len(callers) else float("nan"),
                   denied_total=int(g["n_denied"].sum()),
                   retry_after_denial=round(float((denied["n_denied"] - 1).mean()), 2) if len(denied) else float("nan"),
                   retry_pct=round(100 * float((denied["n_denied"] >= 2).mean()), 1) if len(denied) else float("nan"))
        hist = g["n_calls"].clip(upper=5).value_counts()
        for k in range(6):
            row["calls>=5" if k == 5 else f"calls={k}"] = int(hist.get(k, 0))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_r"] = out["tag"].map({t: i for i, t in enumerate(ROLE_ORDER)}).fillna(9)
    out["_c"] = out["condition"].map({c: i for i, c in enumerate(COND_ORDER)}).fillna(9)
    return out.sort_values(["_c", "_r"]).drop(columns=["_r", "_c"]).reset_index(drop=True)


def file_stamp() -> str:
    """Local date and time for a figure filename, e.g. 2026-10-14_1530."""
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def arm_times(results_dir):
    """Local datetime per *.jsonl in results_dir: the arm's meta finished_at, else the file's mtime."""
    out = {}
    for p in sorted(Path(results_dir).glob("*.jsonl")) if Path(results_dir).is_dir() else []:
        t = None
        m = p.with_suffix(".meta.json")
        if m.exists():
            try:
                t = json.loads(m.read_text()).get("finished_at")
            except (json.JSONDecodeError, OSError):
                t = None
        out[p.name] = (datetime.fromisoformat(t) if t else datetime.fromtimestamp(p.stat().st_mtime)).astimezone()
    return out


def data_stamp(results_dir) -> str:
    """Stamp for output files: when the newest arm in results_dir finished, local time, e.g.
    2026-10-15_1518. Regenerating from unchanged data gives the same name; new arms give a new one."""
    times = arm_times(results_dir)
    return max(times.values()).strftime("%Y-%m-%d_%H%M") if times else file_stamp()


def provenance(results_dir) -> str:
    """One-line figure caption: when the arms in results_dir were generated (each arm's meta
    finished_at, else the file's modification time), how many there are, and when the plot was made."""
    times = list(arm_times(results_dir).values())
    paths = times
    if not times:
        return ""
    lo, hi = min(times), max(times)
    span = lo.strftime("%Y-%m-%d %H:%M")
    if hi - lo >= timedelta(minutes=1):
        span += " to " + hi.strftime("%Y-%m-%d %H:%M")
    return f"{len(paths)} arms generated {span}; plotted {datetime.now():%Y-%m-%d %H:%M}"


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


def plot(df: pd.DataFrame, lines, title=None, color="model", ci=True, iters=300, save=None, caption=None):
    """Draw pass@k curves.

    lines  a list of "model condition" strings for one panel, or a dict {panel title: [lines]} for
           several panels sharing the y axis. Models: base, text_rl, tool_rl. Conditions: prompt_off,
           env_denied, tool_on, plain_cot. "tool_rl/prompt_off" and ("tool_rl", "prompt_off") also work.
    color  "model" or "condition": what the color encodes. The other one sets the line style, but only
           when it varies within a panel.
    ci     shade a 95% bootstrap band over problems.
    save   path of a PNG to write (use file_stamp() in the name to date it).
    caption small gray line at the bottom, e.g. provenance(results_dir).
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
    if caption:
        fig.text(0.01, 0.004, caption, fontsize=7, color=GRAY, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
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
    """Readout for one condition (same prompt and harness for all three models), base first.

    Primary, erosion: is tool_rl below its own untrained base? That comparison has no recipe
    confound, so it is the clean test of "lost what it had". Secondary, tool-specific: is
    tool_rl below text_rl, the RL model trained without the tool? That is the counterfactual,
    and it is only as fair as the pair is matched (use a same-recipe pair, e.g. VerlTool)."""
    pk = passk_table(df)
    sub = pk[pk.condition == cond]
    tags = set(sub.tag)
    if not {"base", "tool_rl"} <= tags:
        print(f"[{cond}] need at least base and tool_rl; have {sorted(tags)}")
        return
    n = int(sub.n_samples.min())
    kmax = ks_for(n)[-1]
    print(f"[{cond}] {sub.n_problems.min()} problems, n={n}, kmax={kmax}")
    pairs = [("tool_rl", "base")] + ([("text_rl", "base"), ("tool_rl", "text_rl")] if "text_rl" in tags else [])
    res = {}
    for a, b in pairs:
        for k in (1, kmax):
            m, lo, hi = paired_diff(df, (a, cond), (b, cond), k, iters=iters)
            res[(a, b, k)] = (m, lo, hi)
            sig = "" if lo <= 0 <= hi else "  *"
            print(f"  pass@{k:<4} {a:>7} - {b:<7} = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{sig}")
    g = df[df.condition == cond].groupby("tag")
    er = (g["arith_wrong"].sum() / g["arith_n"].sum() * 100).round(2)
    print("  arithmetic error rate (%):", er.to_dict())
    below_base = any(res[("tool_rl", "base", k)][2] < 0 for k in (1, kmax))
    above_base = any(res[("tool_rl", "base", k)][1] > 0 for k in (1, kmax))
    if cond == "tool_on":
        print("  -> positive control (both with the tool): "
              + ("tool_rl above base: tool training worked" if above_base else
                 "tool_rl NOT above base with the tool: training did not take, tool-off results are hard to read"))
        return
    print("  -> erosion test (vs untrained base): "
          + ("tool_rl BELOW its base: erosion" if below_base else
             "tool_rl above its base: no erosion" if above_base else "tool_rl level with its base: no erosion, no gain"))
    if "text_rl" in tags:
        d1, dk = res[("tool_rl", "text_rl", 1)], res[("tool_rl", "text_rl", kmax)]
        if d1[2] < 0 or dk[2] < 0:
            msg = "tool_rl below text_rl: a tool-specific cost relative to RL without the tool"
        elif d1[1] > 0 or dk[1] > 0:
            msg = "tool_rl above text_rl: tool training helped tool-free reasoning (internalization)"
        else:
            msg = "tool_rl and text_rl overlap at every k: no tool-specific effect detected"
        print("  -> tool-specific test (vs text_rl): " + msg)
