"""pass@k, failure modes, tool-call behaviour, figures and the verdict for one dataset of a run.

    import analyze as an
    df = an.load_results("<run>/aime_all")           # graded rows (src/grade.py)
    an.summary(df)                                    # one row per (model, condition)
    an.calls_table(df)                                # who called the tool, how often, retries after a denial
    an.plot(df, ["tool_rl no_code", "base no_code"])  # pass@k curves; a dict {title: [lines]} gives panels
    an.verdict(df, "no_code")                         # tool_rl vs base first (atrophy), then vs text_rl

Works on any condition names, so runs made with older conditions still load and plot.
"""
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_ORDER = ["base", "tool_rl", "text_rl"]
# dataviz reference palette, categorical slots 1-3 (CVD-validated): blue, orange, aqua
ROLE_COLOR = {"base": "#2a78d6", "tool_rl": "#eb6834", "text_rl": "#1baf7a"}
ROLE_STYLE = {"base": "-", "tool_rl": ":", "text_rl": "--"}
COND_ORDER = ["tool_on", "env_denied", "no_code", "plain"]
# conditions get hues of their own (slots 4, 8, 6 of the same palette) so a per-model panel never reuses a
# model color; conditions from older runs keep the colors they were plotted with
COND_COLOR = {"tool_on": "#eda100", "env_denied": "#e34948", "no_code": "#008300", "plain": "#6250d6",
              "prompt_off": "#6250d6", "assist_denied": "#1c5cab", "prose_only": "#e87ba4", "prose_redirect": "#52514e"}
COND_STYLE = {"tool_on": ":", "env_denied": "--", "no_code": "-", "plain": "-."}
COND_MARK = {"tool_on": "^", "env_denied": "s", "no_code": "D", "plain": "o"}
GRAY = "#52514e"
# unfenced code in a response (no_code leaks it: the ban stops fences, not every way of spelling code)
CODE_LIKE = re.compile(r"^\s*(python\s*$|import |from \S+ import|def \w+\(|print\(|for \w+ in range\()|\\text\{import",
                       re.M | re.I)


def cond_rank(cond):
    return COND_ORDER.index(cond) if cond in COND_ORDER else len(COND_ORDER)


def role_rank(tag):
    return ROLE_ORDER.index(tag) if tag in ROLE_ORDER else len(ROLE_ORDER)


def load_results(results_dir, allow_ungraded=False) -> pd.DataFrame:
    """Every *.jsonl in the folder as one DataFrame. Raises if any row is ungraded (gen.py writes them
    ungraded) unless allow_ungraded, since pass@k would silently be wrong."""
    paths = sorted(Path(results_dir).glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no .jsonl under {results_dir}")
    frames = []
    for p in paths:
        f = pd.DataFrame([json.loads(line) for line in open(p) if line.strip()])
        f["file"] = p.name
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["arm"] = df["tag"] + " / " + df["condition"]
    if "dataset" not in df.columns:
        df["dataset"] = Path(results_dir).name
    n_un = int(df["correct"].isna().sum()) if "correct" in df else len(df)
    if n_un and not allow_ungraded:
        raise ValueError(f"{n_un} of {len(df)} rows in {results_dir} are ungraded: run  python src/grade.py "
                         f"{results_dir}  (or src/report.py, which grades first)")
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
    out = pd.DataFrame(rows)
    out["_c"], out["_r"] = out.condition.map(cond_rank), out.tag.map(role_rank)
    return out.sort_values(["_c", "condition", "_r"]).drop(columns=["_c", "_r"]).reset_index(drop=True)


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


def paired_diff(df: pd.DataFrame, a: tuple, b: tuple, k: int, iters: int = 2000, seed: int = 0, full=False):
    """Bootstrap CI of pass@k(a) - pass@k(b), paired by problem; a, b = (tag, condition); each arm uses
    its own sample count. Returns (mean, lo, hi), or with full=True (mean, lo, hi, p, n_problems) where p is
    a two-sided bootstrap p-value. Problems missing from either arm are dropped (n_problems says how many
    were paired)."""
    pp = per_problem(df)
    A = pp[(pp.tag == a[0]) & (pp.condition == a[1])].set_index("problem_id")
    B = pp[(pp.tag == b[0]) & (pp.condition == b[1])].set_index("problem_id")
    ids = A.index.intersection(B.index)
    if len(ids) == 0:
        return (float("nan"),) * (5 if full else 3)
    da = np.array([pass_at_k(int(A.loc[i, "n"]), int(A.loc[i, "c"]), k) for i in ids])
    db = np.array([pass_at_k(int(B.loc[i, "n"]), int(B.loc[i, "c"]), k) for i in ids])
    d = da - db
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, size=(iters, len(d)), replace=True).mean(axis=1)
    m, lo, hi = float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    if not full:
        return m, lo, hi
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return m, lo, hi, float(p), len(ids)


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table, one row per (model, condition). pass@1 and pass@k at the k every arm supports
    (shared_kmax), failure modes, tool use, arithmetic error rate.
      unfinished_pct   no boxed answer at the end (truncated at the cap, or stopped early)
      early_stop_pct   stopped before the cap without an answer (the VerlTool tool checkpoint does this)
      code_like_pct    response contains unfenced code (the no_code ban blocks fences, not all code)"""
    pk = passk_table(df)
    ks = shared_kmax(df)
    rows = []
    for (tag, cond), g in df.groupby(["tag", "condition"]):
        pkrow = pk[(pk.tag == tag) & (pk.condition == cond)].iloc[0]
        arith_n = int(g["arith_n"].fillna(0).sum())
        boxed = g["boxed_present"].astype(bool)
        rows.append({
            "tag": tag, "condition": cond, "problems": g["problem_id"].nunique(),
            "samples": int(g.groupby("problem_id").size().min()),
            "pass@1": round(pkrow["pass@1"], 3), f"pass@{ks}": round(pkrow[f"pass@{ks}"], 3),
            "pass1_any": (round(float(g["correct_any"].astype(float).mean()), 3)
                          if "correct_any" in g and g["correct_any"].notna().all() else float("nan")),
            "unfinished_pct": round(100 * (~boxed).mean(), 1),
            "trunc_pct": round(100 * g["truncated"].mean(), 1),
            "early_stop_pct": round(100 * (~boxed & ~g["truncated"].astype(bool)).mean(), 1),
            "code_like_pct": round(100 * g["text"].map(lambda t: bool(CODE_LIKE.search(t or ""))).mean(), 1)
                             if "text" in g else float("nan"),
            "called_pct": round(100 * (g["n_calls"] > 0).mean(), 1),
            "denied_mean": round(g["n_denied"].mean(), 2), "exec_err_mean": round(g["n_exec_err"].mean(), 2),
            "arith_err_pct": round(100 * g["arith_wrong"].fillna(0).sum() / arith_n, 2) if arith_n else float("nan"),
            "arith_checked": arith_n, "tokens_mean": round(g["gen_tokens"].mean()),
        })
    out = pd.DataFrame(rows)
    out["_c"], out["_r"] = out.condition.map(cond_rank), out.tag.map(role_rank)
    return out.sort_values(["_c", "condition", "_r"]).drop(columns=["_c", "_r"]).reset_index(drop=True)


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
        row = dict(tag=tag, condition=cond, rows=len(g),
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
    out["_c"], out["_r"] = out.condition.map(cond_rank), out.tag.map(role_rank)
    return out.sort_values(["_c", "condition", "_r"]).drop(columns=["_c", "_r"]).reset_index(drop=True)


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


def _draw_arm(ax, df, pk, tag, cond, color, style, label, ci, iters, kmax=None):
    """One pass@k curve (plus bootstrap band) for arm (tag, cond), up to kmax. Returns False if absent."""
    row = pk[(pk.tag == tag) & (pk.condition == cond)]
    if row.empty:
        return False
    r = row.iloc[0]
    ks = [k for k in ks_for(int(r["n_samples"])) if kmax is None or k <= kmax]
    ax.plot(ks, [r[f"pass@{k}"] for k in ks], style, color=color, lw=2, marker=COND_MARK.get(cond, "o"), ms=6,
            label=label)
    if ci:
        lo, hi = zip(*[bootstrap_passk(df, tag, cond, k, iters=iters)[1:] for k in ks])
        ax.fill_between(ks, lo, hi, color=color, alpha=0.12, lw=0)
    return True


def _arm(line):
    """'tool_rl no_code', 'tool_rl/no_code' or ('tool_rl', 'no_code') -> ('tool_rl', 'no_code')."""
    if isinstance(line, str):
        parts = line.replace("/", " ").split()
        if len(parts) != 2:
            raise ValueError(f"line must be 'model condition', got {line!r}")
        return parts[0], parts[1]
    tag, cond = line
    return tag, cond


def shared_kmax(df: pd.DataFrame) -> int:
    """Largest k every arm in df supports (the smallest sample count per problem across arms), so curves
    from arms sampled 64 and 16 times are compared over the same range."""
    return ks_for(int(df.groupby(["tag", "condition", "problem_id"]).size().min()))[-1]


def plot(df: pd.DataFrame, lines, title=None, color="model", ci=True, iters=300, save=None, caption=None, kmax="shared"):
    """Draw pass@k curves.

    lines    a list of "model condition" strings for one panel, or a dict {panel title: [lines]} for panels
             sharing the y axis. Models: base, tool_rl, text_rl. Conditions: tool_on, env_denied, no_code.
    color    "model" or "condition": what the color encodes; the other sets the line style when it varies.
    ci       shade a 95% bootstrap band over problems.
    save     path of a PNG to write.   caption  small gray line under the plot, e.g. provenance(dir).
    kmax     x-axis limit: "shared" (default) = the largest k every arm in df supports, an int, or None
             for each arm's own maximum.
    Lines whose arm is not in df are listed and skipped.
    """
    import matplotlib.pyplot as plt
    panels = lines if isinstance(lines, dict) else {None: lines}
    pk = passk_table(df)
    if kmax == "shared":
        kmax = shared_kmax(df)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(7 if n == 1 else 4.6 * n, 4.4), dpi=120, sharey=True, squeeze=False)
    missing = []
    for ax, (name, arms) in zip(axes[0], panels.items()):
        arms = [_arm(a) for a in arms]
        vary_tag, vary_cond = len({t for t, _ in arms}) > 1, len({c for _, c in arms}) > 1
        for tag, cond in arms:
            if color == "condition":
                col, style = COND_COLOR.get(cond, GRAY), (ROLE_STYLE.get(tag, "-") if vary_tag else "-")
            else:
                col, style = ROLE_COLOR.get(tag, GRAY), (COND_STYLE.get(cond, "-") if vary_cond else "-")
            if not _draw_arm(ax, df, pk, tag, cond, col, style, f"{tag} {cond}", ci, iters, kmax):
                missing.append(f"{tag} {cond}")
        _format_axis(ax, min(pk["n_samples"].max(), kmax or 10 ** 9))
        if name:
            ax.set_title(name, loc="left")
        if ax.get_lines():
            ax.legend(frameon=False, fontsize=8, loc="upper left")
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


def best_case(df: pd.DataFrame, cond: str, iters: int = 4000, seed: int = 0) -> pd.DataFrame:
    """pass@1 contrasts on the samples that are reasoning rather than artifacts, paired by problem.
      as run              every sample
      no early stops      drops samples that ended before the cap without an answer (a VerlTool training
                          artifact, 20-31% of the tool checkpoint's samples, 2-4% of base's)
      no early stops, no code   also drops samples with unfenced code (only for code-free conditions)
    The last view is selected by the models themselves, so it is a best-case bound, not an estimate.
    Returns one row per (view, contrast) with mean, 95% bootstrap CI, two-sided p and kept shares."""
    d = df[df.condition == cond].copy()
    if d.empty or "base" not in set(d.tag):
        return pd.DataFrame()
    d["_early"] = ~d.boxed_present.astype(bool) & ~d.truncated.astype(bool)
    d["_code"] = d.text.map(lambda t: bool(CODE_LIKE.search(t or ""))) if "text" in d else False
    views = {"as run": d, "no early stops": d[~d._early]}
    if cond not in ("tool_on", "env_denied"):
        views["no early stops, no code"] = d[~d._early & ~d._code]
    tags = [t for t in ROLE_ORDER if t in set(d.tag)]
    pairs = [(a, "base") for a in tags if a != "base"] + ([("tool_rl", "text_rl")] if {"tool_rl", "text_rl"} <= set(tags) else [])
    rng = np.random.default_rng(seed)
    rows = []
    for vname, v in views.items():
        acc = {t: v[v.tag == t].groupby("problem_id").correct.mean() for t in tags}
        kept = {t: len(v[v.tag == t]) / max(1, len(d[d.tag == t])) for t in tags}
        for a_, b_ in pairs:
            ids = acc[a_].index.intersection(acc[b_].index)
            diff = (acc[a_].loc[ids] - acc[b_].loc[ids]).to_numpy(dtype=float)
            if len(diff) == 0:
                continue
            boots = rng.choice(diff, size=(iters, len(diff)), replace=True).mean(axis=1)
            rows.append(dict(view=vname, contrast=f"{a_} - {b_}", diff=round(float(diff.mean()), 3),
                             lo=round(float(np.percentile(boots, 2.5)), 3), hi=round(float(np.percentile(boots, 97.5)), 3),
                             p=round(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())), 3), problems=len(ids),
                             **{f"pass1_{a_}": round(float(acc[a_].mean()), 3), f"pass1_{b_}": round(float(acc[b_].mean()), 3),
                                f"kept_{a_}": round(kept[a_], 2), f"kept_{b_}": round(kept[b_], 2)}))
    return pd.DataFrame(rows)


def _holm(ps):
    """Holm step-down adjusted p-values for a list of p-values (NaN stays NaN)."""
    idx = [i for i, p in enumerate(ps) if p == p]
    order = sorted(idx, key=lambda i: ps[i])
    adj, running = [float("nan")] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * ps[i]))
        adj[i] = running
    return adj


def _describe(m, lo, hi, p_adj):
    if m != m:
        return "no data"
    if p_adj == p_adj and p_adj < 0.05:
        return "above" if m > 0 else "below"
    return "inconclusive"


def verdict(df: pd.DataFrame, cond: str = "no_code", iters: int = 2000):
    """Readout for one condition (same prompt and harness for every model), base first.

    Contrasts at pass@1 and pass@k (k = the largest every arm supports), paired over problems, 95%
    bootstrap CIs, two-sided bootstrap p-values Holm-adjusted across the contrasts printed here; "*"
    marks adjusted p < 0.05. A difference is called above or below only when that holds; otherwise it is
    inconclusive, with its interval.
      tool_on                 positive control: tool_rl must beat base with the tool, or tool training
                              did not take and the tool-off comparisons cannot be read as atrophy tests.
      any other condition     erosion: tool_rl vs base. Attribution, only if text_rl ran: text_rl vs base
                              and tool_rl vs text_rl, read together (a gap between tool_rl and text_rl
                              with tool_rl level with base is text_rl's gain, not a cost of the tool).
    Returns {(a, b, k): (mean, lo, hi, p_adj, n_problems)}."""
    pk = passk_table(df)
    sub = pk[pk.condition == cond]
    tags = set(sub.tag)
    if not {"base", "tool_rl"} <= tags:
        print(f"[{cond}] need at least base and tool_rl; have {sorted(tags)}")
        return {}
    kmax = shared_kmax(df[df.condition == cond])
    pairs = [("tool_rl", "base")] + ([("text_rl", "base"), ("tool_rl", "text_rl")]
                                     if "text_rl" in tags and cond != "tool_on" else [])
    raw = {(a, b, k): paired_diff(df, (a, cond), (b, cond), k, iters=iters, full=True)
           for a, b in pairs for k in (1, kmax)}
    keys = list(raw)
    adj = dict(zip(keys, _holm([raw[x][3] for x in keys])))
    res = {x: (raw[x][0], raw[x][1], raw[x][2], adj[x], raw[x][4]) for x in keys}
    n_pairs = min(v[4] for v in res.values())
    print(f"[{cond}] {n_pairs} problems paired, k up to {kmax}; Holm-adjusted over {len(keys)} contrasts")
    for (a, b, k), (m, lo, hi, p, _) in res.items():
        star = "  *" if p == p and p < 0.05 else ""
        print(f"  pass@{k:<4} {a:>7} - {b:<7} = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  p_adj={p:.3f}{star}")
    g = df[df.condition == cond].groupby("tag")
    er = (g["arith_wrong"].sum() / g["arith_n"].sum() * 100).round(2)
    print("  arithmetic error rate (%):", er.to_dict())

    def call(a, b):
        d1, dk = _describe(*res[(a, b, 1)][:4]), _describe(*res[(a, b, kmax)][:4])
        return d1, dk

    t1, tk = call("tool_rl", "base")
    if cond == "tool_on":
        ok = "above" in (t1, tk) and "below" not in (t1, tk)
        print("  -> positive control: " + ("tool_rl above base with the tool: tool training worked" if ok else
              f"FAILED (pass@1 {t1}, pass@{kmax} {tk}): tool_rl is not shown above base with the tool, so the "
              "tool-off contrasts cannot be read as tests of atrophy"))
        return res
    if t1 == "above" and tk == "below":
        msg = f"sharpening: tool_rl above base at pass@1 but below at pass@{kmax}"
    elif "below" in (t1, tk):
        msg = f"tool_rl BELOW base (pass@1 {t1}, pass@{kmax} {tk}): erosion"
    elif "above" in (t1, tk):
        msg = f"tool_rl above base (pass@1 {t1}, pass@{kmax} {tk}): no erosion"
    else:
        m, lo, hi = res[("tool_rl", "base", kmax)][:3]
        msg = (f"inconclusive: no difference detected (pass@{kmax} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]); "
               "this is not evidence of no erosion")
    print("  -> erosion test (tool_rl vs untrained base): " + msg)
    if "text_rl" in tags:
        x1, xk = call("text_rl", "base")
        y1, yk = call("tool_rl", "text_rl")
        text_up = "above" in (x1, xk)
        tool_below_text = "below" in (y1, yk)
        if "below" in (t1, tk):
            att = ("text_rl also below base: the drop is RL's, not specific to the tool" if "below" in (x1, xk)
                   else "text_rl not below base: the drop is specific to tool training")
        elif tool_below_text and text_up:
            att = "text_rl rose above base and tool_rl did not: RL without the tool gained, tool RL did not"
        elif tool_below_text:
            att = "tool_rl below text_rl, neither clearly different from base"
        else:
            att = "no difference between tool_rl and text_rl detected"
        print("  -> attribution (vs text_rl): " + att)
    return res
