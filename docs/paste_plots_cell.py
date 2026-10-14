# Paste into one Colab cell after the summary cell. Works with the analyze.py already in the
# notebook (needs passk_table / ks_for / bootstrap_passk). Saves passk_by_model.png and passk_main.png.
import sys, importlib
sys.path.insert(0, "src")
import analyze as an
import matplotlib.pyplot as plt

df = load_run() if "load_run" in globals() else an.load_results(f"{OUT}/{DATASET}")

COND_COLOR = {"prompt_off": "#6250d6", "env_denied": "#e34948", "tool_on": "#eda100", "plain_cot": "#008300"}
MAIN_ARMS = [("tool_rl", "prompt_off"),
             ("text_rl", "prompt_off"), ("text_rl", "tool_on"),
             ("base", "prompt_off"), ("base", "tool_on")]


def _format_axis(ax, n_max):
    ks = an.ks_for(int(n_max))
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (samples)"); ax.set_ylim(0, 1); ax.grid(True, color="#e6e5e1", lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _draw_arm(ax, df, pk, tag, cond, color, style, label, ci=True, iters=300):
    row = pk[(pk.tag == tag) & (pk.condition == cond)]
    if row.empty:
        return False
    r = row.iloc[0]
    ks = an.ks_for(int(r["n_samples"]))
    ax.plot(ks, [r[f"pass@{k}"] for k in ks], style, color=color, lw=2,
            marker=an.COND_MARK.get(cond, "o"), ms=6, label=label)
    if ci:
        lo, hi = zip(*[an.bootstrap_passk(df, tag, cond, k, iters=iters)[1:] for k in ks])
        ax.fill_between(ks, lo, hi, color=color, alpha=0.12, lw=0)
    return True


def plot_passk_by_model(df, title=None):
    pk = an.passk_table(df)
    tags = [t for t in an.ROLE_ORDER if t in set(pk.tag)]
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.2), dpi=120, sharey=True, squeeze=False)
    for ax, tag in zip(axes[0], tags):
        for cond in [c for c in an.COND_ORDER if c in set(pk[pk.tag == tag].condition)]:
            _draw_arm(ax, df, pk, tag, cond, COND_COLOR[cond], "-", cond)
        _format_axis(ax, pk[pk.tag == tag]["n_samples"].max())
        ax.set_title(tag, loc="left"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("pass@k")
    fig.suptitle(title or f"pass@k within each model, {df['problem_id'].nunique()} problems", x=0.01, ha="left")
    fig.tight_layout()
    return fig


def plot_passk_main(df, arms=MAIN_ARMS, title=None):
    pk = an.passk_table(df)
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=120)
    missing = [f"{t}/{c}" for t, c in arms
               if not _draw_arm(ax, df, pk, t, c, an.ROLE_COLOR[t], an.COND_STYLE[c], f"{t} / {c}")]
    if missing:
        print("not in results, skipped:", ", ".join(missing))
    _format_axis(ax, pk["n_samples"].max())
    ax.set_ylabel("pass@k"); ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title or f"pass@k, {df['problem_id'].nunique()} problems", loc="left")
    fig.tight_layout()
    return fig


fig1 = plot_passk_by_model(df, title=f"{RUN_NAME} / {DATASET}: pass@k within each model")
fig1.savefig(f"{OUT}/{DATASET}/passk_by_model.png", bbox_inches="tight")
fig2 = plot_passk_main(df, title=f"{RUN_NAME} / {DATASET}: tool-RL with the tool off vs control and base")
fig2.savefig(f"{OUT}/{DATASET}/passk_main.png", bbox_inches="tight")
