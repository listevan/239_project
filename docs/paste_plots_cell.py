# Paste into one Colab cell after the summary cell. Works with the analyze.py already in the
# notebook. Defines plot(df, lines) and draws the two figures; edit the strings at the bottom.
import sys
sys.path.insert(0, "src")
import analyze as an
import matplotlib.pyplot as plt

df = load_run() if "load_run" in globals() else an.load_results(f"{OUT}/{DATASET}")

COND_COLOR = {"prompt_off": "#6250d6", "env_denied": "#e34948", "tool_on": "#eda100", "plain_cot": "#008300"}
ROLE_STYLE = {"base": "-", "text_rl": "--", "tool_rl": ":"}
GRAY = "#52514e"


def _arm(line):
    if isinstance(line, str):
        line = line.replace("/", " ").split()
    tag, cond = line
    return tag, cond


def _draw_arm(ax, df, pk, tag, cond, color, style, label, ci, iters):
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


def plot(df, lines, title=None, color="model", ci=True, iters=300, save=None):
    """lines: ["model condition", ...] for one panel, or {panel title: [lines]} for several.
    color: "model" or "condition" (what the color means; the other sets line style when it varies)."""
    panels = lines if isinstance(lines, dict) else {None: lines}
    pk = an.passk_table(df)
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
                col, style = an.ROLE_COLOR.get(tag, GRAY), (an.COND_STYLE.get(cond, "-") if vary_cond else "-")
            if not _draw_arm(ax, df, pk, tag, cond, col, style, f"{tag} {cond}", ci, iters):
                missing.append(f"{tag} {cond}")
        ks = an.ks_for(int(pk["n_samples"].max()))
        ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("k (samples)"); ax.set_ylim(0, 1); ax.grid(True, color="#e6e5e1", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if name:
            ax.set_title(name, loc="left")
        if ax.get_lines():
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0][0].set_ylabel("pass@k")
    if title:
        axes[0][0].set_title(title, loc="left") if n == 1 else fig.suptitle(title, x=0.01, ha="left")
    if missing:
        print("not in results, skipped:", ", ".join(missing))
    fig.tight_layout()
    if save:
        fig.savefig(save, bbox_inches="tight")
    return fig


# ---- edit below: each line is "model condition"; models base / text_rl / tool_rl,
# ---- conditions prompt_off / env_denied / tool_on

# Figure 1: within each model, what does turning the tool off cost?
fig1 = plot(df, {
    "base":    ["base prompt_off",    "base env_denied",    "base tool_on"],
    "text_rl": ["text_rl prompt_off", "text_rl env_denied", "text_rl tool_on"],
    "tool_rl": ["tool_rl prompt_off", "tool_rl env_denied", "tool_rl tool_on"],
}, color="condition", title=f"{RUN_NAME}: pass@k within each model", save=f"{OUT}/{DATASET}/passk_by_model.png")

# Figure 2: tool-RL with the tool off, against the text-RL control and the base model
fig2 = plot(df, [
    "tool_rl prompt_off",
    "text_rl prompt_off", "text_rl tool_on",
    "base prompt_off",    "base tool_on",
], title=f"{RUN_NAME}: tool-RL with the tool off vs control and base", save=f"{OUT}/{DATASET}/passk_main.png")
