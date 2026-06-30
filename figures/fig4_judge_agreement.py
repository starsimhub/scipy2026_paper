"""Judge-agreement scatter for the inspect-ai exam.

Every answer is graded by a two-provider judge panel (``rubric_judge_anthropic`` /
``rubric_judge_openai``); this plots one judge's score against the other for every
shared answer, annotating Pearson r (agreement) and the mean signed gap
(systematic bias). Adapted from the exam repo's
``analysis/plot_results.py:plot_judge_agreement`` to read data via
:mod:`defaults` and write the figure alongside this script.

Run: ``python figures/fig4_judge_agreement.py``
"""

import matplotlib
import numpy as np
import sciris as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

import defaults
import utils


def plot_judge_agreement(df):
    """Scatter one judge's score against the other for every shared answer.

    Annotates Pearson r (agreement) and the mean signed gap (systematic bias:
    positive ⇒ the x-axis judge scores higher on average). Skips cleanly unless
    exactly two judges are present.
    """
    judges = sorted(df["judge"].unique())
    if len(judges) != 2:
        print(f"Skipping judge-agreement plot: need exactly 2 judges, found {judges or 'none'}.")
        return

    jx, jy = judges
    # Keep ``arm`` (for hue) and fill NA in index columns with a sentinel —
    # pivot_table groups on the index and silently drops rows with NaN keys
    # (e.g. baseline answers, where ``agent`` is None).
    keys = [k for k in utils._ANSWER_KEYS if k in df.columns] + ["arm"]
    work = df.copy()
    work[keys] = work[keys].fillna("∅")
    wide = (
        work.pivot_table(index=keys, columns="judge", values="score")
        .dropna(subset=[jx, jy])
        .reset_index()
    )
    if wide.empty:
        print("Skipping judge-agreement plot: no answers scored by both judges.")
        return

    if "model" in wide.columns:
        wide["model_provider"] = wide["model"].str.split("/").str[0]
    else:
        wide["model_provider"] = "unknown"
    # Friendly provider names for the "Model" legend.
    wide["Model"] = wide["model_provider"].replace({"anthropic": "Claude", "openai": "GPT"})

    # Canonical arm colours (keyed by raw exam arm name via EXAM_ARM_MAP):
    # chat → black, no skills → blue, skills → orange, nudged → green.
    arm_palette = {arm: defaults.ARM_COLORS[defaults.EXAM_ARM_MAP.get(arm, arm)]
                   for arm in wide["arm"].unique()}

    fig, ax = plt.subplots(figsize=(6, 6))
    # Markers swapped relative to seaborn's default order (Claude → X, GPT → o).
    sns.scatterplot(
        data=wide, x=jx, y=jy, hue="arm", style="Model", palette=arm_palette,
        markers={"Claude": "X", "GPT": "o"}, alpha=0.6, ax=ax,
    )
    ref_line = ax.plot([0, 1], [0, 1], ls=":", c="0.3", lw=1.6)[0]

    # Thin black line of best fit, labelled with slope and R² in the legend.
    slope, intercept = np.polyfit(wide[jx], wide[jy], 1)
    r2 = wide[jx].corr(wide[jy]) ** 2
    xfit = np.array([0, 1])
    fit_line = ax.plot(xfit, slope * xfit + intercept, c="black", lw=1, zorder=5)[0]

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel(f"{jx.capitalize()} judge score")
    ax.set_ylabel("OpenAI judge score")
    ax.set_title("Judge self-preference")
    sc.boxoff(ax=ax)

    # Pull the arm colours seaborn assigned, then rebuild the legend by hand as two
    # blocks (Configuration / Model) with bold titles. Configuration entries are
    # drawn as dots rather than the style markers used in the scatter.
    auto = ax.get_legend()
    section, arm_colors = None, {}
    for h, t in zip(auto.legend_handles, auto.get_texts()):
        lab = t.get_text()
        if lab in ("arm", "Model"):
            section = lab
        elif section == "arm":
            arm_colors[lab] = h.get_color()
    auto.remove()

    # Map the raw exam arm names (baseline / agent / agent+skills) to the canonical
    # display labels in defaults (baseline → chat → "Chat only", etc.).
    def _arm_label(arm):
        return defaults.ARM_LABELS[defaults.EXAM_ARM_MAP.get(arm, arm)]

    cfg_handles = [Line2D([], [], marker="o", linestyle="none", color=c, label=_arm_label(lab))
                   for lab, c in arm_colors.items()]
    model_handles = [
        Line2D([], [], marker="X", linestyle="none", color="0.2", label="Claude"),
        Line2D([], [], marker="o", linestyle="none", color="0.2", label="GPT"),
        ref_line, fit_line,
    ]
    ref_line.set_label("Perfect agreement")
    fit_line.set_label(f"Best fit (slope = {slope:.2f}, $R^2$ = {r2:.2f})")

    leg_cfg = ax.legend(handles=cfg_handles, title="Configuration", loc="lower right",
                        fontsize=8)
    ax.add_artist(leg_cfg)
    ax.legend(handles=model_handles, title="Model", loc="upper left", fontsize=8)
    fig.tight_layout()
    out = defaults.OUTPUT_DIR / "fig4_judge_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    df = utils.load_exam_scores()
    if df.empty:
        print(f"No scores found at {defaults.SCORES}. Run the eval first, then flatten logs.")
        return
    df = utils._prepare(df)
    # The rubric figures compare *judge* scores; exclude the objective
    # ``code_execution`` scorer (judge is None — different value scale, not a
    # judge) so it doesn't break the agreement plot.
    judges = df[df["judge"].notna()]
    # The per-judge agreement figure compares providers, not variants, so pin it
    # to a single variant (default notools) — otherwise the pivot would average
    # each answer across variants.
    default = utils._default_variant(judges)
    base = judges[judges["judge_variant"] == default] if default else judges
    plot_judge_agreement(base)


if __name__ == "__main__":
    main()
