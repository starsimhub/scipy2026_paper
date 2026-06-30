"""Judge-agreement scatter for the inspect-ai exam.

Every answer is graded by a two-provider judge panel (``rubric_judge_anthropic`` /
``rubric_judge_openai``); this plots one judge's score against the other for every
shared answer, annotating Pearson r (agreement) and the mean signed gap
(systematic bias). Adapted from the exam repo's
``analysis/plot_results.py:plot_judge_agreement`` to read data via
:mod:`defaults` and write the figure alongside this script.

Run: ``python figures/judge_agreement.py``
"""

import matplotlib
import sciris as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import defaults  # noqa: E402
from exam_common import _ANSWER_KEYS, _default_variant, _prepare, load_scores  # noqa: E402


def plot_judge_agreement(df) -> None:
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
    keys = [k for k in _ANSWER_KEYS if k in df.columns] + ["arm"]
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

    fig, ax = plt.subplots(figsize=(6, 6))
    # Markers swapped relative to seaborn's default order (Claude → X, GPT → o).
    sns.scatterplot(
        data=wide, x=jx, y=jy, hue="arm", style="Model",
        markers={"Claude": "X", "GPT": "o"}, alpha=0.6, ax=ax,
    )
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="Perfect agreement")

    # Thin black line of best fit, labelled with slope and R².
    import numpy as np

    slope, intercept = np.polyfit(wide[jx], wide[jy], 1)
    r2 = wide[jx].corr(wide[jy]) ** 2
    xfit = np.array([0, 1])
    ax.plot(xfit, slope * xfit + intercept, c="black", lw=1, zorder=5)
    # Square axes with equal data ranges, so the data slope equals the visual angle.
    xm = 0.65
    ax.text(xm, slope * xm + intercept, f"  slope = {slope:.2f}, $R^2$ = {r2:.2f}",
            rotation=np.degrees(np.arctan(slope)), rotation_mode="anchor",
            ha="left", va="bottom", fontsize=8)

    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel(f"{jx.capitalize()} judge score")
    ax.set_ylabel("OpenAI judge score")
    ax.set_title("Judge self-preference")
    sc.boxoff(ax=ax)
    ax.legend(loc="lower right", fontsize=8)
    # Rename the seaborn legend section headers.
    for txt in ax.get_legend().get_texts():
        if txt.get_text() == "arm":
            txt.set_text("Configuration")
    fig.tight_layout()
    out = defaults.OUTPUT_DIR / "judge_agreement.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    df = load_scores()
    if df.empty:
        print(f"No scores found at {defaults.SCORES}. Run the eval first, then flatten logs.")
        return
    df = _prepare(df)
    # The rubric figures compare *judge* scores; exclude the objective
    # ``code_execution`` scorer (judge is None — different value scale, not a
    # judge) so it doesn't break the agreement plot.
    judges = df[df["judge"].notna()]
    # The per-judge agreement figure compares providers, not variants, so pin it
    # to a single variant (default notools) — otherwise the pivot would average
    # each answer across variants.
    default = _default_variant(judges)
    base = judges[judges["judge_variant"] == default] if default else judges
    plot_judge_agreement(base)


if __name__ == "__main__":
    main()
