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

    r = wide[jx].corr(wide[jy])  # pandas Pearson — no scipy needed
    bias = (wide[jx] - wide[jy]).mean()

    if "model" in wide.columns:
        wide["model_provider"] = wide["model"].str.split("/").str[0]
    else:
        wide["model_provider"] = "unknown"

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(
        data=wide, x=jx, y=jy, hue="arm", style="model_provider", alpha=0.7, ax=ax
    )
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1, label="perfect agreement")
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.set_xlabel(f"{jx} judge score")
    ax.set_ylabel(f"{jy} judge score")
    ax.set_title(
        f"Judge agreement (n={len(wide)})\n"
        f"Pearson r = {r:.2f}   mean gap ({jx} − {jy}) = {bias:+.3f}"
    )
    ax.legend(loc="lower right", fontsize=8)
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
