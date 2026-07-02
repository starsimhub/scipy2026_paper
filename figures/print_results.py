"""Print manuscript-ready numbers that mirror the figures.

Each ``fig<N>_*`` function prints numbers matching what the corresponding
figure plots, so the manuscript text can quote values consistent with the
figures. Run: ``python figures/print_results.py``
"""

import pandas as pd

import fig1_cost_vs_score as fig1
import fig2_effort_vs_score as fig2
import utils


def fig1_mean_by_model():
    """Fig. 1: mean exam score (%) and cost (USD) per model and config (arm)."""
    points = pd.concat(
        [fig1._validation_points(), fig1._exam_points()], ignore_index=True
    )
    means = points.groupby(["model", "arm"]).agg(
        score_pct=("score_pct", "mean"), cost=("cost", "mean"), n=("cost", "size")
    )
    model_rank = {m: i for i, m in enumerate(fig1.defaults.MODEL_MARKER_ORDER)}
    arm_rank = {a: i for i, a in enumerate(fig1.ARM_ORDER)}
    means = means.reset_index()
    means = means.sort_values(
        by=["model", "arm"],
        key=lambda s: s.map(model_rank if s.name == "model" else arm_rank),
    )

    print("Fig. 1 — mean exam score and cost per model and config (Q1–Q5)")
    print(f"{'model':<10} {'config':<10} {'score (%)':>10} {'cost (USD)':>12} {'n':>5}")
    for _, row in means.iterrows():
        print(f"{row.model:<10} {row.arm:<10} {row.score_pct:>10.1f} "
              f"{row.cost:>12.2f} {int(row.n):>5}")


def fig1_score_transitions():
    """Fig. 1: score change across the chat -> agent -> skills -> nudged ladder."""
    points = pd.concat(
        [fig1._validation_points(), fig1._exam_points()], ignore_index=True
    )
    means = points.groupby(["model", "arm"])["score_pct"].mean()

    # Unified arm ladder; "agent" is "noskills" in the unified vocabulary.
    ladder = ["chat", "noskills", "skills", "nudged"]
    labels = {"chat": "chat", "noskills": "agent", "skills": "skills", "nudged": "nudged"}
    transitions = list(zip(ladder, ladder[1:]))

    print("\nFig. 1 — score change (%) across configuration transitions")
    header = f"{'model':<10}"
    for a, b in transitions:
        header += f"  {labels[a] + '->' + labels[b]:>16}"
    print(header)
    for model in fig1.defaults.MODEL_MARKER_ORDER:
        row = f"{model:<10}"
        for a, b in transitions:
            sa = means.get((model, a))
            sb = means.get((model, b))
            cell = f"{sb - sa:+.1f}" if pd.notna(sa) and pd.notna(sb) else "n/a"
            row += f"  {cell:>16}"
        print(row)


def fig2_effort_vs_score():
    """Fig. 2: mean exam score (%) per (model, config, effort). Means only."""
    rt = utils.run_totals_agg()
    rt = rt[rt.model.isin(["sonnet", "opus"])]
    rt = rt[rt.effort.isin(fig2.EFFORT_AXIS)]

    print("\nFig. 2 — mean exam score (%) by model, config, and effort")
    header = f"{'model':<8} {'config':<10}"
    for e in fig2.EFFORT_AXIS:
        header += f"  {e:>8}"
    print(header)
    for model in ["sonnet", "opus"]:
        sub = rt[rt.model == model]
        for config in utils.present_configs(sub):
            g = sub[sub.config == config].set_index("effort")["modeling_pct"]
            row = f"{model:<8} {config:<10}"
            for e in fig2.EFFORT_AXIS:
                cell = f"{g[e]:.1f}" if e in g.index else "n/a"
                row += f"  {cell:>8}"
            print(row)


def main():
    fig1_mean_by_model()
    fig1_score_transitions()
    fig2_effort_vs_score()


if __name__ == "__main__":
    main()
