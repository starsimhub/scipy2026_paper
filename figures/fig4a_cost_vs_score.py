"""Figure 4a — modeling score vs. dollar cost (value-for-money).

Score (weighted Q1-Q5) against the exam's dollar cost, one point per arm with ±SE
in both axes. Marker shape = model, size = reasoning effort, colour = config.

Adapted from the ``cost_vs_score`` panel of the exam repo's
``validation/analysis/fig4_extras.py`` (the other panels of fig4 are not copied)
to read data via :mod:`defaults` and write the figure alongside this script.

Run: ``python figures/fig4a_cost_vs_score.py``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import validation_common as C  # noqa: E402


def cost_vs_score(rt) -> None:
    """``rt`` is the per-arm aggregate (mean over reps); plot ±SE in both axes."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for _, r in rt.iterrows():
        ax.errorbar(
            r["total_cost_usd"],
            r["modeling_pct"],
            xerr=r["total_cost_usd_se"],
            yerr=r["modeling_pct_se"],
            fmt="none",
            ecolor="0.5",
            elinewidth=1.0,
            capsize=2,
            zorder=2,
        )
        ax.scatter(
            r["total_cost_usd"],
            r["modeling_pct"],
            s=C.EFFORT_SIZE[r["effort"]],
            marker=C.MODEL_MARKERS[r["model"]],
            facecolor=C.CONFIG_COLORS[r["config"]],
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
            zorder=3,
        )
    ax.set_xlabel("exam cost (USD, all 6 questions)")
    ax.set_ylabel("Q1–Q5 modeling score (%)")
    ax.set_title("Score vs. cost\n(shape = model, size = effort, colour = config)", fontsize=12)
    ax.grid(True, alpha=0.25)

    handles = [Patch(facecolor=C.CONFIG_COLORS[c], edgecolor="black", label=C.CONFIG_LABELS[c])
               for c in C.present_configs(rt)]
    handles += [Line2D([], [], marker=C.MODEL_MARKERS[m], color="0.3", linestyle="none",
                       markersize=9, label=m) for m in C.MODEL_ORDER]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    fig.tight_layout()
    out = C.RESULTS_DIR / "fig4a_cost_vs_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    cost_vs_score(C.run_totals_agg())


if __name__ == "__main__":
    main()
