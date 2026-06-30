"""Figure 3 — reasoning effort vs. modeling score.

X = reasoning effort (low → max), Y = weighted Q1-Q5 modeling score. One line per
(model, config) for the two models that were swept across effort levels: sonnet
and opus, each in noskills / skills / nudged. (Haiku was run at a single effort
and is omitted here.)

Colour encodes config; line style + marker encode model. Any (model, config,
effort) cell with no data is simply skipped.

Adapted from the exam repo's ``validation/analysis/fig3_effort_vs_score.py`` to
read data via :mod:`defaults` and write the figure alongside this script.

Run: ``python figures/fig3_effort_vs_score.py``
"""

from __future__ import annotations

import matplotlib
import numpy as np
import sciris as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import validation_common as C  # noqa: E402

MODEL_LINESTYLE = {"sonnet": "--", "opus": "-"}

# Marker per model for this figure (overrides the shared MODEL_MARKERS): sonnet
# as a "×", opus as a "❋", both rendered as mathtext glyphs ($...$) so they take
# the config colour. The ❋ glyph renders visually smaller than the × at the same
# point size, so size it up per-model for parity.
MODEL_MARKERS = {"sonnet": "$×$", "opus": "$❋$"}
MODEL_MARKERSIZE = {"sonnet": 10, "opus": 16}
# Thin stroke so the filled glyphs read light rather than bold.
MODEL_MARKEREDGEWIDTH = 0.5

# Effort levels shown on the x-axis. Capped at "high" (the swept runs top out
# there); xhigh/max are dropped rather than left as empty ticks.
EFFORT_AXIS = ["low", "medium", "high"]

# Horizontal dodge so coincident points don't overplot. The two models are
# separated coarsely (sonnet left, opus right); within each model the three arms
# are fanned out on a finer scale so all six (model, config) lines stay legible.
MODEL_DX = {"sonnet": -0.08, "opus": +0.08}
CONFIG_FINE = 0.04


def main() -> None:
    # Per-arm aggregate (mean over reps); one point per (model, config, effort)
    # with [min, max] whiskers over the reps, so a swept line is monotone in
    # effort rather than zig-zagging through the duplicate x-positions.
    rt = C.run_totals_agg()
    rt = rt[rt.model.isin(["sonnet", "opus"])]
    rt = rt[rt.effort.isin(EFFORT_AXIS)]

    x_index = {e: i for i, e in enumerate(EFFORT_AXIS)}
    # Centre the per-arm fine offsets on 0 so the group stays centred on its tick.
    configs = C.present_configs(rt)
    config_dx = {c: (i - (len(configs) - 1) / 2) * CONFIG_FINE for i, c in enumerate(configs)}

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for (model, config), g in rt.groupby(["model", "config"], observed=True):
        g = g.sort_values("effort")
        dx = MODEL_DX[model] + config_dx[config]
        xs = [x_index[e] + dx for e in g["effort"]]
        ys = g["modeling_pct"].to_numpy()
        # Asymmetric whiskers spanning the rep-to-rep [min, max] range.
        yerr = np.vstack([ys - g["modeling_pct_min"].to_numpy(),
                          g["modeling_pct_max"].to_numpy() - ys])
        color = C.CONFIG_COLORS[config]
        # Connecting line drawn separately and faintly, so it guides the eye
        # between effort levels without competing with the markers.
        ax.plot(xs, ys, linestyle=MODEL_LINESTYLE[model], color=color,
                linewidth=2, alpha=0, zorder=1)
        # Markers + thin error bars on top (opaque). "x" / 8-spoked asterisk are
        # unfilled line markers, so their stroke is the edge — keep it the config
        # colour (a black edge would erase the colour encoding) and thicken it so
        # the glyph reads clearly.
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            linestyle="none",
            marker=MODEL_MARKERS[model],
            color=color,
            markersize=MODEL_MARKERSIZE[model],
            markeredgewidth=MODEL_MARKEREDGEWIDTH,
            elinewidth=0.7,
            capsize=3,
            capthick=0.7,
            zorder=2,
            label=f"{model} · {config}",
        )

    ax.set_xticks(range(len(EFFORT_AXIS)))
    ax.set_xticklabels(EFFORT_AXIS)
    ax.set_xlabel("reasoning effort")
    ax.set_ylabel("Q1–Q5 modeling score (%)")
    ax.set_title("Score vs. reasoning effort", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(top=100)
    sc.boxoff(ax=ax)

    # Two compact legends: colour = config, style/marker = model.
    config_handles = [
        Line2D([], [], color=C.CONFIG_COLORS[c], lw=3, label=c) for c in C.present_configs(rt)
    ]
    model_handles = [
        Line2D([], [], color="0.3", linestyle=MODEL_LINESTYLE[m], marker=MODEL_MARKERS[m],
               markersize=MODEL_MARKERSIZE[m], markeredgewidth=MODEL_MARKEREDGEWIDTH, label=m)
        for m in ["sonnet", "opus"]
    ]
    leg1 = ax.legend(handles=config_handles, title="config", loc="lower right", fontsize=9)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="model", loc="lower center", fontsize=9)

    fig.tight_layout()
    out = C.RESULTS_DIR / "fig3_effort_vs_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
