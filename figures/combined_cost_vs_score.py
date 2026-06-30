"""Combined cost-vs-score figure: validation runs + inspect-ai exam sittings.

Pools the two cost-vs-score views onto one plot, using fig4a as the base:

* **fig4a** (Group B, ``validation_common``) — per-run validation results.
* **performance_vs_cost** (Group A, ``exam_common``) — inspect-ai exam sittings
  from ``scores.jsonl``.

Both axes describe the Q1–Q5 modeling questions only (the q06 canary is dropped):
x = estimated cost (USD) to answer Q1–Q5, y = Q1–Q5 modeling score (%). Exam
performance, natively in [0, 1], is scaled to a percentage to match fig4a.

Encodings: marker **shape** = model, **colour** = arm. Unlike the source figures
this draws *every raw data point* (one per validation run / per exam sitting) at
``alpha=0.7`` rather than an aggregate ± uncertainty.

Arm vocabulary is unified across the two experiments so the same condition shares
a colour:

* exam ``baseline``     → ``chat`` (new — direct chat, no agent)
* exam ``agent``        → ``noskills``
* exam ``agent+skills`` → ``skills``
* validation ``noskills`` / ``skills`` / ``nudged`` keep their names.

Models shown: haiku, sonnet, opus (4.8), gpt-mini (gpt-5.4-mini), gpt-5.5. Opus
4.6 is dropped. Marker shapes reuse fig3's glyphs for sonnet/opus.

Run: ``python figures/combined_cost_vs_score.py``
"""

from __future__ import annotations

import matplotlib
import pandas as pd
import sciris as sc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import defaults  # noqa: E402
import validation_common as C  # noqa: E402
from exam_common import (  # noqa: E402
    _ANSWER_KEYS,
    _TOKEN_COMPONENTS,
    _prepare,
    _token_prices,
    load_scores,
    point_weighted_totals,
    question_points,
)

# The q06 canary, excluded from both axes so cost and score describe q01–q05.
EXCLUDED_QUESTION = "q06_misc"

# Exam conditions that are a real model+environment arm (drop the gold/judge arm).
_REAL_CONDITIONS = {"baseline", "agent", "agent_skills"}

# Normalise exam model strings (after stripping any "provider/" prefix) to the
# short keys used here. ``claude-opus-4-6`` is intentionally absent so it drops.
EXAM_MODEL_MAP = {
    "claude-haiku-4-5": "haiku",
    "claude-sonnet-4-6": "sonnet",
    "claude-opus-4-8": "opus",
    "gpt-5.4-mini": "gpt-mini",
    "gpt-5.5": "gpt-5.5",
}

# Map exam arms onto the unified arm vocabulary (baseline becomes the new "chat").
EXAM_ARM_MAP = {"baseline": "chat", "agent": "noskills", "agent+skills": "skills"}

# Marker shape / size / edge width come from the shared defaults module
# (sonnet/opus glyphs, haiku star, gpt open circle/square). This figure's scatter
# markers are drawn a touch smaller than fig3's; SCALE keeps the same ratios.
MARKER_SCALE = 0.948

# Colour per arm. The three validation configs reuse their fig4a colours; "chat"
# (exam baseline) is a new purple.
ARM_COLORS = {
    "chat": "black",
    "noskills": C.CONFIG_COLORS["noskills"],
    "skills": C.CONFIG_COLORS["skills"],
    "nudged": C.CONFIG_COLORS["nudged"],
}
ARM_ORDER = ["chat", "noskills", "skills", "nudged"]
# Display names for the four unified arms: the three agent configs share the
# centralized ARM_LABELS; "chat" (exam baseline) is unique to this figure.
ARM_LABELS = {"chat": "Chat only", **C.ARM_LABELS}


def _validation_points() -> pd.DataFrame:
    """Raw validation runs: one row per run, Q1–Q5 cost vs Q1–Q5 score (%)."""
    rt = C.run_totals()
    rt = rt[rt.model.isin(["haiku", "sonnet", "opus"])]
    arm = rt["config"].astype(str).replace({"full": "skills"})  # legacy alias
    out = pd.DataFrame(
        {
            "model": rt["model"].astype(str),
            "arm": arm,
            "cost": rt["modeling_cost_usd"],
            "score_pct": rt["modeling_pct"],
            "source": "validation",
        }
    )
    return out.dropna(subset=["cost", "score_pct"])


def _exam_points() -> pd.DataFrame:
    """Raw exam sittings: one row per (model, arm, epoch), Q1–Q5 cost vs score (%)."""
    df = load_scores()
    if df.empty:
        return pd.DataFrame(columns=["model", "arm", "cost", "score_pct", "source"])
    df = _prepare(df)
    df = df[df["condition"].isin(_REAL_CONDITIONS)]
    df = df[~df["model"].str.startswith("mockllm")]

    has_epoch = "epoch" in df.columns
    gkeys = ["model", "arm"] + (["epoch"] if has_epoch else [])

    # Performance per sitting: pin the notools judge variant, point-weight over
    # q01–q05, then average over judge providers within each (model, arm, epoch).
    judges = df[df["judge"].notna()].copy()
    if "notools" in set(judges["judge_variant"].dropna()):
        judges = judges[judges["judge_variant"] == "notools"]
    scored = judges[judges["question"] != EXCLUDED_QUESTION]
    totals = point_weighted_totals(scored, question_points())
    perf = totals.groupby(gkeys, as_index=False)["score"].mean()

    # Cost per sitting: dedup to one row per answer, cost each, sum over q01–q05.
    keys = [k for k in _ANSWER_KEYS if k in df.columns] + ["arm"]
    answers = df.drop_duplicates(subset=keys).copy()
    answers = answers[answers["question"] != EXCLUDED_QUESTION]
    prices = answers["model"].map(_token_prices)

    def _rate(col: str) -> pd.Series:
        return prices.map(lambda p: (p.get(col, 0.0) if isinstance(p, dict) else 0.0) / 1e6)

    answers["cost"] = sum(
        pd.to_numeric(answers[col], errors="coerce").fillna(0) * _rate(col)
        for col in _TOKEN_COMPONENTS
    )
    per_q = answers.groupby(gkeys + ["question"], as_index=False)["cost"].mean()
    cost = per_q.groupby(gkeys, as_index=False)["cost"].sum()

    pts = perf.merge(cost, on=gkeys, how="inner")
    pts["model"] = pts["model"].str.split("/").str[-1].map(EXAM_MODEL_MAP)
    pts = pts[pts["model"].notna()]  # drops opus-4-6 and any unmapped model
    pts["arm"] = pts["arm"].map(lambda a: EXAM_ARM_MAP.get(a, a))
    out = pd.DataFrame(
        {
            "model": pts["model"],
            "arm": pts["arm"],
            "cost": pts["cost"],
            "score_pct": pts["score"] * 100.0,
            "source": "exam",
        }
    )
    return out.dropna(subset=["cost", "score_pct"])


def plot_combined(points: pd.DataFrame) -> None:
    """Scatter every raw point: x = cost, y = score (%), shape = model, colour = arm."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for (model, arm), g in points.groupby(["model", "arm"]):
        color = ARM_COLORS.get(arm, "0.5")
        size = defaults.marker_area(model, MARKER_SCALE)
        marker = defaults.MARKERS[model]
        open_ = model in defaults.OPEN_MARKERS
        # Faint half-size raw points behind, plus an opaque full-size group mean.
        raw = dict(marker=marker, alpha=0.2, zorder=2)
        mean = dict(marker=marker, alpha=1.0, zorder=3)
        mx, my = g["cost"].mean(), g["score_pct"].mean()
        lw = defaults.MARKER_EDGEWIDTH[model]
        if open_:
            ax.scatter(g["cost"], g["score_pct"], s=size / 2, facecolors="none",
                       edgecolors=color, linewidths=lw, **raw)
            ax.scatter(mx, my, s=size, facecolors="none", edgecolors=color,
                       linewidths=lw, **mean)
        else:
            # Per-model edge width (fig3 uses 0.5 for the ×/❋ glyphs; the haiku
            # star is heavier); scatter's default linewidth would fatten them.
            ax.scatter(g["cost"], g["score_pct"], s=size / 2, color=color,
                       linewidths=lw, **raw)
            ax.scatter(mx, my, s=size, color=color, linewidths=lw, **mean)

    ax.set_xlabel("Cost (USD)")
    ax.set_ylabel("Exam score (%)")
    ax.set_title("Performance vs. cost", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_xlim(left=-0.2)  # a hair below 0 so markers at $0 aren't clipped
    ax.set_ylim(bottom=50, top=101.5)  # a hair above 100 so markers at 100% aren't clipped
    sc.boxoff(ax=ax)

    present_models = set(points["model"])
    present_arms = set(points["arm"])
    arm_handles = [
        Line2D([], [], marker="o", linestyle="none", color=ARM_COLORS[a], markersize=9,
               label=ARM_LABELS[a])
        for a in ARM_ORDER if a in present_arms
    ]
    model_handles = [
        Line2D([], [], marker=defaults.MARKERS[m], linestyle="none", color="0.3",
               markerfacecolor=("none" if m in defaults.OPEN_MARKERS else "0.3"),
               markeredgewidth=defaults.MARKER_EDGEWIDTH[m],
               markersize=defaults.MARKER_DIAMETER[m] * MARKER_SCALE, label=m)
        for m in defaults.MODEL_ORDER if m in present_models
    ]
    leg1 = ax.legend(handles=arm_handles, title="Configuration", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="Model", loc="lower center", fontsize=9,
              labelspacing=1.0, handletextpad=0.8, borderpad=0.8)

    fig.tight_layout()
    out = C.RESULTS_DIR / "combined_cost_vs_score.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    points = pd.concat([_validation_points(), _exam_points()], ignore_index=True)
    if points.empty:
        print("No data found for the combined cost-vs-score figure.")
        return
    plot_combined(points)


if __name__ == "__main__":
    main()
