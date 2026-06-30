"""Scatter of exam performance vs. estimated cost, one point per model+arm.

Each dot is one (model, arm) combination — haiku/opus/sonnet/gpt crossed with
baseline / agent / agent+skills. Marker *shape* encodes the model and marker
*colour* encodes the arm.

* **x — Estimated cost (USD)**: the expected dollar cost to sit the exam once, a
  price-weighted sum of the four token components, each at its own
  per-model-family rate (so Anthropic and OpenAI sit on one dollar axis). Token
  usage is answer-level, so we dedup to one row per answer, cost each answer,
  average over epochs, and sum the per-question means.
* **y — Performance**: the point-weighted exam score (marks earned / max),
  computed via :func:`exam_common.point_weighted_totals`.

The canary question (``EXCLUDED_QUESTION``, q06_misc) is dropped from both axes,
so performance and cost both describe the modeling questions q01–q05. The judge
panel is pinned to the ``notools`` variant and averaged over the two judge
providers.

Adapted from the exam repo's ``analysis/plot_performance_vs_cost.py`` to read
data via :mod:`defaults` and write the figure alongside this script.

Run: ``python figures/performance_vs_cost.py``
"""

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import defaults  # noqa: E402
from exam_common import (  # noqa: E402
    _TOKEN_COMPONENTS,
    _prepare,
    _token_prices,
    load_scores,
    point_weighted_totals,
    question_points,
)

# The canary question, excluded from both axes: performance counts marks for the
# modeling questions (q01–q05) only, and cost is summed over the same questions so
# the two axes describe the same body of work. (The source exam repo had a stale
# "q05_misc" id here that matched nothing; the canary is q06_misc.)
EXCLUDED_QUESTION = "q06_misc"

# Answer identity (independent of which judge scored it).
_ANSWER_KEYS = ["log", "task", "model", "condition", "agent", "question", "epoch", "topic"]

# Arms that correspond to a real model run on the exam. ``gold`` is the
# model-independent judge-calibration arm and is not a model+environment combo.
_REAL_CONDITIONS = {"baseline", "agent", "agent_skills"}

# Marker per model (keyed by the short model name).
_MARKERS = {
    "claude-haiku-4-5": (3, 2, 0),    # three-spoked asterisk
    "claude-sonnet-4-6": "+",
    "claude-opus-4-6": (6, 2, 0),     # six-spoked asterisk
    "claude-opus-4-8": (8, 2, 0),     # eight-spoked asterisk
    "gpt-5.4-mini": "o",
    "gpt-5.5": "s",
}

# Legend/draw order for models (smallest/oldest → largest/newest, gpt last).
_MODEL_ORDER = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "gpt-5.4-mini",
    "gpt-5.5",
]


def _real_arms(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only real model+environment arms (drop mockllm and the gold arm)."""
    df = df[df["condition"].isin(_REAL_CONDITIONS)]
    df = df[~df["model"].str.startswith("mockllm")]
    return df


def _performance(df: pd.DataFrame) -> pd.DataFrame:
    """Point-weighted performance per (model, arm), excluding ``EXCLUDED_QUESTION``.

    Pins to the ``notools`` judge variant, point-weights over the remaining
    questions, then averages over judge providers and epochs to one value per
    (model, arm).
    """
    judges = df[df["judge"].notna()].copy()
    if "notools" in set(judges["judge_variant"].dropna()):
        judges = judges[judges["judge_variant"] == "notools"]
    scored = judges[judges["question"] != EXCLUDED_QUESTION]
    totals = point_weighted_totals(scored, question_points())
    return (
        totals.groupby(["model", "arm"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "performance"})
    )


def _cost(df: pd.DataFrame) -> pd.DataFrame:
    """Expected USD cost to sit the exam once per (model, arm), excluding q05.

    Token usage is answer-level, so dedup to one row per answer, cost each
    answer, take the mean per question (over epochs), then sum those means.
    """
    keys = [k for k in _ANSWER_KEYS if k in df.columns] + ["arm"]
    answers = df.drop_duplicates(subset=keys).copy()
    answers = answers[answers["question"] != EXCLUDED_QUESTION]

    components = list(_TOKEN_COMPONENTS)
    prices = answers["model"].map(_token_prices)
    unpriced = sorted(answers.loc[prices.isna(), "model"].unique())
    if unpriced:
        print(f"Warning: no price table entry for {unpriced}; their cost shows as 0.")

    def _rate(col: str) -> pd.Series:
        return prices.map(lambda p: (p.get(col, 0.0) if isinstance(p, dict) else 0.0) / 1e6)

    answers["cost"] = sum(
        pd.to_numeric(answers[col], errors="coerce").fillna(0) * _rate(col) for col in components
    )
    per_q = answers.groupby(["model", "arm", "question"], as_index=False)["cost"].mean()
    return per_q.groupby(["model", "arm"], as_index=False)["cost"].sum()


def plot_performance_vs_cost(df: pd.DataFrame) -> None:
    """Scatter: x = cost, y = performance, shape = model, colour = arm."""
    df = _real_arms(df)
    perf = _performance(df)
    cost = _cost(df)
    pts = perf.merge(cost, on=["model", "arm"], how="inner")
    if pts.empty:
        print("Skipping performance-vs-cost plot: no real model arms found.")
        return

    pts["model_short"] = pts["model"].str.split("/").str[-1]
    if "cost" not in pts.columns or pts["cost"].fillna(0).eq(0).all():
        print("Skipping performance-vs-cost plot: no token counts recorded.")
        return

    arm_order = ["baseline", "agent", "agent+skills"]
    arms = [a for a in arm_order if a in set(pts["arm"])] + sorted(
        set(pts["arm"]) - set(arm_order)
    )
    present = set(pts["model_short"].unique())
    models = [m for m in _MODEL_ORDER if m in present] + sorted(present - set(_MODEL_ORDER))
    markers = {m: _MARKERS.get(m, "P") for m in models}
    # Colour encodes arm; marker shape encodes model. We draw the points by hand
    # because seaborn refuses to put line markers (``+``) and filled markers in
    # the same ``style`` mapping. Fixed palette so agent is green and
    # agent+skills is orange regardless of which arms are present.
    palette = {"baseline": "C0", "agent": "C2", "agent+skills": "C1"}
    palette = {a: palette.get(a, f"C{i}") for i, a in enumerate(arms)}

    from matplotlib.lines import Line2D  # noqa: E402

    # The gpt circle/square markers read as much larger than the thin asterisk
    # markers at the same ``s``, so draw them smaller for visual parity.
    def _size(model_short: str) -> float:
        return 80.0 if model_short.startswith("gpt") else 160.0

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, r in pts.iterrows():
        ax.scatter(
            r["cost"],
            r["performance"],
            marker=markers[r["model_short"]],
            color=palette[r["arm"]],
            s=_size(r["model_short"]),
            linewidths=1.8,
            alpha=0.7,
        )
    ax.set_xlabel("Estimated cost (USD)")
    ax.set_ylabel("Performance")
    ax.set_title("Performance vs. estimated cost")
    ax.set_ylim(0, 1)

    # Two legends: colour → arm, marker → model.
    arm_handles = [
        Line2D([], [], marker="o", linestyle="", color=palette[a], label=a) for a in arms
    ]
    model_handles = [
        Line2D([], [], marker=markers[m], linestyle="", color="0.3", label=m) for m in models
    ]
    leg1 = ax.legend(handles=arm_handles, title="arm", loc="lower right", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=model_handles, title="model", loc="lower center", fontsize=8)
    fig.tight_layout()
    out = defaults.OUTPUT_DIR / "performance_vs_cost.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    df = load_scores()
    if df.empty:
        print(f"No scores found at {defaults.SCORES}. Run the eval first, then flatten logs.")
        return
    df = _prepare(df)
    plot_performance_vs_cost(df)


if __name__ == "__main__":
    main()
