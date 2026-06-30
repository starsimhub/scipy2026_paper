"""Shared loaders/helpers for the inspect-ai exam figures (Group A).

Extracted from the exam repo's ``analysis/plot_results.py`` and
``src/exam/points.py`` so the paper's figure scripts are self-contained — they
need only this module plus ``defaults.py``, not an installed ``exam`` package.
Data is read from the exam repo via the paths in :mod:`defaults`.
"""

import json

import pandas as pd

import defaults

# Columns that identify one graded *answer* (independent of which judge scored
# it). Pivoting on these turns the long table into one row per answer with a
# column per judge. ``epoch`` is part of the identity: with ``--epochs N`` each
# question is answered N times per run, and every repeat is its own answer.
_ANSWER_KEYS = ["log", "task", "model", "condition", "agent", "question", "epoch", "topic"]

# Estimated API pricing for the cost figure, in USD per *million* tokens, broken
# out per token component because Anthropic and OpenAI price them differently:
# Anthropic charges a cache-*write* premium and cache reads at 0.1x input,
# whereas OpenAI has no separate cache-write fee and discounts cached input
# reads. So we can't keep one input rate and derive the rest via global
# multipliers: each model family carries its own per-component rate. Keys are
# matched to the model string by family substring (longest match wins); update
# these if pricing changes. ``total_tokens`` is intentionally NOT priced because
# it double-counts the components.
_TOKEN_COMPONENTS = ("input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens")
_TOKEN_PRICES_PER_MTOK = {
    # Anthropic standard API pricing. Opus 4.5+ is cheaper than Opus 4/4.1, so
    # keep explicit version keys rather than a broad ``opus`` fallback.
    "opus-4-8": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4.8": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4-7": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4.7": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4-6": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4.6": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4-5": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4.5": {"input_tokens": 5.0, "output_tokens": 25.0, "cache_write_tokens": 6.25, "cache_read_tokens": 0.5},
    "opus-4-1": {"input_tokens": 15.0, "output_tokens": 75.0, "cache_write_tokens": 18.75, "cache_read_tokens": 1.5},
    "opus-4.1": {"input_tokens": 15.0, "output_tokens": 75.0, "cache_write_tokens": 18.75, "cache_read_tokens": 1.5},
    "sonnet": {"input_tokens": 3.0, "output_tokens": 15.0, "cache_write_tokens": 3.75, "cache_read_tokens": 0.3},
    "haiku": {"input_tokens": 1.0, "output_tokens": 5.0, "cache_write_tokens": 1.25, "cache_read_tokens": 0.1},
    # OpenAI standard API pricing. There is no cache-write premium; cached input
    # is billed as cached reads, and any recorded cache-write component is priced
    # at zero to avoid double-counting.
    "gpt-5.5": {"input_tokens": 5.0, "output_tokens": 30.0, "cache_write_tokens": 0.0, "cache_read_tokens": 0.5},
    "gpt-5.4-mini": {"input_tokens": 0.75, "output_tokens": 4.5, "cache_write_tokens": 0.0, "cache_read_tokens": 0.075},
    "gpt-5.4": {"input_tokens": 2.5, "output_tokens": 15.0, "cache_write_tokens": 0.0, "cache_read_tokens": 0.25},
}

# Order judge variants by how much execution signal each gives the judge.
_VARIANT_ORDER = ["notools", "output", "tools"]

# Matches the rubric footer line, e.g. "... is clearly met. Total: 68 points."
import re  # noqa: E402

_TOTAL_RE = re.compile(r"Total:\s*([0-9]+)\s*points", re.IGNORECASE)

# Columns identifying one *exam sitting* graded by one judge (see exam.points).
_SITTING_KEYS = [
    "task",
    "model",
    "condition",
    "agent",
    "arm",
    "judge",
    "judge_variant",
    "epoch",
]


def _token_prices(model: str) -> dict[str, float] | None:
    """Per-component USD/million-token rates for ``model``, matched by substring.

    Tries the longest family key first so a specific entry (``gpt-5.4-mini``)
    wins over a generic one (``gpt``). Returns ``None`` for an unrecognised model
    so the caller can warn rather than silently price it at zero.
    """
    m = str(model).lower()
    for family in sorted(_TOKEN_PRICES_PER_MTOK, key=len, reverse=True):
        if family in m:
            return _TOKEN_PRICES_PER_MTOK[family]
    return None


def load_scores() -> pd.DataFrame:
    """Load ``scores.jsonl`` into a DataFrame, or an empty frame if absent."""
    if not defaults.SCORES.exists():
        return pd.DataFrame()
    with defaults.SCORES.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def _arm_label(row: pd.Series) -> str:
    """Human-readable experimental arm: baseline / agent / agent+skills."""
    condition = row.get("condition")
    if condition == "baseline":
        return "baseline"
    if condition == "agent_skills":
        return "agent+skills"
    if condition == "agent":
        return "agent"
    return str(condition)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add display helpers: ``judge`` / ``judge_variant`` and a combined ``arm``.

    The flattened ``scores.jsonl`` already carries ``judge`` (provider) and
    ``judge_variant`` columns, so we use them directly.
    """
    df = df.copy()
    if "judge" not in df.columns or "judge_variant" not in df.columns:
        raise ValueError(
            "scores.jsonl lacks 'judge'/'judge_variant' columns; re-run exam-flatten."
        )
    # The experimental arm: baseline, or the specific agent with/without skills.
    df["arm"] = df.apply(_arm_label, axis=1)
    return df


def _default_variant(judges: pd.DataFrame) -> str | None:
    """The variant the per-judge figures default to (prefer ``notools``)."""
    present = set(judges["judge_variant"].dropna())
    if not present:
        return None
    return "notools" if "notools" in present else sorted(present)[0]


def question_points(questions_dir=None) -> dict[str, int]:
    """Map each question id to its rubric point total (``q03_sirs`` -> 68).

    Reads the ``Total: N points.`` line from every ``questions/<id>/rubric.md``.
    A question whose rubric omits that line is skipped (callers fall back to a
    weight of 1 and warn), so a malformed rubric degrades to equal weighting
    rather than crashing the figure build.
    """
    if questions_dir is None:
        questions_dir = defaults.QUESTIONS_DIR
    points: dict[str, int] = {}
    for rubric in sorted(questions_dir.glob("*/rubric.md")):
        match = _TOTAL_RE.search(rubric.read_text())
        if match:
            points[rubric.parent.name] = int(match.group(1))
    return points


def point_weighted_totals(df: pd.DataFrame, points: dict[str, int] | None = None) -> pd.DataFrame:
    """Collapse per-question rubric rows into one point-weighted total per sitting.

    Given the long rubric frame (one row per question×judge×sitting, ``score`` in
    [0, 1]), returns one row per exam sitting (see ``_SITTING_KEYS``) with
    ``score`` set to the point-weighted mean of its questions —
    ``sum(score_q * points_q) / sum(points_q)`` over the questions actually
    present in that sitting. Questions missing from a run are simply left out of
    both sums, so a partial run is weighted over the questions it did answer.

    A ``n_questions`` column records how many questions backed each total.
    """
    if df.empty:
        return df

    if points is None:
        points = question_points()

    work = df.copy()
    work["_w"] = work["question"].map(points).astype("float")
    unknown = sorted(work.loc[work["_w"].isna(), "question"].unique())
    if unknown:
        print(f"Warning: no rubric point total for {unknown}; weighting them equally (1).")
    work["_w"] = work["_w"].fillna(1.0)

    keys = [k for k in _SITTING_KEYS if k in work.columns]
    # Collapse to one score per (sitting, question) first, so a question that
    # appears in more than one log for the same sitting contributes once.
    per_q = work.groupby(keys + ["question"], dropna=False).agg(
        score=("score", "mean"), _w=("_w", "first")
    ).reset_index()
    per_q["_ws"] = per_q["score"] * per_q["_w"]

    grouped = per_q.groupby(keys, dropna=False)
    out = grouped.agg(
        _ws=("_ws", "sum"),
        _w=("_w", "sum"),
        n_questions=("question", "nunique"),
    ).reset_index()
    out["score"] = out["_ws"] / out["_w"]
    return out.drop(columns=["_ws", "_w"])
