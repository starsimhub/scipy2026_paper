"""Shared paths and plotting style for the paper's figure scripts.

Every figure script reads its data from the companion exam repository rather than
re-deriving it here, so the single source of truth for *where that repo lives* is
``RESULTSDIR`` below. Point this at a different checkout and all the figure
scripts follow. This module also holds the shared per-model marker styling (see
the bottom of the file) so the model-comparison figures stay consistent.

``RESULTSDIR`` is the literal ``../scipy2026_starsim_exam`` resolved relative to
this paper repo's root (the parent of this ``figures/`` directory), so it works
no matter what directory the scripts are launched from. The derived paths
(``SCORES``, ``QUESTIONS_DIR``, ``VALIDATION_ANSWERS``) are the specific inputs
the figures consume; ``OUTPUT_DIR`` is where the generated PNGs are written
(alongside these scripts).
"""

from pathlib import Path

# This ``figures/`` directory, and the paper repo root one level up.
FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent

# The companion exam repo, as a sibling of this paper repo. Kept as the literal
# ``../scipy2026_starsim_exam`` (resolved against the repo root) so the location
# is easy to spot and override.
RESULTSDIR = (REPO_ROOT / "../scipy2026_starsim_exam").resolve()

# ── derived inputs the figures read ──────────────────────────────────────────
# Group A (inspect-ai exam) data.
SCORES = RESULTSDIR / "results" / "scores.jsonl"
QUESTIONS_DIR = RESULTSDIR / "questions"
# Group B (validation runs) data.
VALIDATION_ANSWERS = RESULTSDIR / "validation" / "answers"

# Where generated figures are written (this directory).
OUTPUT_DIR = FIGURES_DIR


# ── question taxonomy ─────────────────────────────────────────────────────────
# Q1-Q5 measure modeling skill and are the headline; Q6 is the "canary" question
# (reported separately, not part of the modeling total).
MODELING_QIDS = ["q01", "q02", "q03", "q04", "q05"]
CANARY_QID = "q06"
ALL_QIDS = MODELING_QIDS + [CANARY_QID]


# ── run-condition (config / arm) conventions ──────────────────────────────────
# The current three-arm preset is noskills / skills / nudged; "full" is the
# legacy label for the same tools as "skills" (web + plugin) from earlier runs
# and is kept only so any old run still parses (it sorts last).
CONFIG_ORDER = ["noskills", "skills", "nudged", "full"]
CONFIG_COLORS = {
    "noskills": "#4C72B0",
    "skills": "#DD8452",
    "nudged": "#55A868",
    "full": "#DD8452",
}
# Short display names (used by fig5 / lost_marks). "full" aliases "skills".
CONFIG_LABELS = {
    "noskills": "No skills",
    "skills": "Skills",
    "nudged": "Skills + nudged",
    "full": "Skills",
}
# Longer "Agent …" display names (used by fig3 and the combined cost-vs-score
# figure, which extends these with a "chat" arm).
ARM_LABELS = {
    "noskills": "Agent (no skills)",
    "skills": "Agent + skills",
    "nudged": "Agent + skills + nudged",
    "full": "Agent + skills",  # legacy alias for skills
}
# Config labels whose runs have the starsim-ai plugin/skills loaded.
PLUGIN_CONFIGS = ["skills", "nudged", "full"]

# Categorical ordering for the validation data's model / effort columns.
MODEL_ORDER = ["haiku", "sonnet", "opus"]
EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max"]


# ── shared per-model marker styling ───────────────────────────────────────────
# Single source of truth for marker shape, size, edge width, line style, and draw
# order, so ``fig3_effort_vs_score`` and ``combined_cost_vs_score`` stay visually
# consistent (e.g. the sonnet:opus size ratio is defined once here).
#
# Sizes are stored as marker *diameters* in points. Line-marker callers
# (``plot`` / ``errorbar`` ``markersize``) use the diameter directly; ``scatter``
# callers want an *area* and should go through :func:`marker_area`. Each figure
# may pass its own ``scale`` so the two plots can share ratios while differing in
# absolute size.

# Marker draw / legend order: gpt models first, then anthropic small → large.
# (Distinct from MODEL_ORDER above, which is the 3-model validation categorical
# order; this set includes the gpt models that appear in the combined figure.)
MODEL_MARKER_ORDER = ["gpt-mini", "gpt-5.5", "haiku", "sonnet", "opus"]

# Marker shape per model: mathtext glyphs for sonnet/opus (so they take the
# series colour), a 3-spoked star for haiku, open circle/square for the gpts.
MARKERS = {
    "haiku": (3, 2, 0),
    "sonnet": "$×$",
    "opus": "$❋$",
    "gpt-mini": "o",
    "gpt-5.5": "s",
}

# Models drawn as outlines (facecolor "none") so a series colour shows on the
# edge; the line-stroke markers (star / glyphs) are unfilled already.
OPEN_MARKERS = {"gpt-mini", "gpt-5.5"}

# Line style per model (fig3's connecting lines / legend).
MODEL_LINESTYLE = {"sonnet": "--", "opus": "-"}

# Marker edge width (stroke) per model. The ×/❋ glyphs read light at 0.5; the
# haiku star needs a heavier stroke, and the gpt outlines a clearly visible one.
MARKER_EDGEWIDTH = {"haiku": 1.8, "sonnet": 0.5, "opus": 0.5, "gpt-mini": 1.6, "gpt-5.5": 1.6}

# Marker diameter (points) per model. sonnet:opus ≈ 0.72; the ❋ glyph renders
# small, so opus is sized up for visual parity.
MARKER_DIAMETER = {"haiku": 12.86, "sonnet": 11.5, "opus": 16.0, "gpt-mini": 8.78, "gpt-5.5": 8.78}


def marker_area(model, scale=1.0):
    """``scatter`` size (pt^2) for ``model``: ``(diameter * scale)`` squared."""
    return (MARKER_DIAMETER[model] * scale) ** 2
