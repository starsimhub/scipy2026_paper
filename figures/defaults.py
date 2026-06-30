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

# Draw / legend order: gpt models first, then anthropic small → large.
MODEL_ORDER = ["gpt-mini", "gpt-5.5", "haiku", "sonnet", "opus"]

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


def marker_area(model: str, scale: float = 1.0) -> float:
    """``scatter`` size (pt^2) for ``model``: ``(diameter * scale)`` squared."""
    return (MARKER_DIAMETER[model] * scale) ** 2
