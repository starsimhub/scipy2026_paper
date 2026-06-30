"""Shared per-model marker styling for the model-comparison figures.

Single source of truth for marker shape, size, edge width, line style, and draw
order, so ``fig3_effort_vs_score`` and ``combined_cost_vs_score`` stay visually
consistent (e.g. the sonnet:opus size ratio is defined once here).

Sizes are stored as marker **diameters** in points. Line-marker callers
(``plot`` / ``errorbar`` ``markersize``) use the diameter directly; ``scatter``
callers want an **area** and should go through :func:`area`. Each figure may pass
its own ``scale`` so the two plots can share ratios while differing in absolute
size.
"""

from __future__ import annotations

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


def area(model: str, scale: float = 1.0) -> float:
    """``scatter`` size (pt^2) for ``model``: ``(diameter * scale)`` squared."""
    return (MARKER_DIAMETER[model] * scale) ** 2
