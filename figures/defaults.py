"""Shared paths for the paper's figure scripts.

Every figure script reads its data from the companion exam repository rather than
re-deriving it here, so the single source of truth for *where that repo lives* is
``RESULTSDIR`` below. Point this at a different checkout and all the figure
scripts follow.

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
