"""Shared loader for the validation-run results (Group B).

Copied from the exam repo's ``validation/analysis/common.py`` and modified so the
input folder (``answers/``) is read from the companion exam repo via
:mod:`defaults`, and figures are written alongside these scripts (``OUTPUT_DIR``)
rather than into the exam repo's ``validation/results/``.

The validation flow stores one folder per run under ``validation/answers/`` (e.g.
``jun20.1817_opus-max-full``). Each folder carries two machine-readable manifests
plus per-question marker files:

- ``manifest.yaml``         — run identity (model / effort / config) and, per
                              question, the tokens / cost / wall-clock the *taker* spent.
- ``marking_manifest.yaml`` — per-question awarded / possible / percentage and the
                              cheating verdict, plus the exam totals, from the *marker*.
- ``marked0N.info``         — per-question rubric *section* scores (id/title/awarded/possible).
"""

from __future__ import annotations

import re

import pandas as pd
import yaml

import defaults

# ── paths ───────────────────────────────────────────────────────────────────
# Inputs come from the companion exam repo; outputs land beside these scripts.
ANSWERS_DIR = defaults.VALIDATION_ANSWERS
RESULTS_DIR = defaults.OUTPUT_DIR

# ── question taxonomy ─────────────────────────────────────────────────────────
# Q1-Q5 measure modeling skill and are the headline; Q6 is the "canary" question
# (reported separately, not part of the modeling total).
MODELING_QIDS = ["q01", "q02", "q03", "q04", "q05"]
CANARY_QID = "q06"
ALL_QIDS = MODELING_QIDS + [CANARY_QID]

# ── plotting conventions (shared across figures) ──────────────────────────────
MODEL_ORDER = ["haiku", "sonnet", "opus"]
MODEL_MARKERS = {"haiku": "o", "sonnet": "s", "opus": "^"}

EFFORT_ORDER = ["low", "medium", "high", "xhigh", "max"]
# Marker areas (pt^2) for the scatter; monotone in reasoning effort.
EFFORT_SIZE = {eff: 70 + i * 90 for i, eff in enumerate(EFFORT_ORDER)}

# Run conditions. The current three-arm preset is noskills / skills / nudged;
# "full" is the legacy label for the same tools as "skills" (web + plugin) from
# earlier runs and is kept only so any old run still parses (it sorts last).
CONFIG_ORDER = ["noskills", "skills", "nudged", "full"]
CONFIG_COLORS = {
    "noskills": "#4C72B0",
    "skills": "#DD8452",
    "nudged": "#55A868",
    "full": "#DD8452",
}
# Display names for the three run arms, used across every figure. Edit here to
# rename an arm everywhere. "full" is the legacy alias for "skills" and shares
# its label.
CONFIG_LABELS = {
    "noskills": "No skills",
    "skills": "Skills",
    "nudged": "Skills + nudged",
    "full": "Skills",
}
# Config labels whose runs have the starsim-ai plugin/skills loaded.
PLUGIN_CONFIGS = ["skills", "nudged", "full"]

# Longer "Agent …" display labels shared by fig3 and the combined cost-vs-score
# figure (the latter extends these with a "chat" arm). fig4a keeps the shorter
# CONFIG_LABELS above.
ARM_LABELS = {
    "noskills": "Agent (no skills)",
    "skills": "Agent + skills",
    "nudged": "Agent + skills + nudged",
    "full": "Agent + skills",  # legacy alias for skills
}


def present_configs(df) -> list[str]:
    """Configs that actually appear in ``df``, in canonical :data:`CONFIG_ORDER`."""
    present = set(df["config"].dropna().unique())
    return [c for c in CONFIG_ORDER if c in present]


def _read_yaml(path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def elapsed_to_seconds(human: str | None) -> float | None:
    """Parse ``manifest.yaml`` elapsed strings like ``1h09m28s`` / ``21m12s`` / ``48s``."""
    if not human or not isinstance(human, str):
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", human):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or None


def run_dirs() -> list:
    """All marked run folders, sorted by slug (chronological)."""
    return sorted(d for d in ANSWERS_DIR.glob("*_*") if (d / "marking_manifest.yaml").exists())


def run_label(model: str, effort: str, config: str) -> str:
    """Human-friendly run label, e.g. ``opus · max · full``."""
    return f"{model} · {effort} · {config}"


def load_scores() -> pd.DataFrame:
    """One row per (run, question): identity + score + tokens/cost/time + cheating."""
    rows = []
    for d in run_dirs():
        manifest = _read_yaml(d / "manifest.yaml")
        marking = _read_yaml(d / "marking_manifest.yaml")

        model = manifest.get("model")
        effort = manifest.get("effort")
        config = manifest.get("config")  # "full" | "noskills"

        # Taker resource use, keyed by qid.
        taker = {q["qid"]: q for q in manifest.get("questions", [])}
        # Marker scores, keyed by qid.
        marks = {q["qid"]: q for q in marking.get("questions", [])}

        for qid in sorted(set(taker) | set(marks)):
            t = taker.get(qid, {})
            m = marks.get(qid, {})
            rows.append(
                {
                    "run": d.name,
                    "slug": manifest.get("slug"),
                    "model": model,
                    "effort": effort,
                    "config": config,
                    "label": run_label(model, effort, config),
                    "qid": qid,
                    "awarded": m.get("total_awarded"),
                    "possible": m.get("total_possible"),
                    "percentage": m.get("percentage"),
                    "cheating": bool(m.get("cheating_detected", False)),
                    "cheat_succeeded": m.get("cheating_n_succeeded", 0),
                    "status": m.get("status") or t.get("status"),
                    "total_tokens": t.get("total_tokens"),
                    "total_cost_usd": t.get("total_cost_usd"),
                    "elapsed_human": t.get("elapsed_human"),
                    "elapsed_seconds": elapsed_to_seconds(t.get("elapsed_human")),
                }
            )
    df = _categoricalize(pd.DataFrame(rows))
    return df


def load_sections() -> pd.DataFrame:
    """One row per (run, question, rubric section) from the ``marked0N.info`` files."""
    rows = []
    for d in run_dirs():
        manifest = _read_yaml(d / "manifest.yaml")
        model, effort, config = (
            manifest.get("model"),
            manifest.get("effort"),
            manifest.get("config"),
        )
        for info_path in sorted(d.glob("marked*.info")):
            info = _read_yaml(info_path)
            qid = info.get("qid")
            for sec in info.get("sections", []) or []:
                possible = sec.get("possible")
                awarded = sec.get("awarded")
                pct = (
                    100.0 * awarded / possible
                    if possible not in (None, 0) and awarded is not None
                    else None
                )
                rows.append(
                    {
                        "run": d.name,
                        "model": model,
                        "effort": effort,
                        "config": config,
                        "label": run_label(model, effort, config),
                        "qid": qid,
                        "section_id": str(sec.get("id")),
                        "section_title": sec.get("title"),
                        "awarded": awarded,
                        "possible": possible,
                        "percentage": pct,
                    }
                )
    return _categoricalize(pd.DataFrame(rows))


def run_totals(scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per run: weighted Q1-Q5 modeling score + Q6 canary + summed resources."""
    if scores is None:
        scores = load_scores()
    rows = []
    for run, g in scores.groupby("run", observed=True):
        meta = g.iloc[0]
        modeling = g[g.qid.isin(MODELING_QIDS)]
        canary = g[g.qid == CANARY_QID]
        poss = modeling["possible"].sum()
        awd = modeling["awarded"].sum()
        rows.append(
            {
                "run": run,
                "model": meta["model"],
                "effort": meta["effort"],
                "config": meta["config"],
                "label": meta["label"],
                # modeling (Q1-Q5) headline
                "modeling_awarded": awd,
                "modeling_possible": poss,
                "modeling_pct": 100.0 * awd / poss if poss else None,
                # canary (Q6)
                "canary_pct": canary["percentage"].mean() if len(canary) else None,
                # resources summed over Q1-Q5 (consistent with the modeling score)
                "modeling_tokens": modeling["total_tokens"].sum(min_count=1),
                "modeling_cost_usd": modeling["total_cost_usd"].sum(min_count=1),
                # resources over the whole exam (all 6)
                "total_tokens": g["total_tokens"].sum(min_count=1),
                "total_cost_usd": g["total_cost_usd"].sum(min_count=1),
                "total_seconds": g["elapsed_seconds"].sum(min_count=1),
                "n_cheating": int(g["cheating"].sum()),
            }
        )
    return _categoricalize(pd.DataFrame(rows))


def run_totals_agg(scores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Collapse repeated runs of the same arm into one row with mean ± SE.

    For each numeric metric ``m`` the frame carries ``m`` (mean over reps),
    ``m_se`` (standard error of the mean; NaN for a single rep), and ``m_min`` /
    ``m_max`` (the rep-to-rep range, for min–max whiskers). ``n_runs`` is the
    repetition count per arm.
    """
    if scores is None:
        scores = load_scores()
    rt = run_totals(scores)
    metrics = ["modeling_pct", "canary_pct", "modeling_tokens", "total_cost_usd", "total_seconds"]
    rows = []
    for (model, effort, config), g in rt.groupby(["model", "effort", "config"], observed=True):
        if g.empty:
            continue
        row = {
            "model": model,
            "effort": effort,
            "config": config,
            "label": run_label(model, effort, config),
            "n_runs": len(g),
        }
        for m in metrics:
            row[m] = g[m].mean()
            row[f"{m}_se"] = g[m].sem()  # std(ddof=1)/sqrt(n); NaN when n==1
            row[f"{m}_min"] = g[m].min()
            row[f"{m}_max"] = g[m].max()
        rows.append(row)
    return _categoricalize(pd.DataFrame(rows))


def _categoricalize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply consistent ordering to model / effort / config columns for plotting."""
    if "model" in df:
        df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)
    if "effort" in df:
        df["effort"] = pd.Categorical(df["effort"], categories=EFFORT_ORDER, ordered=True)
    if "config" in df:
        df["config"] = pd.Categorical(df["config"], categories=CONFIG_ORDER, ordered=True)
    return df
