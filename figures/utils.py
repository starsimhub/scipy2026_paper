"""Shared data loaders for the paper's figure scripts.

Two groups of results live in the companion exam repo (paths in :mod:`defaults`):

- **Group A** (inspect-ai exam) — a flattened ``results/scores.jsonl``. Helpers
  here were extracted from the exam repo's ``analysis/plot_results.py`` and
  ``src/exam/points.py`` so the figures need only this module plus ``defaults``,
  not an installed ``exam`` package.
- **Group B** (validation runs) — one folder per run under
  ``validation/answers/`` with ``manifest.yaml`` (run identity + taker
  tokens/cost/time), ``marking_manifest.yaml`` (per-question scores + cheating
  verdict), and ``marked0N.info`` (rubric section scores).

Shared constants (question taxonomy, config conventions, plotting style) live in
:mod:`defaults`; this module holds only the loaders.
"""

import json
import re

import pandas as pd
import yaml

import defaults

# Outputs land beside these scripts; validation inputs come from the exam repo.
ANSWERS_DIR = defaults.VALIDATION_ANSWERS
RESULTS_DIR = defaults.OUTPUT_DIR


# ══ Group A — inspect-ai exam (scores.jsonl) ══════════════════════════════════

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
_TOTAL_RE = re.compile(r"Total:\s*([0-9]+)\s*points", re.IGNORECASE)

# Columns identifying one *exam sitting* graded by one judge (see exam.points).
_SITTING_KEYS = ["task", "model", "condition", "agent", "arm", "judge", "judge_variant", "epoch"]


def _token_prices(model):
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


def load_exam_scores():
    """Load ``scores.jsonl`` into a DataFrame, or an empty frame if absent."""
    if not defaults.SCORES.exists():
        return pd.DataFrame()
    with defaults.SCORES.open() as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(rows)


def _arm_label(row):
    """Human-readable experimental arm: baseline / agent / agent+skills."""
    condition = row.get("condition")
    if condition == "baseline":
        return "baseline"
    if condition == "agent_skills":
        return "agent+skills"
    if condition == "agent":
        return "agent"
    return str(condition)


def _prepare(df):
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


def _default_variant(judges):
    """The variant the per-judge figures default to (prefer ``notools``)."""
    present = set(judges["judge_variant"].dropna())
    if not present:
        return None
    return "notools" if "notools" in present else sorted(present)[0]


def question_points(questions_dir=None):
    """Map each question id to its rubric point total (``q03_sirs`` -> 68).

    Reads the ``Total: N points.`` line from every ``questions/<id>/rubric.md``.
    A question whose rubric omits that line is skipped (callers fall back to a
    weight of 1 and warn), so a malformed rubric degrades to equal weighting
    rather than crashing the figure build.
    """
    if questions_dir is None:
        questions_dir = defaults.QUESTIONS_DIR
    points = {}
    for rubric in sorted(questions_dir.glob("*/rubric.md")):
        match = _TOTAL_RE.search(rubric.read_text())
        if match:
            points[rubric.parent.name] = int(match.group(1))
    return points


def point_weighted_totals(df, points=None):
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


# ══ Group B — validation runs (validation/answers/) ═══════════════════════════


def present_configs(df):
    """Configs that actually appear in ``df``, in canonical config order."""
    present = set(df["config"].dropna().unique())
    return [c for c in defaults.CONFIG_ORDER if c in present]


def _read_yaml(path):
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def elapsed_to_seconds(human):
    """Parse ``manifest.yaml`` elapsed strings like ``1h09m28s`` / ``21m12s`` / ``48s``."""
    if not human or not isinstance(human, str):
        return None
    total = 0.0
    for value, unit in re.findall(r"(\d+(?:\.\d+)?)([hms])", human):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total or None


def run_dirs():
    """All marked run folders, sorted by slug (chronological)."""
    return sorted(d for d in ANSWERS_DIR.glob("*_*") if (d / "marking_manifest.yaml").exists())


def run_label(model, effort, config):
    """Human-friendly run label, e.g. ``opus · max · full``."""
    return f"{model} · {effort} · {config}"


def load_validation_scores():
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
    return _categoricalize(pd.DataFrame(rows))


def load_sections():
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


def run_totals(scores=None):
    """One row per run: weighted Q1-Q5 modeling score + Q6 canary + summed resources."""
    if scores is None:
        scores = load_validation_scores()
    rows = []
    for run, g in scores.groupby("run", observed=True):
        meta = g.iloc[0]
        modeling = g[g.qid.isin(defaults.MODELING_QIDS)]
        canary = g[g.qid == defaults.CANARY_QID]
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


def run_totals_agg(scores=None):
    """Collapse repeated runs of the same arm into one row with mean ± SE.

    For each numeric metric ``m`` the frame carries ``m`` (mean over reps),
    ``m_se`` (standard error of the mean; NaN for a single rep), and ``m_min`` /
    ``m_max`` (the rep-to-rep range, for min–max whiskers). ``n_runs`` is the
    repetition count per arm.
    """
    if scores is None:
        scores = load_validation_scores()
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


def _categoricalize(df):
    """Apply consistent ordering to model / effort / config columns for plotting."""
    if "model" in df:
        df["model"] = pd.Categorical(df["model"], categories=defaults.MODEL_ORDER, ordered=True)
    if "effort" in df:
        df["effort"] = pd.Categorical(df["effort"], categories=defaults.EFFORT_ORDER, ordered=True)
    if "config" in df:
        df["config"] = pd.Categorical(df["config"], categories=defaults.CONFIG_ORDER, ordered=True)
    return df
