"""Figure 5 (utilization) — fraction of relevant starsim-ai skill uses realized.

How often did the agents *actually* invoke the vendored ``starsim-ai`` skills,
relative to how often they could have? Each skill invocation shows up in a run's
``answerNN.log`` transcript as a ``TOOL_USE · Skill`` event naming the skill.

Each question runs as its own isolated agent session, so the natural unit is
binary "used in this (run, question)?". A skill is *relevant* to a question if it
was invoked there in at least one full run; potential is one use per run.
Utilization = realized / potential, summed over the relevant (question, skill)
pairs. The organic ``skills`` arm and the ``nudged`` arm are scored against a
*common* relevant set (the union of pairs either arm invoked) so their bars share
a per-question denominator.

This is the *utilization* panel only, extracted from the exam repo's
``validation/analysis/fig5_skill_usage.py`` (the per-question / per-skill usage
panels are not copied), modified to read data via :mod:`defaults` and write the
figure alongside this script.

Run: ``python figures/fig5_utilization.py``
"""

from __future__ import annotations

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import validation_common as C  # noqa: E402

# A Skill tool call in the transcript: the ``TOOL_USE · Skill`` marker followed by
# its JSON payload, from which we pull the ``"skill": "<name>"`` field.
SKILL_RE = re.compile(r'TOOL_USE · Skill\s*\n\s*\{[^}]*?"skill":\s*"([^"]+)"', re.DOTALL)
# answerNN.log → qid (answer01 → q01); the answer ids line up 1:1 with questions.
ANSWER_LOG_RE = re.compile(r"answer(\d+)\.log$")


def load_skill_invocations() -> pd.DataFrame:
    """One row per skill invocation across every run: (run, config, qid, skill)."""
    rows = []
    for d in C.run_dirs():
        manifest = C._read_yaml(d / "manifest.yaml")
        config = manifest.get("config")
        for log_path in sorted(d.glob("answer*.log")):
            m = ANSWER_LOG_RE.search(log_path.name)
            if not m:
                continue
            qid = f"q{m.group(1)}"
            text = log_path.read_text(errors="replace")
            for skill in SKILL_RE.findall(text):
                rows.append(
                    {
                        "run": d.name,
                        "model": manifest.get("model"),
                        "effort": manifest.get("effort"),
                        "config": config,
                        "qid": qid,
                        "skill": skill,
                    }
                )
    return pd.DataFrame(rows, columns=["run", "model", "effort", "config", "qid", "skill"])


def skill_utilization(full: pd.DataFrame, n_runs: int, relevant: set | None = None) -> pd.DataFrame:
    """Per-question utilization of the relevant skills.

    Returns one row per question with the realized/potential counts and the
    utilization fraction; a skill is "relevant" to a question if it was invoked
    there in >=1 run, and its potential is one use per run.

    Pass ``relevant`` (a set of ``(qid, skill)`` pairs) to score this arm against a
    *fixed* relevant set instead of the one derived from ``full`` itself — used to
    give the skills and nudged arms a common denominator. Pairs in ``relevant``
    that this arm never used count as 0 realized.
    """
    # Binary "used in this (run, question, skill)?" — re-invocation within a run
    # doesn't count (the skill is already loaded into context).
    pres = full.groupby(["run", "qid", "skill"]).size().clip(upper=1).reset_index(name="used")
    # Relevant (qid, skill) pairs and how many runs actually used each.
    pair = pres.groupby(["qid", "skill"])["used"].sum().reset_index(name="runs_used")
    if relevant is not None:
        # Restrict/expand to the supplied relevant set; unused pairs get 0 runs.
        idx = pd.MultiIndex.from_tuples(sorted(relevant), names=["qid", "skill"])
        pair = pair.set_index(["qid", "skill"]).reindex(idx, fill_value=0).reset_index()

    rows = []
    for qid in C.ALL_QIDS:
        sub = pair[pair["qid"] == qid]
        n_relevant = len(sub)
        realized = int(sub["runs_used"].sum())
        potential = n_relevant * n_runs
        rows.append(
            {
                "qid": qid,
                "n_relevant_skills": n_relevant,
                "realized": realized,
                "potential": potential,
                "utilization": (realized / potential) if potential else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    inv = load_skill_invocations()

    # Only starsim-ai skills are of interest; filter to be explicit.
    inv = inv[inv["skill"].str.startswith("starsim-ai:")]

    # Sanity check: noskills runs must never invoke a starsim-ai skill.
    n_noskills = int((inv["config"] == "noskills").sum())
    assert n_noskills == 0, f"expected 0 starsim-ai invocations in noskills runs, got {n_noskills}"

    # Number of runs per config, taken from the manifests so a plugin-enabled run
    # that happened to invoke zero skills still counts in the denominator.
    runs_per_config = Counter(
        C._read_yaml(d / "manifest.yaml").get("config") for d in C.run_dirs()
    )

    # Organic plugin runs: skills loaded, no prompt steer (configs "skills"/"full").
    organic_cfgs = [c for c in C.PLUGIN_CONFIGS if c != "nudged" and runs_per_config.get(c, 0)]
    organic = inv[inv["config"].isin(organic_cfgs)]
    n_organic_runs = sum(runs_per_config.get(c, 0) for c in organic_cfgs)

    nudged = inv[inv["config"] == "nudged"]
    n_nudged_runs = runs_per_config.get("nudged", 0)

    # Both arms are scored against a *common* relevant set — the union of the
    # (q, skill) pairs either arm ever invoked — so the bars share a denominator
    # per question and read as "of the skills found relevant by either arm, how
    # often did this arm reach for them".
    relevant = set(zip(organic["qid"], organic["skill"])) | set(zip(nudged["qid"], nudged["skill"]))
    skills_util = skill_utilization(organic, n_organic_runs, relevant)
    skills_overall = skills_util["realized"].sum() / (skills_util["potential"].sum() or float("nan"))
    arm_util = {"skills": (skills_util, skills_overall, n_organic_runs)}

    if n_nudged_runs:
        nud_util = skill_utilization(nudged, n_nudged_runs, relevant)
        nr, npot = int(nud_util["realized"].sum()), int(nud_util["potential"].sum())
        nud_overall = nr / npot if npot else float("nan")
        arm_util["nudged"] = (nud_util, nud_overall, n_nudged_runs)

    # ── utilization figure ──────────────────────────────────────────────────────
    # Grouped bars per arm (skills vs nudged).
    util_arms = [a for a in ("skills", "nudged") if a in arm_util]
    util_by_qid = {a: arm_util[a][0].set_index("qid") for a in util_arms}
    # Questions with nonzero potential in at least one arm, in canonical order.
    qids = [q for q in C.ALL_QIDS
            if any(util_by_qid[a].loc[q, "potential"] for a in util_arms)]

    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(util_arms)
    width = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    x = np.arange(len(qids))
    for i, a in enumerate(util_arms):
        ub = util_by_qid[a]
        pct = [100 * ub.loc[q, "utilization"] if ub.loc[q, "potential"] else 0 for q in qids]
        ax.bar(x + offsets[i], pct, width, color=C.CONFIG_COLORS[a], edgecolor="black",
               label=f"{a} — overall {100 * arm_util[a][1]:.0f}%")
        # Per-arm overall reference line in the same colour.
        ax.axhline(100 * arm_util[a][1], color=C.CONFIG_COLORS[a], linestyle="--",
                   linewidth=1.2, alpha=0.8)
        for xi, q in zip(x + offsets[i], qids):
            r = ub.loc[q]
            if r["potential"]:
                ax.text(xi, 100 * r["utilization"], f"{int(r['realized'])}/{int(r['potential'])}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(qids)
    ax.set_xlabel("question")
    ax.set_ylabel("utilization of relevant skills (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Skill utilization: realized vs. potential uses, by plugin arm\n"
                 "(common denominator: skills relevant to either arm; potential = 1 use/run)",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, title="arm")
    fig.tight_layout()
    out = C.RESULTS_DIR / "fig5_skill_utilization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
