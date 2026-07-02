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

import re
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sciris as sc
from matplotlib.lines import Line2D

import defaults
import utils

# A Skill tool call in the transcript: the ``TOOL_USE · Skill`` marker followed by
# its JSON payload, from which we pull the ``"skill": "<name>"`` field.
SKILL_RE = re.compile(r'TOOL_USE · Skill\s*\n\s*\{[^}]*?"skill":\s*"([^"]+)"', re.DOTALL)
# answerNN.log → qid (answer01 → q01); the answer ids line up 1:1 with questions.
ANSWER_LOG_RE = re.compile(r"answer(\d+)\.log$")


def load_skill_invocations():
    """One row per skill invocation across every run: (run, config, qid, skill)."""
    rows = []
    for d in utils.run_dirs():
        manifest = utils._read_yaml(d / "manifest.yaml")
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


def skill_utilization(full, n_runs, relevant=None):
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
    for qid in defaults.ALL_QIDS:
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


def main():
    inv = load_skill_invocations()

    # Only starsim-ai skills are of interest; filter to be explicit.
    inv = inv[inv["skill"].str.startswith("starsim-ai:")]

    # Sanity check: noskills runs must never invoke a starsim-ai skill.
    n_noskills = int((inv["config"] == "noskills").sum())
    assert n_noskills == 0, f"expected 0 starsim-ai invocations in noskills runs, got {n_noskills}"

    # Number of runs per config, taken from the manifests so a plugin-enabled run
    # that happened to invoke zero skills still counts in the denominator.
    manifests = [utils._read_yaml(d / "manifest.yaml") for d in utils.run_dirs()]
    runs_per_config = Counter(m.get("config") for m in manifests)
    # Runs per (config, model, effort) for the by-model/effort utilization panel.
    runs_per_cme = Counter(
        (m.get("config"), m.get("model"), m.get("effort")) for m in manifests
    )

    # Organic plugin runs: skills loaded, no prompt steer (configs "skills"/"full").
    organic_cfgs = [c for c in defaults.PLUGIN_CONFIGS if c != "nudged" and runs_per_config.get(c, 0)]
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

    # ── utilization by model + effort (top panel) ───────────────────────────────
    # Overall utilization split by model and effort, computed separately for the
    # organic (skills) arm and the nudged arm, scored against the same common
    # relevant set used per-question.
    def _util_by_model_effort(arm_inv, cfgs):
        out = {}  # (model, effort) -> overall utilization fraction
        for (model, effort), sub in arm_inv.groupby(["model", "effort"]):
            n_runs = sum(runs_per_cme.get((c, model, effort), 0) for c in cfgs)
            if not n_runs:
                continue
            u = skill_utilization(sub, n_runs, relevant)
            pot = u["potential"].sum()
            out[(model, effort)] = u["realized"].sum() / pot if pot else float("nan")
        return out

    util_me = {
        "skills": _util_by_model_effort(organic, set(organic_cfgs)),
        "nudged": _util_by_model_effort(nudged, {"nudged"}),
    }
    util_me = {a: d for a, d in util_me.items() if d}
    all_me = set().union(*util_me.values())
    me_models = [m for m in defaults.MODEL_ORDER if any(k[0] == m for k in all_me)]
    # (model, effort) groups present for either arm, in canonical order.
    me_groups = [
        (m, e)
        for m in me_models
        for e in defaults.EFFORT_ORDER
        if (m, e) in all_me
    ]

    # ── utilization by question (bottom panel) ───────────────────────────────────
    # Grouped bars per arm (skills vs nudged).
    util_arms = [a for a in ("skills", "nudged") if a in arm_util]
    util_by_qid = {a: arm_util[a][0].set_index("qid") for a in util_arms}
    # Questions with nonzero potential in at least one arm, in canonical order.
    qids = [q for q in defaults.ALL_QIDS
            if any(util_by_qid[a].loc[q, "potential"] for a in util_arms)]

    fig, (axT, ax) = plt.subplots(2, 1, figsize=(9, 10))

    # Top panel: grouped bars, x = (model, effort) group, one bar per arm.
    me_arms = list(util_me)
    na = len(me_arms)
    wT = 0.8 / na
    offT = (np.arange(na) - (na - 1) / 2) * wT
    xT = np.arange(len(me_groups))
    for i, a in enumerate(me_arms):
        pct = [100 * util_me[a].get(g, np.nan) for g in me_groups]
        axT.bar(xT + offT[i], pct, wT, color=defaults.CONFIG_COLORS[a],
                label=defaults.CONFIG_LABELS[a])
    axT.set_xticks(xT)
    axT.set_xticklabels([f"{m}\n{e}" for m, e in me_groups])
    axT.set_xlabel("Model / effort")
    axT.set_ylabel("Utilization (%)")
    axT.set_ylim(0, 100)
    axT.set_title("Skill utilization by model and effort", fontsize=13)
    axT.grid(True, axis="y", alpha=0.3)
    sc.boxoff(ax=axT)
    axT.legend(loc="upper left", fontsize=9, title="Configuration")

    n = len(util_arms)
    width = 0.8 / n
    offsets = (np.arange(n) - (n - 1) / 2) * width
    x = np.arange(len(qids))
    for i, a in enumerate(util_arms):
        ub = util_by_qid[a]
        pct = [100 * ub.loc[q, "utilization"] if ub.loc[q, "potential"] else 0 for q in qids]
        ax.bar(x + offsets[i], pct, width, color=defaults.CONFIG_COLORS[a],
               label=defaults.CONFIG_LABELS[a])
        # Per-arm mean-utilization reference line in the same colour.
        ax.axhline(100 * arm_util[a][1], color=defaults.CONFIG_COLORS[a], linestyle="--",
                   linewidth=1.2, alpha=0.95)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{int(q[1:])}" for q in qids])
    ax.set_xlabel("Question")
    ax.set_ylabel("Utilization (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Skill utilization by question", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    sc.boxoff(ax=ax)
    # Bar entries plus a single neutral entry for the per-arm dotted mean lines.
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color="0.4", linestyle="--", linewidth=1.2))
    labels.append("Mean utilization")
    ax.legend(handles, labels, loc="upper left", fontsize=9, title="Configuration")
    fig.tight_layout()
    out = utils.RESULTS_DIR / "fig5_skill_utilization.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # ── heatmap version: model (x) × question (y), with marginal bar charts ───────
    # A single value per (model, question) cell requires pooling arms, so all
    # plugin-enabled runs (organic skills + nudged) are combined and scored against
    # the same common relevant set. The top bar chart sums utilization per model
    # and the right bar chart sums it per question (each = pooled realized/potential
    # over the marginalised axis), mirroring the two panels above.
    plugin_cfgs = [c for c in defaults.PLUGIN_CONFIGS if runs_per_config.get(c, 0)]
    plugin_inv = inv[inv["config"].isin(plugin_cfgs)]

    def _runs_for_model(model):
        return sum(v for (c, m, _e), v in runs_per_cme.items()
                   if c in plugin_cfgs and m == model)

    hm_models = [m for m in defaults.MODEL_ORDER if (plugin_inv["model"] == m).any()]

    # Per-question marginal (pooled over models) and the question set to show.
    total_runs = sum(_runs_for_model(m) for m in hm_models)
    q_util = skill_utilization(plugin_inv, total_runs, relevant).set_index("qid")
    hm_qids = [q for q in defaults.ALL_QIDS if q_util.loc[q, "potential"]]

    # Heatmap cells + per-model marginal.
    Z = np.full((len(hm_qids), len(hm_models)), np.nan)
    model_overall = {}
    for j, model in enumerate(hm_models):
        sub = plugin_inv[plugin_inv["model"] == model]
        u = skill_utilization(sub, _runs_for_model(model), relevant).set_index("qid")
        for i, q in enumerate(hm_qids):
            if u.loc[q, "potential"]:
                Z[i, j] = 100 * u.loc[q, "utilization"]
        pot = u["potential"].sum()
        model_overall[model] = 100 * u["realized"].sum() / pot if pot else np.nan

    q_pct = [100 * q_util.loc[q, "utilization"] for q in hm_qids]

    fig2, axes = plt.subplots(
        2, 2, figsize=(8, 7),
        gridspec_kw=dict(width_ratios=[4, 1], height_ratios=[1, 4],
                         wspace=0.05, hspace=0.05),
    )
    ax_top, ax_corner = axes[0]
    ax_heat, ax_right = axes[1]
    ax_corner.axis("off")

    im = ax_heat.imshow(Z, aspect="auto", cmap="viridis", vmin=0,
                        vmax=np.nanmax(Z) if np.isfinite(Z).any() else 1)
    ax_heat.set_xticks(range(len(hm_models)))
    ax_heat.set_xticklabels(hm_models)
    ax_heat.set_yticks(range(len(hm_qids)))
    ax_heat.set_yticklabels([f"Q{int(q[1:])}" for q in hm_qids])
    ax_heat.set_xlabel("Model")
    ax_heat.set_ylabel("Question")
    # Annotate each cell with its utilization %.
    for i in range(len(hm_qids)):
        for j in range(len(hm_models)):
            if np.isfinite(Z[i, j]):
                ax_heat.text(j, i, f"{Z[i, j]:.0f}", ha="center", va="center",
                             fontsize=8, color="white" if Z[i, j] < 0.6 * np.nanmax(Z) else "black")

    # Top marginal: per-model overall utilization, aligned to heatmap columns.
    ax_top.bar(range(len(hm_models)), [model_overall[m] for m in hm_models],
               width=0.8, color="0.5")
    ax_top.set_xlim(ax_heat.get_xlim())
    ax_top.set_xticks([])
    ax_top.set_ylabel("Util. (%)", fontsize=9)
    sc.boxoff(ax=ax_top)

    # Right marginal: per-question overall utilization, aligned to heatmap rows.
    ax_right.barh(range(len(hm_qids)), q_pct, height=0.8, color="0.5")
    ax_right.set_ylim(ax_heat.get_ylim())
    ax_right.set_yticks([])
    ax_right.set_xlabel("Util. (%)", fontsize=9)
    sc.boxoff(ax=ax_right)

    fig2.colorbar(im, ax=ax_corner, fraction=0.6, aspect=10, label="Utilization (%)")
    fig2.suptitle("Skill utilization by model and question", fontsize=13)
    out2 = utils.RESULTS_DIR / "fig5_skill_utilization_heatmap.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"wrote {out2}")


if __name__ == "__main__":
    main()
