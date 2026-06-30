"""Where are marks lost? — answered-poorly vs. omitted vs. cheating vs. canary.

Every ``marked0N.md`` records the rubric as a checklist of line items::

    - [x] Correct specification of sims: **2/2** — n_agents=10000, ... all correct.
    - [ ] Correct value of beta identified: **0/4** — student reports β ≈ 0.044, ...

This parses those line items, finds the ones that *lost* marks (awarded <
possible), and buckets the marker's stated reason into incorrect / omitted /
off_spec / quality / unclassified, then plots where modeling marks are lost by
config arm and by question.

Adapted from the exam repo's ``validation/analysis/lost_marks.py`` to read data
via :mod:`defaults` and write the figure alongside this script.

Run: ``python figures/lost_marks.py``
Writes ``lost_marks.png`` and prints the full breakdown.
"""

from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import validation_common as C  # noqa: E402

# A rubric line item: "- [x] <criterion>: **N/M** <reason>".
LINE_ITEM = re.compile(
    r"^\s*-\s*\[([ x])\]\s*(?P<crit>.*?):\s*\*\*\s*(?P<aw>\d+)\s*/\s*(?P<poss>\d+)\s*\*\*\s*(?P<reason>.*)$",
    re.M,
)

# Reason → category. Checked in order; first hit wins.
CATEGORY_RULES = [
    # off_spec: reasonable answer that doesn't match the rubric's *specific*
    # expected answer. Dominates the canary's open-ended 6.1.3.
    ("off_spec", [
        "demographics-only", "not demographics", "not 10", "agents, not", "not a 10",
        "no birth rate", "no births", "not a demographics", "exponential population growth",
        "exponential growth", "in the context required", "in the context of",
        "only in the co", "scheme's only awardable", "marking scheme explicitly states",
        "did not provide; \"do not award", "no 10-agent",
    ]),
    # quality / style deductions on otherwise-correct work.
    ("quality", [
        "deducted marks for", "deducted for", "unclear", "not succinct", "succinctness",
        "verbose", "too long", "exceeds", "clarity", "readab", "well-engineered",
        "engineering", "docstring", "minor", "stylistic", "could be more",
    ]),
    # omitted: the required element is absent / never produced / left blank.
    ("omitted", [
        "no plot", "not plotted", "no assert", "no comparison", "no programmatic",
        "no code", "no step", "no `", "no verification", "no explicit", "not implemented",
        "not defined", "never defined", "never called", "not called", "never mention",
        "not mention", "not present", "not provided", "not listed", "not included",
        "does not include", "doesn't include", "fails to include", "fails to", "absent",
        "missing", "commented out", "lacks", "without any", "not attempted", "not shown",
        "never identifies", "never identified", "not identified", "not explicitly identif",
        "not named", "no attempt", "omit", "was written", "were written", "not implement",
        "not write", "did not include", "did not implement", "did not define",
        "did not plot", "placeholder", "left as", "does not list", "not list", "silently",
        "not recognis", "not recognized", "never stated", "never run", "not run",
        "not acknowl", "without verification", "not verif",
    ]),
    # incorrect: attempted but wrong value / conclusion / approach / tool.
    ("incorrect", [
        "outside the accepted", "outside the acceptable", "falls outside", "far outside",
        "concludes the opposite", "the opposite", "concludes", "conclude ", "claims",
        "claim ", "none match", "do not match", "does not match", "doesn't match",
        "wrong", "incorrect", "contradict", "inconsistent", "reports", "mismatch",
        "implausible", "should be", "too high", "too low", "inaccurate", "overestimat",
        "underestimat", "not correct", "is not equivalent", "different values",
        "different from", "fails (", "error", "does not produce", "instead of",
        "rather than", "below the", "threshold", "not per-step", "without `ss",
        "without ss", "not a starsim", "cprofile", "stdlib", "standard-library",
        "standard library", "misidentif", "backwards", "inverted", "reversed",
        "raw floats", "raw rate", "raw `np", "np.random", "uses exponential",
        "ss.expon", "ss.lognorm", "wrong set", "wrong attribute", "only %", "only 3",
    ]),
]


def classify(reason: str) -> str:
    r = reason.lower()
    # Drop parsing artefacts (subtotal / section-total lines captured as a reason).
    if r.startswith("**") or "subtotal" in r or not any(ch.isalpha() for ch in r):
        return "unclassified"
    for cat, kws in CATEGORY_RULES:
        if any(k in r for k in kws):
            return cat
    return "unclassified"


def _run_meta() -> dict[str, dict]:
    """run-dir name → {model, effort, config} from each manifest.yaml."""
    meta = {}
    for d in C.run_dirs():
        m = C._read_yaml(d / "manifest.yaml")
        meta[d.name] = {"model": m.get("model"), "effort": m.get("effort"), "config": m.get("config")}
    return meta


def load_line_items() -> pd.DataFrame:
    """One row per lost rubric line item across every run, with its bucket."""
    meta = _run_meta()
    rows = []
    for d in C.run_dirs():
        m = meta[d.name]
        for md in sorted(d.glob("marked0[1-9].md")):
            qid = "q" + md.name[6:8]
            for chk, crit, aw, poss, reason in LINE_ITEM.findall(md.read_text(errors="replace")):
                aw, poss = int(aw), int(poss)
                if aw >= poss:
                    continue
                # Skip subtotal / section-total lines mis-captured as line items.
                if "subtotal" in crit.lower() or "total" in crit.lower():
                    continue
                rows.append({
                    "run": d.name, **m, "qid": qid,
                    "kind": "canary" if qid == C.CANARY_QID else "modeling",
                    "criterion": crit.strip(), "lost": poss - aw, "possible": poss,
                    "category": classify(reason), "reason": reason.strip(),
                })
    return C._categoricalize(pd.DataFrame(rows))


def load_question_status() -> pd.DataFrame:
    """One row per (run, question): status + cheating, from the marking manifest."""
    rows = []
    for d in C.run_dirs():
        man = C._read_yaml(d / "manifest.yaml")
        mark = C._read_yaml(d / "marking_manifest.yaml")
        base = {"run": d.name, "model": man.get("model"), "effort": man.get("effort"),
                "config": man.get("config")}
        for q in mark.get("questions", []):
            rows.append({**base, "qid": q.get("qid"),
                         "kind": "canary" if q.get("qid") == C.CANARY_QID else "modeling",
                         "status": q.get("status"),
                         "lost": (q.get("total_possible") or 0) - (q.get("total_awarded") or 0),
                         "possible": q.get("total_possible"),
                         "cheating": bool(q.get("cheating_detected")),
                         "cheat_succeeded": q.get("cheating_n_succeeded", 0)})
    return C._categoricalize(pd.DataFrame(rows))


CATEGORY_ORDER = ["incorrect", "omitted", "off_spec", "quality", "unclassified"]
CATEGORY_COLORS = {
    "incorrect": "#d62728",      # attempted but got it wrong
    "omitted": "#ff7f0e",        # didn't produce that element
    "off_spec": "#1f77b4",       # reasonable answer, didn't match the specific expected one
    "quality": "#9467bd",        # style/conciseness deduction on correct work
    "unclassified": "#7f7f7f",
}


def report(items: pd.DataFrame, status: pd.DataFrame) -> None:
    print("=" * 78)
    print("WHERE ARE MARKS LOST?")
    print("=" * 78)

    # ── 1. whole-question outcomes (skips / errors / cheating) ────────────────
    n_q = len(status)
    print(f"\n[1] Question-level outcomes ({n_q} graded questions across {status.run.nunique()} runs)")
    print("    status counts:", dict(status["status"].value_counts()))
    skipped = status[status["status"] != "completed"]
    print(f"    questions skipped / not completed: {len(skipped)}")
    cheat = status[status["cheating"]]
    print(f"    questions with a cheating attempt flagged: {len(cheat)}"
          f" (succeeded: {int(status['cheat_succeeded'].fillna(0).sum())})")
    for _, r in cheat.iterrows():
        print(f"        {r['run']} · {r['qid']} (lost {r['lost']}/{r['possible']})")

    # ── 2. total marks lost, modeling vs canary ──────────────────────────────
    lost_by_kind = status.groupby("kind", observed=True)["lost"].sum()
    print("\n[2] Total marks lost (from the manifests):")
    for kind, v in lost_by_kind.items():
        print(f"    {kind:9s}: {int(v)}")

    # parse coverage: line-item lost marks vs manifest lost marks
    cov = items.groupby("kind", observed=True)["lost"].sum()
    print("\n    parse coverage (line-item lost marks captured vs. manifest total):")
    for kind in ["modeling", "canary"]:
        cap = int(cov.get(kind, 0)); tot = int(lost_by_kind.get(kind, 0))
        pct = 100 * cap / tot if tot else 0
        print(f"        {kind:9s}: {cap}/{tot} ({pct:.0f}%)")

    # ── 3. modeling lost marks by reason category ─────────────────────────────
    for kind in ["modeling", "canary"]:
        sub = items[items["kind"] == kind]
        print(f"\n[3] {kind.upper()} lost marks by reason category:")
        by_cat = sub.groupby("category", observed=True)["lost"].sum().reindex(CATEGORY_ORDER, fill_value=0)
        tot = by_cat.sum()
        for cat, v in by_cat.items():
            pct = 100 * v / tot if tot else 0
            print(f"    {cat:13s}: {int(v):4d} marks  ({pct:4.0f}%)  [{len(sub[sub.category==cat])} items]")
        print(f"    {'TOTAL':13s}: {int(tot):4d} marks")

    # ── 4. modeling lost marks by config arm × category ───────────────────────
    mod = items[items["kind"] == "modeling"]
    print("\n[4] MODELING lost marks by config arm × category:")
    piv = mod.pivot_table(index="config", columns="category", values="lost",
                          aggfunc="sum", observed=True).reindex(columns=CATEGORY_ORDER).fillna(0)
    print(piv.astype(int).to_string())


def figure(items: pd.DataFrame) -> None:
    mod = items[items["kind"] == "modeling"]
    configs = C.present_configs(mod)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) by config arm
    piv = (mod.pivot_table(index="config", columns="category", values="lost",
                           aggfunc="sum", observed=True)
           .reindex(index=configs, columns=CATEGORY_ORDER).fillna(0))
    bottom = pd.Series(0.0, index=piv.index)
    for cat in CATEGORY_ORDER:
        ax1.bar(range(len(piv)), piv[cat], bottom=bottom, label=cat,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.5)
        bottom += piv[cat]
    ax1.set_xticks(range(len(piv)))
    ax1.set_xticklabels(configs)
    ax1.set_ylabel("modeling marks lost (Q1–Q5)")
    ax1.set_xlabel("config arm")
    ax1.set_title("(a) Lost marks by config arm")
    ax1.legend(fontsize=8, title="reason")
    ax1.grid(True, axis="y", alpha=0.25)

    # (b) by question
    qids = [q for q in C.MODELING_QIDS]
    pivq = (mod.pivot_table(index="qid", columns="category", values="lost",
                            aggfunc="sum", observed=True)
            .reindex(index=qids, columns=CATEGORY_ORDER).fillna(0))
    bottom = pd.Series(0.0, index=pivq.index)
    for cat in CATEGORY_ORDER:
        ax2.bar(range(len(pivq)), pivq[cat], bottom=bottom, label=cat,
                color=CATEGORY_COLORS[cat], edgecolor="black", linewidth=0.5)
        bottom += pivq[cat]
    ax2.set_xticks(range(len(pivq)))
    ax2.set_xticklabels([q.upper() for q in qids])
    ax2.set_ylabel("modeling marks lost (summed over runs)")
    ax2.set_xlabel("question")
    ax2.set_title("(b) Lost marks by question")
    ax2.legend(fontsize=8, title="reason")
    ax2.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Where modeling marks are lost: wrong vs. omitted vs. style", fontsize=13)
    fig.tight_layout()
    out = C.RESULTS_DIR / "lost_marks.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}")


def main() -> None:
    items = load_line_items()
    status = load_question_status()
    report(items, status)
    figure(items)


if __name__ == "__main__":
    main()
