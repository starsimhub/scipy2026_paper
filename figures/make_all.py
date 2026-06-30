#!/usr/bin/env python3
"""Regenerate every paper figure.

Runs each figure script's ``main()`` in turn, reading data from the exam repo
(see :mod:`defaults`) and writing the PNGs into this directory. A failure in one
figure is reported but does not stop the rest.

Run: ``python figures/make_all.py``
"""

import importlib
import traceback

# Figure modules to run, in order. Each exposes a ``main()`` that writes its PNG.
FIGURE_MODULES = [
    "fig1_cost_vs_score",
    "fig2_effort_vs_score",
    "fig3_lost_marks",
    "fig4_judge_agreement",
    "fig5_utilization",
]


def main():
    failures = []
    for name in FIGURE_MODULES:
        print(f"\n=== {name} ===")
        try:
            importlib.import_module(name).main()
        except Exception:  # keep going so one broken figure doesn't block the rest
            traceback.print_exc()
            failures.append(name)

    print("\n" + "=" * 40)
    if failures:
        print(f"Done with {len(failures)} failure(s): {', '.join(failures)}")
        raise SystemExit(1)
    print(f"Done — regenerated all {len(FIGURE_MODULES)} figures.")


if __name__ == "__main__":
    main()
