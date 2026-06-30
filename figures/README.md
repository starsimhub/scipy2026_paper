# Paper figures

Scripts that regenerate the figures used in the paper. Each reads its data from
the companion exam repository and writes a PNG into this directory.

## Configuration

All data paths flow from a single setting in [`defaults.py`](defaults.py):

```python
RESULTSDIR = "../scipy2026_starsim_exam"   # resolved relative to the repo root
```

Point `RESULTSDIR` at a different checkout of the exam repo and every script
follows. The derived inputs are `results/scores.jsonl` and `questions/` (Group A)
and `validation/answers/` (Group B). Figures are written next to the scripts
(`OUTPUT_DIR`).

## Scripts

Group A — inspect-ai exam (`scores.jsonl`); shared helpers in `exam_common.py`:

| Script | Output |
| --- | --- |
| `judge_agreement.py` | `judge_agreement.png` |
| `performance_vs_cost.py` | `performance_vs_cost.png` |

Group B — validation runs (`validation/answers/`); shared loader in
`validation_common.py`:

| Script | Output |
| --- | --- |
| `fig3_effort_vs_score.py` | `fig3_effort_vs_score.png` |
| `fig4a_cost_vs_score.py` | `fig4a_cost_vs_score.png` |
| `fig5_utilization.py` | `fig5_skill_utilization.png` |
| `lost_marks.py` | `lost_marks.png` |

## Running

From anywhere (the scripts are CWD-independent). Regenerate everything:

```bash
python figures/make_all.py
```

Or run a single figure:

```bash
python figures/judge_agreement.py
```

Requires `pandas`, `seaborn`, `matplotlib`, `numpy`, and `pyyaml`.
