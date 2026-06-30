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

All data loaders live in `utils.py` (Group A reads `scores.jsonl`; Group B reads
`validation/answers/`); shared config and plotting style live in `defaults.py`.

All figures use **Mulish** as their font — Starsim's default plotting font, which ships bundled in `starsim/assets/`. `defaults.py` registers it with Matplotlib on import and sets it as the default `font.family`, falling back silently to the Matplotlib default if Starsim isn't installed.

| Script | Output | Data |
| --- | --- | --- |
| `fig1_cost_vs_score.py` | `fig1_cost_vs_score.png` | Group A + B |
| `fig2_effort_vs_score.py` | `fig2_effort_vs_score.png` | Group B |
| `fig3_lost_marks.py` | `fig3_lost_marks.png` | Group B |
| `fig4_judge_agreement.py` | `fig4_judge_agreement.png` | Group A |
| `fig5_utilization.py` | `fig5_skill_utilization.png` | Group B |

## Running

From anywhere (the scripts are CWD-independent). Regenerate everything:

```bash
python figures/make_all.py
```

Or run a single figure:

```bash
python figures/fig1_cost_vs_score.py
```

Requires `pandas`, `seaborn`, `matplotlib`, `numpy`, and `pyyaml`.
