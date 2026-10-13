# One Score Cannot Serve Every Decision: paper and reproducibility artifacts

Camera-ready source and PDF of *One Score Cannot Serve Every Decision: Criterion Conflict and Failure
Composition in Segmentation Quality*, with the aggregate result artifacts behind every number in it.

## Contents

| Path | What it is |
|---|---|
| `paper/main_camera_ready.pdf` | The paper |
| `overleaf.zip` | Overleaf-ready source; upload, extract, and use `main.tex` as the root document |
| `paper/` | The same source, the figures, and the figure script |
| `paper/merged_numbers.tex` | Every number in the text, as a LaTeX macro generated from the result artifacts |
| `reports/merged_ledger.csv` | For every macro: its value and the artifact and field it came from |
| `outputs/` | `metrics.json` and `config.yaml` for each cited experiment (16 experiments) |
| `reports/PREREGISTRATION_PHASE2.md` | Dated pre-registration, including the human study protocol |
| `reports/E0xx_*.md` | Per-experiment outcome reports |

## Regenerate the data figures

With Python 3, NumPy and Matplotlib, from this directory:

```
python paper/make_figures.py
```

This rewrites `paper/figures/composition_gap.pdf` and `gains_forest.pdf` from `reports/paper2_numbers.json`.
The teaser and method overview are illustrations. The qualitative figure is rendered from model predictions and
needs the ADE20K data and checkpoints.

## Not included

Datasets (COCO, PASCAL VOC, ADE20K, Cityscapes, KITTI, AMOS), model checkpoints, per-image records, and the images
shown to human reviewers. Obtain datasets from their rights holders under their licences. Checkpoint repositories and
pinned revisions are listed in each experiment's `config.yaml`.
