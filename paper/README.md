# Segmentation reliability evaluation: paper and reproducibility artifacts

This repository contains the manuscript source, the exact aggregate artifacts
and configurations used for the reported analyses, and the script that
regenerates the paper figures. The Overleaf-ready source is in `overleaf.zip`;
select `main.tex` as the project’s root document after uploading and extracting
the archive.

## Reproduce the figures

Use Python 3 with NumPy and Matplotlib installed, then run from this directory:

```powershell
python paper/make_figures.py
```

The script reads aggregate JSON artifacts from E021, E023, E024, E025, E027,
and E028 and writes four PDF figures to `paper/figures/`. E026 aggregate
ranking-stability results are also included for auditing the manuscript.
Configuration snapshots, aggregate metrics, CSV sensitivity tables, and
sanitized package-version/GPU provenance are included for independent
inspection.

## Scope and data

The release intentionally excludes source datasets, model checkpoints,
per-image Cityscapes records, and machine-specific logs. Cityscapes must be
obtained from its rights holder under its terms. The aggregate artifacts allow
inspection and figure regeneration; rerunning model inference requires the
datasets, checkpoints, and upstream model-evaluation artifacts specified by
the experiment configurations.

The named camera-ready source is `paper/main_camera_ready.tex` (also provided
as `main.tex` at the root of `overleaf.zip`); it identifies the sole author and
links to this repository. The anonymous review source/PDF is kept outside this
identity-linked public repository. Update the bundled CVPR style
files when the official CVPR 2027 author kit becomes available, and verify the
applicable submission rules before using an identified author version for
review.
