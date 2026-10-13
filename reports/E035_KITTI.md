# E035: KITTI driving-domain panel (pre-registered in prereg/PHASE2.md section 4)

Generated from `outputs/E035_kitti_panel/metrics.json`. Six Cityscapes-trained SegFormer B0-B5 checkpoints on 200 KITTI images; frozen formulas and Cityscapes priority map.

**Regressions (listed first):** mixed_vs_best_single. Mixed score minus best single metric (severity_arithmetic) in mixed-failure AUROC at 0.25: -0.206 [-0.308, -0.099].

**Pre-registered replication rule:** met (B interval excludes 0 and the three non-trivial endpoint differences are Holm-significant and positive).

- Conflict bound B = 0.189 [0.180, 0.198]; exact minimum 0.191.
- Endpoint-specific minus mixed: policy rank +0.341 [+0.198, +0.481], operational rank +0.355 [+0.212, +0.497], severe-miss AUROC +0.175 [+0.052, +0.303] (Holm p 0.0084).
- Label composition: severe miss in 0.842 of records, high error in 0.055; every image has a critical component.

Interpretation: as on Cityscapes (E023), the mixed score, which weights false-positive quality at 0.8, underperforms severity-weighted arithmetic in a driving domain dominated by severe misses, while endpoint-specific scoring replicates.
