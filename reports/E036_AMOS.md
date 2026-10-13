# E036: AMOS22 abdominal CT panel (pre-registered in prereg/PHASE2.md section 5; amendment 6)

Generated from `outputs/E036_amos_panel/metrics.json`. Six public CT models (TotalSegmentator full/fast, MONAI wholeBody high/low-res, Swin-UNETR BTCV, MONAI multi-organ DiNTS), 100 AMOS22 validation CT volumes, five shared organs, approved clinical priority map, frozen formulas.

**Regressions (listed first):** mixed_vs_best_single. Mixed score minus best single metric (cvar_quality) in mixed-failure AUROC at 0.25: -0.020 [-0.034, -0.007].

**Pre-registered replication rule:** met.

- Conflict bound B = 0.116 [0.106, 0.125]; exact minimum 0.116.
- Endpoint-specific minus mixed: policy rank +0.130 [+0.092, +0.170], operational rank +0.112 [+0.084, +0.142], severe-miss AUROC +0.025 [+0.011, +0.040]; all Holm p = 0.0010 or 0.0010.
- High-error AUROC difference +0.001 [+0.000, +0.002], Holm p 0.3111 (not significant).
- Label composition: severe miss in 0.447 of records, high error in 0.025, mixed failure in 0.448.

Execution notes: MONAI bundles needed loader-compatibility overrides (img_size removed in MONAI 1.5; weights_only=False for the official DiNTS architecture file), were run in chunks of 10 volumes, and the DiNTS sliding-window output was stitched on the CPU to avoid GPU-memory spill. None changes model weights or pipelines.

Interpretation: endpoint-specific scoring replicates in a medical domain. As in the driving panels (E035, E037), high-error failures are rare here, so the mixed failure label is almost entirely severe misses and the mixed score (0.8 weight on false-positive quality) trails a tail/coverage score, although by a smaller margin than in driving.
