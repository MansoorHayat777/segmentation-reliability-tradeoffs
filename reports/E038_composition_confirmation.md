# E038: confirmatory test of the failure-composition account (prereg/PHASE2.md section 6)

Generated from `outputs/E038_composition_confirmation/metrics.json`.

**Pre-registered decision: NOT SUPPORTED.** P1 failed; P2 held on 3/3 panels.

**Regressions:** none.

- Development data (Cityscapes train): 17850 records, 15874 severe miss, 2 high error.
- Selected formula: `pair__severity_arithmetic__fp_quality__w0d40` (development mixed AUROC 0.836).
- **P1 (weight on false-positive quality <= 0.5): FAILED.** The selected weight is 0.6 (frozen mixed score: 0.8).
- **P2 (re-selected beats frozen mixed score; Holm over 3 panels):**
  - AMOS-CT: 0.985 vs 0.968, delta +0.017 [+0.006, +0.034], Holm p 0.0006. P3 vs best single (cvar_quality): -0.002 [-0.009, +0.006].
  - Cityscapes-val: 0.882 vs 0.768, delta +0.114 [+0.065, +0.169], Holm p 0.0006. P3 vs best single (severity_arithmetic): -0.010 [-0.028, +0.006].
  - KITTI: 0.807 vs 0.623, delta +0.184 [+0.110, +0.260], Holm p 0.0006. P3 vs best single (severity_arithmetic): -0.022 [-0.060, +0.018].

Interpretation (recorded with the pre-registered verdict, not replacing it):
- Re-selecting the mixed score on severe-miss-dominated development data transfers strongly. It beats the frozen mixed score on all three held-out panels, and its gap to the best single metric is no longer distinguishable from zero.
- The specific mechanism prediction P1 failed. Selection did not reduce the weight on false-positive quality below 0.5. It replaced the component aggregate with severity-weighted arithmetic and kept a 0.6 weight on false-positive quality.
- The account as stated ("weight shifts to coverage") is therefore not confirmed. The gain comes from which coverage component is paired with false-positive quality, not from the nominal weight. Nominal weights are not comparable across components with different spreads, so P1 was a poorly chosen operationalization. That is a lesson for the next pre-registration, not grounds to change this verdict.
