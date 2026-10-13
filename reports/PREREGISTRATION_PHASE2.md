# Phase 2 pre-registration: E033 (annotation-free predictors) and E034 (modern panel)

Committed before any Phase 2 inference on training, validation or test data. Each Phase 2 `metrics.json` records the
commit hash of this file (`prereg_commit`). Any change after that commit is an **amendment**: it gets a dated
section at the end of this file, and the report must name every result it affects.

## 0. Fixed context

- Frozen formulas: `outputs/E014_profile_development/head_definitions.json`, SHA-256
  `836637e1a8b49ab7ae6b2ecd73986875aa0e1eade5d51c760fa52b9e4a738d25`. Never retuned.
- Priority maps, critical threshold (s ≥ 4), component rule (8-connected, ≥ 16 px), τ (0.20 COCO/VOC, 0.25 ADE/City),
  high-error threshold r ≥ 0.25, and the endpoint definitions are exactly as in the paper and the locked configs.
- Panels and checkpoints are the five locked panels with their pinned revisions (verified in Phase 0).
- Budget, approved by the user: **24 GPU-hours** in total for E033 + E034. Before each GPU stage, its projected time is
  checked against the remaining budget. If the projection exceeds it, the run stops and asks.
- Disk: about 58 GB free. Full softmax tensors are **not** stored. Per-pixel probability statistics are reduced to
  per-image and per-segment features at inference time (§1.3). Only those features are written.
- Serial GPU use; one panel at a time.

## 1. E033: annotation-free quality predictors

### 1.1 Question
Does the best annotation-free predictor of segmentation quality depend on the declared endpoint?

### 1.2 Data splits (fitting and tuning never touch validation panels)

| Panel | Training split used for fitting/tuning | Evaluation (never used for fitting or tuning) |
|---|---|---|
| COCO | Seeded subset of COCO **train2017**: 5,000 images drawn (seed 2061) from train images with ≥ 1 eligible VOC-category component, using the same eligibility rule as val. Only the needed images are downloaded from the official COCO server, together with `annotations_trainval2017.zip`. Hashes are recorded at download. | 4,031 val2017 images |
| VOC | VOC 2012 `train` (1,464 images). SBD/aug is not available locally and is not used. | 1,449 val images |
| ADE-S | Seeded subset of 5,000 ADE20K training images (seed 2062) | 2,000 val images |
| ADE-X | The same 5,000 ADE20K training images | 2,000 val images |
| City | All 2,975 Cityscapes train images | 500 val images |

Within each training split, 20% of **images** are held out for tuning (seed 2063, image-level, all six models stay
together). Training and validation inference must reproduce the locked per-image records exactly for the validation
images, checked as in E030. A mismatch stops the run.

### 1.3 Predictors (all computed from model outputs only; no labels)

Per pixel: softmax p (computed in float32 from the logits at label resolution), max-softmax m, entropy H, and
margin g (top-1 minus top-2 probability).

**(i) Training-free per-image scores.** Higher means better predicted quality, so entropy-type scores are negated.
1. `msp`: mean m over pixels
2. `neg_entropy`: −mean H
3. `margin`: mean g
4. `neg_boundary_entropy`: −mean H over pixels within 2 px of a predicted-class boundary (the same band definition as boundary IoU). If no boundary pixels exist, it falls back to −mean H.
5. `neg_worst_segment_entropy`: −max over predicted segments of the segment's mean H. Segments are 8-connected
   predicted-class regions with ≥ 16 px, or the whole image if there is none.

**(ii) MetaSeg-style segment features.** For each predicted segment: log pixel count; mean and variance of H; mean and
variance of g; boundary-to-interior ratio of mean H; and boundary pixel fraction. These are aggregated per image as
mean, size-weighted mean, min and max of each feature, plus the number of segments. Together with the five
training-free scores, this gives a fixed feature vector of 34 dimensions per (image, model) record. The exact list is
written to `config.yaml`. Features are standardized using training-split statistics only.

**(iii) Learned predictors.** Model family: `sklearn.ensemble.HistGradientBoosting{Regressor,Classifier}` (sklearn 1.8.0).
- Grid, declared now: `learning_rate` ∈ {0.05, 0.1} × `max_leaf_nodes` ∈ {15, 31} × `max_iter` ∈ {200, 500};
  `l2_regularization` = 1.0; `random_state` = 2064; no early stopping.
- **Endpoint-trained** (one per panel per endpoint):
  - U_P: regressor on U_P
  - U_O: regressor on U_O
  - Severe miss: classifier on severe-miss label, critical-image records only
  - High error: classifier on r ≥ 0.25
- **Mixed-trained** (one per panel): classifier on mixed failure, applied unchanged to every endpoint as score = 1 − P(failure).
- Selection rule: each grid point is scored on the 20% tuning hold-out using the predictor's own training endpoint metric. Rank endpoints use mean per-image Spearman; detection endpoints use AUROC. The best point is chosen, with ties broken by the smaller model (lower `max_iter`, then fewer leaves, then lower learning rate). It is then refit on 100% of the training split.

### 1.4 Evaluation metrics (same as the paper)
- Rank endpoints: per-image Spearman (average ranks) between predictor score and U_P or U_O across the six models,
  averaged over images where both orderings are non-constant.
- Detection endpoints: AUROC, with loss = −score; severe miss only on images with a critical component.

### 1.5 Hypotheses and tests

**H1 (primary).** For each endpoint, the endpoint-trained predictor beats the mixed-trained predictor.
- Family: 5 panels × 4 endpoints (U_P, U_O, severe miss, high error) = **20 one-sided tests**. All 20 are always reported, and none is dropped.
- Δ = metric(endpoint-trained) − metric(mixed-trained), paired on the same evaluation records.
- Bootstrap: 5,000 image-cluster draws (seed 2065 + 20·panel + endpoint).
  - Rank endpoints use per-image differences.
  - AUROC endpoints use **exact recomputation per draw** (E032 method). This was accepted under the paired-exchange null calibration, by user decision (E032; Phase 1 report). The linearized result is reported alongside.
- p = (1 + #{Δ* ≤ 0}) / (B_valid + 1). Undefined draws are excluded and counted.
- **Holm** correction across the 20, at α = 0.05.
- Decision rules:
  - A test supports H1 if Holm p ≤ 0.05 and Δ > 0.
  - A **regression** is any test with Δ < 0 whose two-sided 95% CI excludes 0. Every regression is reported in the first table of the report.
  - H1 is **supported** for an endpoint if it holds in ≥ 3 of 5 panels with no regression on that endpoint.
  - H1 is **supported overall** if it is supported for all four endpoints.
  - Otherwise H1 is reported as partially supported or not supported. The report states it plainly.

**H2 (secondary).** The best training-free score (from the five in §1.3 (i)) differs across endpoints.
- Winners are chosen **on the training split** (all training images; training-free scores need no fitting). For each panel and each of the 5 endpoints, the winner is the argmax of the endpoint metric. Ties go to the earlier score in the §1.3 list.
- For each panel and each endpoint pair (e1, e2) whose training winners differ (w1 ≠ w2), an intersection–union test is run on the evaluation panel:
  - Δ1 = metric_e1(w1) − metric_e1(w2) > 0, and Δ2 = metric_e2(w2) − metric_e2(w1) > 0
  - p_pair = max(p1, p2), using the same bootstrap as H1
- Family: all such pairs across panels; Holm at α = 0.05. If no pair has differing winners on a panel, that panel contributes no tests and is reported as "no crossing".
- Decision rules:
  - H2 is supported on a panel if at least one pair is Holm-significant.
  - H2 is **supported** overall if it holds on ≥ 3 of 5 panels.

**Context (descriptive, untested).** For each panel and endpoint, the report gives the best annotation-free predictor's
metric minus the frozen annotation-based head h_* (h_P, h_O, h_S, h_G, h_M). No test is run on these gaps.

### 1.6 Compute plan (projection; checked against the budget before each stage)
- Inference with feature extraction, training split plus validation: COCO about 0.8 h, VOC about 0.3 h, ADE-S about 2.6 h,
  ADE-X about 4–5 h, City about 1.5 h. Total about 10–11 GPU-hours.
- Fitting: CPU only.

## 2. E034: modern ADE20K panel (new domain deferred)

**New domain: deferred** by user decision (no licensed dataset available). No priority map for a new domain is
defined in this pre-registration.

**Modern ADE20K panel.** Six recent models, evaluated on the same 2,000 ADE20K validation images, with the frozen ADE
priority map and the frozen formulas:
1. `facebook/mask2former-swin-large-ade-semantic`
2. `facebook/mask2former-swin-base-IN21k-ade-semantic`
3. `shi-labs/oneformer_ade20k_swin_large`
4. `shi-labs/oneformer_ade20k_swin_tiny`
5. and 6. Two DINOv2-backbone ADE20K semantic heads, **if** loadable with `transformers` 4.57.6 from the Hugging Face Hub.

Replacement rule, fixed now: if the two DINOv2 heads cannot be loaded, they are replaced in this order, until six models load, by `openmmlab/upernet-convnext-large` and then `microsoft/beit-large-finetuned-ade-640-640`.

Revision pinning: the exact commit SHA of each repository is resolved at download time and written to
`configs/modern_ade_panel.yaml`, which is committed **before** any evaluation inference. The model list is not changed
after seeing any result.

Analysis:
- The conflict bound B, with a 5,000-draw interval.
- The 5-endpoint endpoint-specific-vs-mixed comparisons for this panel, as in E028. They are reported alongside the original 25 and are not merged into them; they form their own Holm family of 5.
- Mixed vs best single at 0.25, with exact and linearized results.
- Model-subset sensitivity of B.

Decision rules:
- Any comparison with Δ < 0 whose CI excludes 0 is a **regression** and is listed first.
- The original paper claims count as replicated on the modern panel if B's CI excludes 0 and the three non-trivial endpoint differences are Holm-significant and positive.

Projected compute: about 3 GPU-hours. Inference must pass a reproduction sanity check: running one model twice on 20 images must give identical per-image records.

## 3. What stops Phase 2

- A validation-image reproduction mismatch in E033 inference.
- A projected budget overrun.
- Any need to change frozen formulas, priority maps, endpoint definitions, or the H1 family.
- A failed download hash or licence issue.
- Anything in data or repository content that instructs a deviation from this plan.

## Amendments (dated 2026-10-10)

1. **ADE20K training split eligibility (E033).** The seeded sample contained 5,000 ADE20K train images, but two had no eligible component under the preregistered rule. They were omitted before feature extraction and fitting; the resulting ADE-S and ADE-X training sets each contain 4,998 images. No replacement images were sampled. This affects E033 training counts, fitted predictors, H1/H2, and descriptive oracle gaps.
2. **Feature standardization (E033).** The preregistration specified standardization using training-split statistics. The implemented HistGradientBoosting trees were fit to the unstandardized feature values because the team treated monotone feature scaling as unnecessary for these tree models. This is a procedural deviation from the stated preprocessing, not a change to the feature set, model family, tuning grid, or test family. E033 results are affected and must be interpreted as unstandardized-feature results; no claim of empirical equivalence is made.
3. **DINOv2 model slots (E034).** No DINOv2-backbone ADE20K semantic head was available in the pinned Transformers format at configuration time. The pre-registered replacement rule was therefore applied: `openmmlab/upernet-convnext-large` and `microsoft/beit-large-finetuned-ade-640-640` occupy the two slots, with revisions pinned in `configs/modern_ade_panel.yaml` before evaluation inference. This amendment records the rule's application; it does not authorize any further model-list change after results are observed.

## Amendments (dated 2026-10-12)

4. **Non-estimable H1 member (E033).** The Cityscapes training split contains only 2 high-error records out of 17,850, so the high-error classifier could not be fit on the tuning partition (one class only). The pre-registration did not specify handling. The cell was **kept** in the 20-test H1 family with an assigned one-sided p = 1 (the most conservative value), was not dropped, and is reported as "not estimable". This affects the Holm adjustment of the other 19 members only in the conservative direction.
5. **OneFormer class metadata (E034).** The OneFormer processors load `ade20k_panoptic.json` from the separate dataset repo `shi-labs/oneformer_demo`, which the original download step did not fetch; the repeat-run sanity check caught this before any inference. The file was fetched at the `main` ref and the resolved commit is recorded in `outputs/E034_modern_ade_panel/oneformer_metadata_revision.json`. This is a loader dependency; the six pinned model revisions are unchanged.


## Section 4 (added 2026-10-12, before any KITTI inference): E035 KITTI driving-domain panel

- **Data:** KITTI semantic segmentation (`data_semantics.zip`, SHA-256
  `7e64a6a17c1b8c620180e6ddae2736bea4e5e420a9e1088d56c0c4358fcbfa51`), the 200 annotated `training` images
  (Cityscapes labelId format). Licence CC BY-NC-SA 3.0. No KITTI image is used for fitting or tuning anything.
- **Models:** the six SegFormer B0–B5 Cityscapes checkpoints, at the revisions pinned in
  `configs/cityscapes_confirmation.yaml`. None was trained on KITTI.
- **Protocol:** identical to E023. This covers the priority map, the critical threshold s ≥ 4, components (8-connected, ≥ 16 px), τ = 0.25, the high-error threshold 0.25, the scorer, mixed precision and labelId decoding. Images with no eligible component are excluded, and their count is reported.
- **Frozen formulas:** h_P, h_O, h_S, h_G, h_M are unchanged.
- **Analyses:** the same as E034.
  - B with a 5,000-draw interval, plus B_exact.
  - Model-subset sensitivity of B.
  - The 5 endpoint-specific-vs-mixed comparisons, as their own Holm family of 5 (exact AUROC; per-image rank differences).
  - Mixed vs best single at 0.25, with exact and linearized results.
- **Decision rules:** as in E034.
  - Any Δ < 0 whose CI excludes 0 is a regression and is listed first.
  - "Replicated" means B's CI excludes 0 and the three non-trivial differences are Holm-significant and positive.
- **Sanity:** one model run twice on 20 images must give identical records before full inference.
- **Reporting:** the outcome is recorded in `reports/` whatever its direction.

## Section 5 (added 2026-10-12, approved by the user, before any AMOS evaluation inference): E036 abdominal CT panel

- **Data:** AMOS22 (`amos22.zip`, Zenodo 10.5281/zenodo.7262581, MD5 `67717b2a483ac0744c89c3016b7aaef7`).
  - Licence: Zenodo lists CC-BY 4.0; `dataset.json` lists CC-BY-SA 4.0.
  - Evaluation set: the 100 labelled **validation CT** volumes (`imagesVa`/`labelsVa`, ids < 500). MRI volumes are excluded.
  - No AMOS volume is used for fitting or tuning.
  - One AMOS *training* volume is used only for model run-checks before evaluation.
- **Models (six):**
  - TotalSegmentator 2.18.0, task `total`, full resolution
  - TotalSegmentator 2.18.0, task `total`, `--fast`
  - MONAI `wholeBody_ct_segmentation` 0.2.7, `model.pt` (high resolution)
  - MONAI `wholeBody_ct_segmentation` 0.2.7, `model_lowres.pt` (low resolution)
  - MONAI `swin_unetr_btcv_segmentation` 0.5.8
  - MONAI `multi_organ_segmentation` 0.0.6

  Each model runs with its own published inference pipeline, and predictions are resampled (nearest neighbour) to the AMOS label grid.
- **Leakage review:**
  - TotalSegmentator and wholeBody are trained on TotalSegmentator data (University Hospital Basel), Swin-UNETR on BTCV, and multi-organ on Aichi Cancer Center data. None is documented as using AMOS.
  - **Known residual risk:** the TotalSegmentator README states that "a large part" of its training data is public, so a remainder is undocumented.
- **Labels:** the five organs predicted by all six models: liver, spleen, stomach, gallbladder, pancreas.
  - Every other AMOS label and every other model class is mapped to background.
- **Priority map (clinically motivated, approved):**
  - pancreas 5, gallbladder 4, spleen 4, liver 3, stomach 2
  - Critical means s ≥ 4; a severe miss is a critical component with coverage < 0.5.
- **Scorer:** `saec.volume_metric`, the N-d port of the COCO/VOC scorer, validated to reproduce it exactly in 2-D.
  - Full 3-D connectivity.
  - Components ≥ 1,000 mm³, converted to a voxel count per volume from its spacing.
  - Error rate r = fraction of predicted-organ voxels whose class is wrong (the COCO/VOC adapter).
  - CVaR tail fraction τ = 0.25; boundary band 2 voxels; high-error threshold r ≥ 0.25.
- **Frozen formulas:** h_P, h_O, h_S, h_G, h_M are unchanged.
- **Analyses and decision rules:** identical to E034/E035.
  - B with a 5,000-draw interval, plus B_exact.
  - Model-subset B.
  - The 5 endpoint-specific-vs-mixed comparisons, as a Holm family of 5.
  - Mixed vs best single at 0.25.
  - Regressions are listed first.
  - "Replicated" means B's CI excludes 0 and the three non-trivial differences are Holm-significant and positive.
  - Endpoints with a single label class are reported as not estimable.
- **Budget:** projected ≤ 6 GPU-hours, within the remaining Phase 2 budget (about 16.5 h).
- **Reporting:** the outcome is recorded in `reports/` whatever its direction.

## Amendment 6 (dated 2026-10-12, before any AMOS evaluation inference)

The E036 run-check script first required every model to predict all five organs on the training volume `amos_0001`. The MONAI `multi_organ_segmentation` model ran correctly and its output lay on the image grid, but it predicted only liver, spleen and stomach on that volume; its raw output contained no gallbladder or pancreas classes. Model quality is not a run-check criterion, and excluding a weak model would bias the panel. The run-check was therefore corrected to: the pipeline runs, the output is on the image grid, and at least one organ is predicted. The model list and every other protocol element are unchanged.

## Section 6 (added 2026-10-13, before any computation): E038 confirmatory test of the failure-composition account

**Background.** E037 (post hoc) attributed the frozen mixed score's losses on street and CT panels to failure composition. Where nearly all failures are severe misses, the best detector weights coverage, not false-positive quality. This test checks that account on data not used to form it.

- **Development data:** Cityscapes **train** records only (`outputs/E033_features/City/train/*.csv`: 2,975 images × the six pinned SegFormer checkpoints; scorer outputs under the E023 protocol). Its failure composition is known: 15,874 severe-miss records and 2 high-error records.
- **Selection:** the 260-formula library of E014, with the E014 single-dataset rule.
  - Maximize mixed-failure AUROC on the development data; ties are broken by formula id.
  - Endpoint definitions are unchanged (high error r ≥ 0.25; severe miss as declared).
  - The selected formula is frozen before any evaluation.
- **Evaluation panels:**
  - Cityscapes validation (E023)
  - KITTI (E035)
  - AMOS CT (E036)

  None is used for selection. Cityscapes val and KITTI were analysed in E037, so the test is confirmatory for the **selection outcome and its transfer**, not blind to those panels' composition.
- **Pre-registered predictions:**
  - **P1:** the selected formula puts weight ≤ 0.5 on false-positive quality (`fp_quality`); the frozen mixed score puts 0.8.
  - **P2 (primary):** on each evaluation panel, the re-selected score beats the frozen mixed score on mixed-failure AUROC: Δ = AUROC(re-selected) − AUROC(h_M) > 0.
    - Tested one-sided with a 5,000-draw exact image-cluster bootstrap.
    - p = (1 + #{Δ* ≤ 0}) / (B + 1); Holm across the 3 panels; α = 0.05.
  - **P3 (secondary, descriptive):** re-selected minus the strongest single metric (chosen per panel, as before), with a 95% CI.
- **Decision rules:**
  - The account is **supported** if P1 holds and P2 is Holm-significant on at least 2 of 3 panels with no significant reversal.
  - It is **not supported** if P1 fails, or if P2 fails on 2 or more panels.
  - Any significant Δ < 0 is a regression and is listed first.
- **Reporting:** the outcome is recorded in `reports/` whatever its direction.

## Section 7 (added 2026-10-13, before any pair sampling or data collection): E039 human preference study

**Purpose.** Provide an endpoint not constructed from the scores' own components, addressing the critique that endpoint-specific scores win "by construction".

- **Stimuli:**
  - Source: the ADE-X panel (ADE20K validation, six locked checkpoints).
  - A pair is (image, model A, model B), taken from images with a critical component.
  - **Main pairs:** 100, drawn uniformly at random (seed 2079) from pairs where the endpoint-specific policy score h_P (= q_t) and the general mixed score h_M order the two models strictly and in opposite directions.
  - **Control pairs:** 20 more, drawn at random (seed 2080) from pairs where h_P and h_M agree. These are used only for annotator consistency.
  - Each item shows the image, the ground truth and the two predictions. Left/right placement is randomized per item (seed 2081), and item order is shuffled per annotator.
- **Question (fixed wording):** "Focus on people and vehicles (cars, trucks, buses, vans, bicycles, motorbikes). Which segmentation covers them better — fewer missed or badly segmented people or vehicles?" The answer options are A / B / Can't tell.
- **Annotators:** at least 3, recruited by the author, blind to model identity and to all scores. Each gives informed consent on the first screen. No personal data beyond an anonymous annotator code is recorded.
- **Primary analysis (H-human):**
  - For each main pair, the human majority is the option chosen by most annotators. Pairs whose majority is "Can't tell", or that have no majority, are excluded and counted.
  - Statistic: the proportion of retained main pairs on which h_P agrees with the majority. On these pairs h_M necessarily agrees on the complement.
  - Test: one-sided against 0.5 with a 5,000-draw bootstrap clustered by image; p = (1 + #{π* ≤ 0.5}) / (B + 1); α = 0.05.
  - **Supported** if p ≤ 0.05; **reversed** if the 95% CI lies wholly below 0.5; otherwise inconclusive.
- **Secondary (descriptive):**
  - Agreement of foreground IoU and of each single criterion with the majority, over all retained pairs.
  - Fleiss' kappa among annotators.
  - Agreement on the control pairs.
- **Reporting:** the outcome is recorded in `reports/` whatever its direction.
