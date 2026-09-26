# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [To be filled by team]  
**Team Members:** [To be filled by team]  
**Submission Date:** [To be filled at submission]

---

## 1. Executive Summary

We implement country-aware multi-pass candidate generation followed by a histogram
gradient-boosted classifier using business-name and address comparisons. A decision
threshold is chosen using reference-level macro F0.5, with explicit singleton scoring.
This document describes the current working baseline, not a completed final submission.

---

## 2. Methodology

### 2.1 Problem Analysis

A full streaming audit counted 24,229,173 records: 2,206,821 training references,
10,320,219 training vendor records, 1,732,544 test references and 9,969,589 test vendor
records. Training includes US and India; France occurs only in test, with 259,452
reference records. Training labels contain 123,247 singletons (5.58%) and up to 11
matches per reference. Vendor addresses are missing in 344,883 training records.
Training reference names are ASCII, while 1,371,345 training vendor names contain
non-ASCII characters, making script variation an important challenge.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier  
**Core Innovation:** Complementary name-pair and name-number retrieval signatures,
Unicode-safe normalization and explicit measurement of blocking misses in the final metric.

Use a uniform reservoir of training references and search the complete training
vendor population. Split reference entities deterministically into 60% training,
20% threshold tuning and 20% model holdout. Do not use IDs as matching features.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** country plus sorted normalized core name; country plus
  sorted address tokens; pairs among the six longest core-name tokens; and combinations
  of the four longest core-name tokens with up to three address numbers. V2 also
  uses phonetic-name/number, address-token/number and address-token-pair signatures.
- **Candidate pairs generated:** measured in `reports/model_metrics.json`; full test
  candidate counts are not available until full inference finishes.
- **How true-match loss is monitored:** measure retained true links divided by all
  ground-truth links on tuning and holdout references. No claim of lossless blocking.

Legal suffixes are excluded from core names. Normalization standardizes selected
abbreviations and preserves Unicode combining marks. Blocks covering more than 200
references are discarded. A cheap weighted fuzzy name/address score retains at most
120 candidates per reference. These final candidates are exactly the records passed
to classifier inference and exported in `candidate_pairs.tsv`.

---

## 4. Matching Model

**Features used:**
- Name features: character ratio, token-sort ratio, token-set ratio, Jaro-Winkler,
  normalized core-name ratio, token Jaccard, exact agreement, length ratio.
- Address features: character and token similarities, token Jaccard, numeric Jaccard,
  numeric conflict, exact agreement, missingness, length ratio.
- Other: same-country indicator, script indicators, Latin accent-folded similarities,
  bidirectional fuzzy token coverage, overlap counts and partial similarities.
  There are 36 features; country identity is not one-hot encoded.

**Model type:** scikit-learn HistGradientBoostingClassifier, 160 iterations, learning
rate 0.08, at most 15 leaves, L2 regularization 3, minimum leaf size 30. No pretrained
weights are used. Original fitted model parameters are distributed under the MIT
license in `code/business_entity_resolution/MODEL_LICENSE`; scikit-learn remains a
separate BSD-licensed software dependency. This tree model is far below the eight
billion parameter limit. No third-party pretrained model is redistributed.

**Threshold selection method:** maximize mean per-reference F0.5 over the tuning
split using thresholds from 0.05 to 0.99. Evaluate the chosen threshold once on the
held-out reference split. A correct singleton gets one; a false match on it gets zero.
No unconstrained transitive merging is used.

---

## 5. Results & Error Analysis

The earlier rule baseline scored macro F0.5 0.4359 on 2,084 held-out references,
with link precision 85.20%, recall 24.26% and candidate recall 28.44% over its full
4,000-reference sample. This motivated broader candidate generation.

The upgraded model scored **0.8642250 macro F0.5** on the same 795 development
holdout references, versus **0.8329477** for v1. Link precision is **97.01%**, recall
**76.41%**, candidate recall **85.14%** and singleton accuracy **92.0%**. Country
macro F0.5 is US **0.9304**, India **0.7675**.

V2 trained on 266,695 candidate pairs (7,302 positive) from 2,423 references;
782 tuning references selected threshold **0.61**. The three splits contain
439,765 candidate pairs. Tuning macro F0.5 improved from **0.8553** to **0.8847**,
and this tuning result was the basis for promoting v2. A feature-only ablation
using v1 candidates scored development F0.5 **0.8400**, compared with **0.8642**
for new features plus broader retrieval. The holdout has been reused for development
comparisons, so it must not be described as a new, untouched final validation set.
A fresh reference sample and country-transfer evaluation are still required.

No France ground truth, test score or leaderboard score is available.

- **Common false positives:** observed cases include a pediatric dentistry name at a different street
  in the same city, and different industry names sharing city/address tokens. See
  `reports/model_errors.tsv` for the complete holdout error list.
- **Common false negatives:** exact retrieval misses simultaneous name/address
  perturbations; cross-script variants remain unresolved by basic normalization.

---

## 6. Conclusion

The implementation provides reproducible candidate preparation, supervised training,
threshold selection, inference and validated packaging under the requested filenames.
Further work should improve retrieval recall and multilingual handling, verify country
transfer, and replace repeated full vendor scans with a persistent index for full inference.

---

## Appendix

### A. Code Artefacts

All source is under `code/business_entity_resolution/src/`. The README gives exact
commands and `requirements.txt` pins the environment. Run `pipeline.py prepare`,
`pipeline.py train`, then `pipeline.py predict` to create `output/matching_results.tsv`
and `output/candidate_pairs.tsv`. The unchanged official validator is included.
`package_submission.py` assembles the required zip only after outputs exist and validate.

### B. Additional Results

See `reports/analysis.md`, `reports/dataset_profile.json`,
`reports/baseline_metrics.json`, and `reports/model_metrics.json`.
Nine automated tests cover metric edge cases, normalization, blocking and end-to-end
output validation on a synthetic dataset. Synthetic model scores are not validation results.

**Current limitations:** full global ID/label integrity audit is outstanding; France
transfer is unmeasured. Prediction uses repeated vendor scans per reference batch,
which is slow. Block-frequency caps are batch-dependent, so full-scale retrieval can
differ from sampled validation. Full test outputs have not yet been generated.
