# Roadmap toward macro F0.5 = 0.990556

This is an experimental target, not a forecast or a measured leaderboard score.
The current measured development score is 0.864225 on 795 training references.
The reused development holdout cannot serve as final confirmation.

## Measured loss budget

| Holdout group | References | Candidate link recall | Best possible macro F0.5 with current candidates | Current model macro F0.5 |
|---|---:|---:|---:|---:|
| US | 472 | 0.9503 | 0.9839 | 0.9304 |
| India | 323 | 0.7073 | 0.8444 | 0.7675 |
| All | 795 | 0.8514 | 0.9272 | 0.8642 |

Of 2,679 true holdout links, 398 never reach the classifier. A perfect scorer
restricted to current candidates cannot exceed 0.9272 macro F0.5 on these
references. The remaining 0.0630 gap between that ceiling and 0.8642 reflects
scoring/decision errors. France has no training labels and remains unmeasured.
A score of 0.990556 leaves only 0.009444 total average loss, so both candidate
retrieval and scoring must become close to complete on every group.

## 1. Build an indexed, multi-channel candidate generator

Replace repeated scans and the single globally ranked top-120 heap with a persistent
vendor index, separate for train/test and country. Query several channels per reference:

- Existing exact normalized name/address keys and number keys.
- Rare name and address token postings, weighted by corpus frequency. Keep common city,
  state and generic legal words from dominating.
- Character 3-5-gram retrieval on name and address to recover typos, joined words,
  reordered components and website-style names.
- Address-only retrieval using building number plus rare street/building tokens, including
  records whose names use different scripts. Normalize leading zeroes and common
  house-number prefixes as alternative keys, retaining the original value as a feature.
- A name-only channel for missing vendor addresses. Use script-aware transformations
  trained solely from provided labels if cross-script names need comparison.

Reserve candidates per channel before taking the union, so a common-name pile cannot
push out address-only evidence. Score retrieval recall both before and after top-k
truncation. Measure the oracle macro F0.5 and candidate volume, not only pair recall.
Build retrieval from the allowed unlabeled vendor pool; keep validation labels and
their reference entities out of supervised model fitting and threshold selection.

**Gate:** On fresh, entity-separated validation, candidate oracle macro F0.5 >= 0.995
and strong recall separately for US, India, missing-address, and cross-script cases.
That gate leaves little room for classifier errors, and may need to rise further.

## 2. Use many more labeled references

The current classifier uses only 2,423 training references selected from 2.21 million.
After indexing, prepare at least 100,000 reference entities, then scale further if
recall and runtime allow. Keep all positives retained by retrieval. Mine negatives
from the hardest retrieved false matches rather than relying on easy random pairs.
Maintain entity-grouped train/tuning/test splits and a fresh final validation sample.
Check that linked vendor records cannot leak across splits; if they do, split whole
connected components together. Report per-country and singleton performance.

**Gate:** More training data improves a new validation fold, not just the previous
795-entity development holdout.

## 3. Train stronger scorers only after retrieval improves

Compare the current histogram tree with a LightGBM classifier using the same
candidate sets and reference splits. Add country-neutral address structure features:
number compatibility, rare street/building token overlap, city/region agreement,
missing components, cross-script indicators, and retrieval-channel ranks.
Train/calibrate scores for the actual candidate mixture. Consider a second-stage
text-pair model only for a small top-ranked subset; do not depend on it to recover
links absent from retrieval. Use only provided challenge data and permitted model
licenses/weights.

**Gate:** Quantify the residual gap between candidate oracle F0.5 and actual F0.5.
The new scorer must reduce it on fresh validation without harming singleton accuracy.

## 4. Tune per-entity decisions for macro F0.5

Select thresholds on the tuning set using the official per-reference metric, not
pairwise F1. Examine separate thresholds for records with missing addresses, script
mismatch or different sources only when each subgroup has enough validation data.
Estimate whether a reference has no matches from its top scores and score gap.
Retain multiple legitimate vendor matches; do not impose one-to-one matching or
unconstrained transitive closure. Audit conflicts before any assignment rule.

**Gate:** Each decision rule improves full macro F0.5 on a fresh grouped validation
fold; track false positives on true singletons explicitly.

## 5. Validate country transfer and full-scale inference

France is about 15% of test references and has no labels. Simulate transfer by
training without one labeled country and evaluating on that country, while noting
that US/India transfer cannot prove France accuracy. Test French-like accent,
address and legal-suffix perturbations without accessing external business data.
The current prediction path rescans nearly 10 million test vendors for every
10,000-reference batch, which is impractical for iterative full submissions. A
persistent index must be shared by training and inference; block-frequency policies
must be stable across batch sizes. Measure runtime, memory and candidate volume on
large test partitions before full inference.

**Final gate:** Generate both required TSVs, check that predictions are a subset of
candidates, run the supplied validator with ID checks, and package the exact code
and documentation that generated them. A leaderboard score, if available, is the
only evidence for the unlabelled test distribution.

## Evidence and methodological sources

The local measurements come from `model_metrics.json`, `model_comparison.json` and
`training_candidates.json`. The proposed character n-gram, SQLite FTS5 trigram,
and LightGBM methods are implementation options, not measured improvements.
No external business-identity lookup is part of this plan.
