# Initial dataset analysis

Full-file streaming audit; no external data used.

| Split | Reference records | Source 2 | Source 3 |
|---|---:|---:|---:|
| train | 2,206,821 | 5,034,616 | 5,285,603 |
| test | 1,732,544 | 4,887,273 | 5,082,316 |

Total source records: 24,229,173.
Training label rows: 2,206,821; singletons: 123,247 (5.58%); maximum matches: 11.

## Findings

- France is absent from training and has 259,452 test reference entities (14.98% of the test references).
- Reference names and addresses are populated throughout. Vendor addresses are missing for 344,883 training and 265,506 test records.
- Training reference names are entirely ASCII, while 1,371,345 training vendor names contain non-ASCII characters. Transliteration and cross-script matching need attention.
- Ground-truth lists contain no repeated target IDs within a row.
- The US share of reference records drops from approximately 60% in training to 38% in test; India rises from 40% to 47%.

## Work implemented

A reproducible 4,000-entity reservoir sample, candidate search over the full training vendor pool, a deterministic tuning/holdout split, threshold selection, per-entity macro F0.5, country metrics, singleton accuracy, and an error export.

This initial baseline uses exact normalized name OR address within country for retrieval. It will miss records with variations in both fields. No test predictions or trained ML model are claimed.

## Next experiments

1. Add rare-token and character n-gram retrieval; measure blocking recall and candidate volume before model tuning.
2. Preserve combining marks explicitly and develop training-data-only transliteration handling; basic alphanumeric normalization can lose Indic vowel marks.
3. Train a pair classifier using name, address, number agreement and missingness features; split by reference entity.
4. Run country holdouts to estimate transfer risk before applying to France.
5. Audit global ID uniqueness, label references and cross-split overlap; generate both required TSVs and run the official validator once the model is ready.

Detailed counts: `dataset_profile.json`. Baseline results: `baseline_metrics.json`. Holdout errors: `baseline_errors.tsv`.

## Completed baseline results

Evaluated 2,084 held-out reference entities after threshold selection on 1,916 separate entities. Searched all 10,320,219 training vendor records.

- Macro F0.5: 0.4359 (empty-prediction baseline: 0.0581).
- Link precision: 85.20%; recall: 24.26%.
- Singleton accuracy: 90.08%.
- Candidate recall over all 4,000 sampled references: 28.44%; 40,562 candidate pairs.
- Country macro F0.5: US 0.4985; India 0.3395.

The dominant measured bottleneck is candidate retrieval: over 71% of true links never reach the scorer. Expand retrieval before spending effort on a complex matching model. These are sampled training holdout results, not a test or leaderboard score. Three metric unit tests pass.

## Supervised pipeline follow-up

The implemented gradient-boosted model scored macro F0.5 **0.8329** on 795 held-out
references, with precision **97.06%**, recall **70.32%**, and blocking recall **78.24%**.
It trained on 149,551 candidate pairs and selected threshold 0.64 using 782 separate
references. This uses a different split from the first rule baseline. Details are
in `model_metrics.json`; the required methodology is in `../Documentation_template.md`.
Full test inference has not been run.

## V2 improvement experiment

On the identical 795-reference development holdout, macro F0.5 improved from
0.8329 to **0.8642**. Precision stayed near 97%, recall increased from 70.32% to
76.41%, and candidate recall rose from 78.24% to 85.14%. Singleton accuracy
increased from 88% to 92%. A feature-only ablation scored 0.8400.

V2 was selected using the separate tuning split (0.8847 versus v1 0.8553).
The development holdout has been reused; a fresh validation sample is needed
before final generalization claims. See `model_comparison.json` for paired results.
