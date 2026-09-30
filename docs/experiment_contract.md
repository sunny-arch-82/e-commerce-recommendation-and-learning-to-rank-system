# Temporal and experiment contract

## Four periods, three prediction origins

- Candidate prefix: events timestamp < A (2026-01-21).
- Ranker features/candidates at A; graded labels from [A,B), B=2026-01-31.
- Validation features/candidates at B; labels [B,C), C=2026-02-10.
- Test features/candidates at C; labels [C,D), D=2026-02-20.

Every snapshot uses events strictly before origin and catalog items created strictly
before origin. Boundaries include explicit UTC timezones. All interactions at one timestamp
stay on the same side. Item availability is a static catalog assumption, not historical
inventory replay. Do not apply this data model to time-varying availability without adapting it.

BPR is fit once on prefix A. LambdaMART is fit on [A,B) query labels; early stopping
uses candidate-only [B,C) validation groups. Model selection/tuning use validation only.
There is no refit on validation/test labels. Test inference uses updated history at C:
validation events are now legitimately historical features, not training targets.
Artifacts persist only history <C and catalog creation <C; no test events are stored.
The recommendation service is a fixed C snapshot; it does not receive live events.
All tune trials use identical data boundaries, with configurable training choices.

The default ranker training cohort is users with eligible future events in [A,B).
Each query is one user at A with a ten-day outcome horizon, not a user session.
Training adds observed outcome positives and sampled unlabeled negatives; validation
and test never inject future positives into the retrieved candidate set. This intentional
training construction changes the distribution and can bias relevance scores. Ranker
scores are ordering scores, not conversion probabilities. The long label window and
one-origin training are prototype limitations; production would use repeated historical
requests/exposures, propensity treatment, and delayed outcome labeling.

Informative groups only (at least two distinct relevance grades) are used for ranker fit
and early stopping; final evaluation includes every query with at least one eligible
click/cart/purchase. Selection metrics on filtered early-stopping groups differ from
full validation metrics in evaluate/tune. Both are validation, never test.

## BPR and unobserved items

CSR entries sum configurable positive event weights. Users are sampled uniformly,
then observed positives proportionally to their aggregate weights. Uniform unobserved
negatives form a pairwise logistic objective with L2 regularization. Thus BPR is weighted
positive sampling, not ALS confidence weighting. No missing event means a proven dislike.
BPR excludes all observed prefix items when sampling; ranker negatives exclude all
future observed items in its supervised label window. A ranker negative has no observed
future interaction in that window, not a known explicit negative judgment.

Candidate retrieval blends BPR, historical category/global popularity and a freshness
reserve. Existing-user scores use frozen embeddings; new-user factors are absent.
New users use category popularity when observed session/history exists, otherwise global
popularity. No secret generated user preferences are read by feature code.
New items receive no learned factor, score zero for CF, and can enter freshness reserves.
New categories/brands encode as -1 because vocabularies are frozen at A.

## Metrics and scope

Future per-item relevance is max event grade: view 1, click 2, cart 3, purchase 4.
Precision/Recall/MAP/hit/MRR use relevance>=2 by default. NDCG uses graded gains
2^grade-1, so weak views still have nonzero gain. MAP divisor=min(eligible positives,K).
Precision divides by requested K even if fewer items returned; MRR is truncated at K.
Users without eligible positive targets are skipped, and counts are reported. Future
items created after recommendation time or previously purchased/excluded items are
removed from eligible truth and reported as excluded. These metrics measure novel,
recommendable observed behavior, not all future purchases or actual user utility.
No exposure/propensity data exists. Popularity-biased synthetic outcomes can reward
popularity; comparisons are not causal A/B tests or proof of business improvement.

Configured candidate recall is a warm ranker retrieval ceiling, not a cold fallback ceiling:
cold serving bypasses the collaborative pool. All baselines and ranker share exclusion
policy/catalog/time, but popularity baselines score the full eligible catalog while CF
and ranker score the blended candidate pool. Diversity/novelty are descriptive only;
no artificial diversity reranking is implemented. Native importance is not causal.

## Reproducibility and upgrades

Config contains all event weights, seeds, temporal boundaries, retrieval quotas, negative
sampling, features, ranker parameters and K values. Artifacts store BPR NPZ (no pickle),
LightGBM text booster, frozen encoders, columns, metadata and checksums. Dataset hash
covers the three CSV files. Use new output paths for new models. Loaders reject unknown
artifact formats/checksum mismatch. Scores and metadata schemas use version 1; future
incompatible upgrades should increment it and retrain, not silently reinterpret old files.
No production registry, automatic promotion, authentication, or live model reload is included.
