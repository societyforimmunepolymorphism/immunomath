# API Reference — Weighted Scoring (TRS-Aware)

Observation-weighted proper scoring, paired model comparison, and
TRS-stratified validation. See
[Weighted Scoring](../mathematical-foundations/weighted-scoring.md) for the
mathematics, the conditional-propriety argument, and the handling of empty
strata and infinite losses.

The statistical kernel (`immunomath.weighting`) is domain-independent: it
consumes externally supplied positive weights and caller-defined stratum labels.
The mapping from a typing-resolution score (TRS) to weights lives in the
optional, non-kernel adapter `immunomath.trs`.

## Weighted scoring kernel

::: immunomath.weighting
    options:
      members:
        - weighted_mean_loss
        - WeightedScore
        - compare_methods_weighted
        - PairedWeightedComparison
        - stratified_comparison
        - StratifiedComparison
        - StratumComparison
        - per_observation_losses
        - ObservationLoss
        - validate_weights
        - LossMetric

## TRS → weight adapter (optional, non-kernel)

::: immunomath.trs
    options:
      members:
        - trs_weights
