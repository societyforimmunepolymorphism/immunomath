"""Evaluate a single probability distribution end-to-end.

Run with:

    python examples/evaluate_single_distribution.py

This mirrors Tutorial 1 in the documentation. It uses no randomness and no
external data, so its output is fully reproducible.
"""

from __future__ import annotations

import immunomath as im


def main() -> None:
    """Score one prediction against a known truth and print the results."""
    prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
    truth = "G1"

    print("Prediction:", dict(zip(prediction.outcomes, prediction.probabilities, strict=True)))
    print(f"Truth:       {truth}\n")

    print("Proper scoring rules (primary; lower is better)")
    print(f"  negative_log_score : {im.negative_log_score(truth, prediction):.6f} nats")
    print(f"  brier_score        : {im.brier_score(truth, prediction):.6f} (range [0, 2])\n")

    print("Descriptive diagnostics (context only)")
    rank = im.rank_of_truth(truth, prediction)
    print(f"  top_1_accuracy     : {im.top_k_accuracy(truth, prediction, k=1)}")
    print(f"  rank_of_truth      : [{rank.min_rank}, {rank.max_rank}]")
    print(f"  probability_of_truth: {im.probability_of_truth(truth, prediction)}")
    print(f"  entropy            : {im.entropy(prediction):.6f} nats")
    print(f"  effective_outcomes : {im.effective_number_of_outcomes(prediction):.6f}")
    print(f"  num_candidates     : {im.num_candidates(prediction)}")


if __name__ == "__main__":
    main()
