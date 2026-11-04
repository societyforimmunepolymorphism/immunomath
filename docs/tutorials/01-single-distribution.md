# Tutorial 1 — Evaluating a Single Probability Distribution

This tutorial walks through evaluating one prediction end-to-end. Every code
block is runnable and reproducible; there is no randomness and no external data.

## The scenario

An inference engine predicts, for one subject, a probability distribution over
three candidate genotypes. The (independently known) truth is `G1`.

```python
import immunomath as im

prediction = im.Distribution(
    {
        "G1": 0.80,
        "G2": 0.15,
        "G3": 0.05,
    }
)
truth = "G1"
```

## Step 1 — Proper scoring rules (primary)

```python
>>> import immunomath as im
>>> prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> round(im.negative_log_score("G1", prediction), 6)
0.223144
>>> round(im.brier_score("G1", prediction), 10)
0.065
```

Both are `0` for a perfect prediction and increase as the prediction gets worse.
The log score is unbounded and punishes overconfidence harshly; the Brier score
is bounded in `[0, 2]`. These are your headline quality numbers.

## Step 2 — Descriptive diagnostics (context)

```python
>>> import immunomath as im
>>> prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> im.top_k_accuracy("G1", prediction, k=1)
1.0
>>> r = im.rank_of_truth("G1", prediction)
>>> (r.min_rank, r.max_rank)
(1, 1)
>>> im.probability_of_truth("G1", prediction)
0.8
```

The truth was the single most probable candidate (rank 1), to which the engine
assigned 80% probability.

## Step 3 — Describe the prediction's uncertainty

```python
>>> import immunomath as im
>>> prediction = im.Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
>>> round(im.entropy(prediction), 4)
0.6129
>>> round(im.effective_number_of_outcomes(prediction), 4)
1.8457
>>> im.num_candidates(prediction)
3
```

There are 3 candidates, but the prediction behaves like a choice among roughly
1.85 equally likely options — it is fairly confident. **Remember:** entropy
describes confidence, not correctness.

## Step 4 — Contrast with an overconfident wrong prediction

```python
>>> import immunomath as im
>>> overconfident = im.Distribution({"G1": 0.01, "G2": 0.99})
>>> round(im.negative_log_score("G1", overconfident), 4)   # truth is G1
4.6052
>>> im.top_k_accuracy("G1", overconfident, k=1)
0.0
```

The log score of `4.6` is roughly 20× worse than our good prediction's `0.22`,
precisely because the engine confidently backed the wrong genotype.

## What you learned

- Lead with **proper scoring rules** (`negative_log_score`, `brier_score`).
- Use **descriptive diagnostics** (accuracy, rank, entropy) for context, never
  as a substitute for quality.
- Overconfident wrong predictions are penalized steeply by the log score — by
  design.

Next: evaluating a cohort of many subjects (Milestone 2).
