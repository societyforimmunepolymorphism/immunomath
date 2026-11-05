r"""Demonstrate TRS-aware weighted scoring on two synthetic HF-assignment models.

Run with:

    python examples/trs_weighted_comparison.py

This example constructs a deliberately imbalanced synthetic cohort and compares
an *incumbent* and a *candidate* probabilistic model under four lenses:

1. the conventional (donor-weighted) log score,
2. the TRS-weighted log score (upweighting poorly resolved donors),
3. the TRS-balanced macro score (equal aggregate weight per TRS stratum), and
4. per-stratum paired differences.

It also shows the Brier metric for the same cohort.

The scenario (synthetic data only)
----------------------------------
* A large **high-TRS** majority has nearly deterministic predictions. The
  candidate improves them only marginally.
* A small **low-TRS** minority has substantial predictive uncertainty. The
  candidate is *worse* for them.

Averaged conventionally, the many small high-TRS gains outweigh the few low-TRS
losses, so the candidate looks better. TRS weighting shrinks that margin and the
TRS-balanced macro-average reverses it, exposing the low-TRS regression — the
exact situation this milestone is designed to detect.

Statistical note
----------------
Weights are derived **only** from each donor's typing-resolution score (TRS), a
fixed covariate. They never depend on the truth or on either model's
predictions, so each underlying proper score remains proper conditional on the
covariate. This example uses synthetic data and seeded randomness confined to
the script; ImmunoMath performs no simulation and computes no TRS.
"""

from __future__ import annotations

import numpy as np

import immunomath as im
from immunomath.trs import trs_weights

SEED = 7
N_HIGH = 60  # high-resolution, near-deterministic donors
N_LOW = 8  # low-resolution, uncertain donors
OUTCOMES = ("G1", "G2", "G3", "G4")

# TRS weighting parameters (explicit, as required — no scientific default).
LAMBDA = 4.0
GAMMA = 2.0


def _dist(probs: dict[str, float]) -> im.Distribution:
    """Build a validated distribution, repairing float residue onto the max."""
    vec = np.array([probs.get(o, 0.0) for o in OUTCOMES], dtype=float)
    vec = np.round(vec, 12)
    vec[int(np.argmax(vec))] += 1.0 - vec.sum()
    return im.Distribution({o: float(p) for o, p in zip(OUTCOMES, vec, strict=True)})


def build_cohort() -> tuple[
    list[str], list[im.Distribution], list[im.Distribution], list[float], list[str]
]:
    """Return (truths, incumbent, candidate, trs, strata) for the scenario."""
    rng = np.random.default_rng(SEED)
    truths: list[str] = []
    incumbent: list[im.Distribution] = []
    candidate: list[im.Distribution] = []
    trs: list[float] = []
    strata: list[str] = []

    # High-TRS majority: both models near-certain; candidate clearly sharper so
    # the many small per-donor gains sum to a sizeable total improvement.
    for _ in range(N_HIGH):
        p_inc = rng.uniform(0.70, 0.78)
        p_cand = min(p_inc + rng.uniform(0.12, 0.17), 0.97)
        rem_inc = (1.0 - p_inc) / 3.0
        rem_cand = (1.0 - p_cand) / 3.0
        truths.append("G1")
        incumbent.append(_dist({"G1": p_inc, "G2": rem_inc, "G3": rem_inc, "G4": rem_inc}))
        candidate.append(_dist({"G1": p_cand, "G2": rem_cand, "G3": rem_cand, "G4": rem_cand}))
        trs.append(float(rng.uniform(0.9, 1.0)))
        strata.append("high")

    # Low-TRS minority: uncertain; candidate pulls mass AWAY from the truth, but
    # only mildly, so the conventional mean still narrowly favors the candidate.
    for _ in range(N_LOW):
        p_inc = rng.uniform(0.45, 0.6)
        p_cand = p_inc - rng.uniform(0.08, 0.16)  # candidate is worse here
        rem_inc = (1.0 - p_inc) / 3.0
        rem_cand = (1.0 - p_cand) / 3.0
        truths.append("G1")
        incumbent.append(_dist({"G1": p_inc, "G2": rem_inc, "G3": rem_inc, "G4": rem_inc}))
        candidate.append(_dist({"G1": p_cand, "G2": rem_cand, "G3": rem_cand, "G4": rem_cand}))
        trs.append(float(rng.uniform(0.0, 0.25)))
        strata.append("low")

    return truths, incumbent, candidate, trs, strata


def main() -> None:
    """Build the scenario and print the four side-by-side views."""
    truths, incumbent, candidate, trs, strata = build_cohort()
    weights = trs_weights(trs, lam=LAMBDA, gamma=GAMMA)

    print(f"Synthetic cohort: {N_HIGH} high-TRS + {N_LOW} low-TRS donors (seed={SEED}).")
    print(f"TRS weighting: w = 1 + {LAMBDA}*(1 - r)^{GAMMA}\n")

    # --- 1 & 2: conventional vs TRS-weighted log score (paired) ----------- #
    log_cmp = im.compare_methods_weighted(
        truths,
        incumbent,
        candidate,
        weights,
        metric="log",
        label_incumbent="incumbent",
        label_candidate="candidate",
    )
    print("Log score (lower is better), paired incumbent vs candidate")
    print(
        f"  conventional mean:  incumbent={log_cmp.mean_loss_incumbent:.4f}  "
        f"candidate={log_cmp.mean_loss_candidate:.4f}"
    )
    print(
        f"  TRS-weighted mean:  incumbent={log_cmp.weighted_mean_loss_incumbent:.4f}  "
        f"candidate={log_cmp.weighted_mean_loss_candidate:.4f}"
    )
    print(f"  conventional improvement (inc - cand): {log_cmp.mean_improvement:+.4f}")
    print(f"  TRS-weighted improvement:              {log_cmp.weighted_mean_improvement:+.4f}")
    print(
        f"  improved={log_cmp.n_improved}  worsened={log_cmp.n_worsened}  "
        f"unchanged={log_cmp.n_unchanged}  N_eff={log_cmp.effective_sample_size:.1f}"
    )
    _verdict("  conventional verdict:", log_cmp.mean_improvement)
    _verdict("  TRS-weighted verdict:", log_cmp.weighted_mean_improvement)

    # --- 3 & 4: TRS-balanced macro score and per-stratum differences ------ #
    strat = im.stratified_comparison(
        truths,
        incumbent,
        candidate,
        strata,
        stratum_order=["low", "high"],
    )
    print("\nTRS-stratified log score")
    for s in strat.strata:
        print(
            f"  [{s.label:>4}] n={s.n:>3}  "
            f"inc={s.mean_log_incumbent:.4f}  cand={s.mean_log_candidate:.4f}  "
            f"improvement={s.mean_improvement_log:+.4f}  "
            f"(improved={s.n_improved}, worsened={s.n_worsened})"
        )
    if strat.empty_strata:
        print(f"  empty strata (excluded from macro-average): {list(strat.empty_strata)}")
    print(f"  overall (donor-weighted) improvement: {strat.overall_mean_improvement_log:+.4f}")
    print(f"  TRS-balanced macro improvement:       {strat.macro_mean_improvement_log:+.4f}")
    _verdict("  overall verdict:", strat.overall_mean_improvement_log)
    _verdict("  macro-balanced verdict:", strat.macro_mean_improvement_log)

    # --- Brier metric, same cohort ---------------------------------------- #
    brier_cmp = im.compare_methods_weighted(
        truths,
        incumbent,
        candidate,
        weights,
        metric="brier",
    )
    print("\nBrier score (sum form, lower is better)")
    print(f"  conventional improvement: {brier_cmp.mean_improvement:+.4f}")
    print(f"  TRS-weighted improvement: {brier_cmp.weighted_mean_improvement:+.4f}")

    print(
        "\nInterpretation: the conventional mean favors the candidate because the\n"
        "large high-TRS majority each improves. TRS weighting shrinks that margin\n"
        "toward neutral, and the TRS-balanced macro-average reverses the verdict\n"
        "entirely (and the TRS-weighted Brier improvement turns negative),\n"
        "revealing that the candidate degrades the poorly resolved donors. Neither\n"
        "metric establishes clinical utility; donor-ranking evaluation is a\n"
        "separate downstream task."
    )


def _verdict(prefix: str, improvement: float) -> None:
    """Print whether an improvement favors the candidate, incumbent, or neither."""
    if improvement > 0:
        print(f"{prefix} candidate preferred")
    elif improvement < 0:
        print(f"{prefix} incumbent preferred")
    else:
        print(f"{prefix} tie")


if __name__ == "__main__":
    main()
