"""Tests for the Distribution abstraction, residual semantics, and projection."""

from __future__ import annotations

import math

import pytest

from immunomath import Distribution, Residual


class TestConstructionAndValidation:
    def test_valid_string_distribution(self) -> None:
        d = Distribution({"G1": 0.8, "G2": 0.15, "G3": 0.05})
        assert d["G1"] == 0.8
        assert d.is_complete
        assert len(d) == 3

    def test_integer_outcomes(self) -> None:
        d = Distribution({0: 0.5, 1: 0.5})
        assert d[1] == 0.5

    def test_tuple_outcomes_genotype_like(self) -> None:
        g = ("A*01:01", "A*02:01")
        d = Distribution({g: 1.0})
        assert d[g] == 1.0

    def test_rejects_unnormalized_over(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            Distribution({"G1": 0.8, "G2": 0.8})

    def test_rejects_unnormalized_under(self) -> None:
        with pytest.raises(ValueError, match="sum to 1"):
            Distribution({"G1": 0.4, "G2": 0.4})

    def test_rejects_negative_probability(self) -> None:
        with pytest.raises(ValueError, match="nonnegative"):
            Distribution({"G1": 1.2, "G2": -0.2})

    def test_rejects_nonfinite_probability(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            Distribution({"G1": math.inf, "G2": 0.0})
        with pytest.raises(ValueError, match="finite"):
            Distribution({"G1": math.nan})

    def test_explicit_zero_is_allowed_and_distinct(self) -> None:
        d = Distribution({"G1": 1.0, "G2": 0.0})
        assert d["G2"] == 0.0
        assert "G2" in d  # explicit zero is listed
        assert "G3" not in d  # omitted

    def test_tolerance_is_respected(self) -> None:
        # Sum is 1 + 1e-10, within default atol 1e-9.
        Distribution({"G1": 0.5, "G2": 0.5 + 1e-10})
        # Sum off by 1e-6 is rejected under default tol.
        with pytest.raises(ValueError):
            Distribution({"G1": 0.5, "G2": 0.5 + 1e-6})

    def test_no_silent_normalization(self) -> None:
        # The library must never repair a truncated distribution automatically.
        with pytest.raises(ValueError):
            Distribution({"G1": 0.3, "G2": 0.3})


class TestResidualSemantics:
    def test_open_distribution_sums_with_residual(self) -> None:
        d = Distribution({"G1": 0.7, "G2": 0.2, Residual(0.1): None})
        assert not d.is_complete
        assert d.residual_mass == pytest.approx(0.1)

    def test_rejects_truncated_without_residual(self) -> None:
        # 0.7 + 0.2 = 0.9; must be rejected unless residual is explicit.
        with pytest.raises(ValueError):
            Distribution({"G1": 0.7, "G2": 0.2})

    def test_truncated_with_residual_is_valid(self) -> None:
        Distribution({"G1": 0.7, "G2": 0.2, Residual(0.1): None})

    def test_at_most_one_residual(self) -> None:
        with pytest.raises(ValueError, match="at most one Residual"):
            Distribution({"G1": 0.5, Residual(0.3): None, Residual(0.2): None})

    def test_residual_mass_bounds(self) -> None:
        with pytest.raises(ValueError):
            Residual(0.0)
        with pytest.raises(ValueError):
            Residual(1.5)
        with pytest.raises(ValueError):
            Residual(-0.1)

    def test_probability_three_way_distinction(self) -> None:
        complete = Distribution({"G1": 0.8, "G2": 0.2})
        # explicit
        assert complete.probability("G1") == 0.8
        # omitted in complete -> 0
        assert complete.probability("G99") == 0.0

        open_d = Distribution({"G1": 0.8, Residual(0.2): None})
        # explicit, scored normally even with residual present
        assert open_d.probability("G1") == 0.8
        # unlisted in open -> unidentifiable
        with pytest.raises(KeyError):
            open_d.probability("G99")

    def test_event_probability_is_identifiable_even_when_open(self) -> None:
        open_d = Distribution({"G1": 0.5, "G2": 0.3, Residual(0.2): None})
        assert open_d.event_probability({"G1", "G2"}) == pytest.approx(0.8)
        assert open_d.event_probability({"G1"}) == pytest.approx(0.5)


class TestRenormalization:
    def test_renormalize_is_explicit_and_opt_in(self) -> None:
        raw = Distribution({"G1": 0.4, "G2": 0.1}, atol=0.5)
        out = raw.renormalize()
        assert out["G1"] == pytest.approx(0.8)
        assert out["G2"] == pytest.approx(0.2)
        assert out.is_complete

    def test_renormalize_rejects_zero_mass(self) -> None:
        d = Distribution({"G1": 0.0, "G2": 0.0}, atol=1.0)
        with pytest.raises(ValueError, match="zero total mass"):
            d.renormalize()


class TestProjection:
    def test_complete_projection_preserves_mass(self) -> None:
        fine = Distribution({"A*01:01": 0.5, "A*01:02": 0.3, "A*02:01": 0.2})
        coarse = fine.project(lambda a: a.split(":")[0])
        assert coarse["A*01"] == pytest.approx(0.8)
        assert coarse["A*02"] == pytest.approx(0.2)
        assert coarse.is_complete
        assert math.fsum(coarse.probabilities) == pytest.approx(1.0)

    def test_partial_projection_moves_dropped_mass_to_residual(self) -> None:
        fine = Distribution({"A*01:01": 0.5, "A*01:02": 0.3, "A*02:01": 0.2})
        partial = fine.project(lambda a: "A*01" if a.startswith("A*01") else None)
        assert partial["A*01"] == pytest.approx(0.8)
        assert partial.residual_mass == pytest.approx(0.2)
        assert not partial.is_complete

    def test_projection_carries_existing_residual(self) -> None:
        fine = Distribution({"A*01:01": 0.5, "A*02:01": 0.3, Residual(0.2): None})
        coarse = fine.project(lambda a: a.split(":")[0])
        assert coarse.residual_mass == pytest.approx(0.2)

    def test_identity_projection_is_noop_on_mass(self) -> None:
        d = Distribution({"G1": 0.6, "G2": 0.4})
        out = d.project(lambda x: x)
        assert out["G1"] == pytest.approx(0.6)
        assert out["G2"] == pytest.approx(0.4)


class TestSerialization:
    def test_roundtrip_strings(self) -> None:
        d = Distribution({"G1": 0.8, "G2": 0.2})
        assert Distribution.from_json(d.to_json()) == d

    def test_roundtrip_mixed_types(self) -> None:
        d = Distribution({(1, 2): 0.5, 7: 0.25, "x": 0.25})
        assert Distribution.from_json(d.to_json()) == d

    def test_roundtrip_with_residual(self) -> None:
        d = Distribution({"G1": 0.8, Residual(0.2): None})
        out = Distribution.from_json(d.to_json())
        assert out == d
        assert out.residual_mass == pytest.approx(0.2)

    def test_rejects_unknown_version(self) -> None:
        with pytest.raises(ValueError, match="version"):
            Distribution.from_dict({"version": 99, "entries": []})


class TestDunder:
    def test_permuted_order_equal(self) -> None:
        a = Distribution({"G1": 0.5, "G2": 0.3, "G3": 0.2})
        b = Distribution({"G3": 0.2, "G1": 0.5, "G2": 0.3})
        assert a == b
        assert hash(a) == hash(b)

    def test_support_excludes_zero(self) -> None:
        d = Distribution({"G1": 0.8, "G2": 0.2, "G3": 0.0})
        assert d.support == frozenset({"G1", "G2"})
