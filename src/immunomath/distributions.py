"""Finite discrete probability distributions over hashable outcomes.

This module provides :class:`Distribution`, the foundational abstraction of
ImmunoMath. Every scoring and diagnostic function in the library operates on a
:class:`Distribution`, never on raw arrays or dictionaries. Centralizing
validation here guarantees that no downstream metric ever sees an invalid,
unnormalized, or silently repaired probability vector.

Design principles
-----------------
*   **Never silently normalize.** A distribution that does not sum to 1 (within
    an explicit tolerance) is rejected at construction time. Renormalization is
    available only through the explicit, opt-in :meth:`Distribution.renormalize`
    method.
*   **Preserve the three-way mass distinction.** The library distinguishes
    between (a) an *omitted* outcome, (b) an outcome with *explicit zero*
    probability, and (c) *unknown residual* probability mass. See
    :class:`Residual` and the :ref:`residual semantics <residual-semantics>`
    discussion below.
*   **Projections preserve mass.** Aggregating a distribution into a coarser
    outcome space via a :class:`Projection` is mass-preserving for total
    (complete) projections and explicitly flagged as partial otherwise.

.. _residual-semantics:

Residual semantics
------------------
A distribution may be *complete* (closed-world) or *open* (it carries explicit
residual mass). The distinction controls how a truth outcome that is **not**
listed is interpreted:

*   In a **complete** distribution, an unlisted outcome has probability exactly
    ``0``. Scoring it with the logarithmic score yields ``+inf``.
*   In an **open** distribution (one carrying :class:`Residual` mass), an
    unlisted outcome *might* be hidden inside the residual. Its exact
    probability is therefore **not identifiable**, and exact-outcome scoring is
    *undefined* rather than zero.
"""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar, runtime_checkable

__all__ = [
    "RESIDUAL",
    "Distribution",
    "Projection",
    "Residual",
]

Outcome = Hashable
OutcomeT = TypeVar("OutcomeT", bound=Hashable, contravariant=True)
CoarseT = TypeVar("CoarseT", bound=Hashable, covariant=True)

#: Default absolute tolerance for the "probabilities sum to one" check.
DEFAULT_ATOL: float = 1e-9


@dataclass(frozen=True)
class Residual:
    """Sentinel representing unknown (unspecified) residual probability mass.

    A :class:`Residual` entry declares that a distribution is *open-world*: the
    listed outcomes account for ``1 - mass`` of the total probability, and the
    remaining ``mass`` is distributed over an unspecified set of outcomes that
    are **not** enumerated.

    This is semantically distinct from both an omitted outcome and an outcome
    with explicit zero probability. See the module docstring for details.

    Parameters
    ----------
    mass : float
        The amount of unspecified residual probability, in the open interval
        ``(0, 1]``. A residual mass of exactly zero is not a residual at all and
        is rejected; use a complete distribution instead.

    Examples
    --------
    >>> from immunomath import Distribution, Residual
    >>> d = Distribution({"G1": 0.7, "G2": 0.2, Residual(0.1): None})
    >>> d.is_complete
    False
    >>> d.residual_mass
    0.1
    """

    mass: float

    def __post_init__(self) -> None:
        """Validate that the residual mass lies in ``(0, 1]``."""
        if not math.isfinite(self.mass):
            raise ValueError(f"Residual mass must be finite, got {self.mass!r}.")
        if not (0.0 < self.mass <= 1.0):
            raise ValueError(
                f"Residual mass must lie in the open interval (0, 1], got {self.mass!r}."
            )


#: A convenience constructor is not provided as a constant because the mass
#: varies; ``RESIDUAL`` is re-exported as the class itself for discoverability.
RESIDUAL = Residual


@runtime_checkable
class Projection(Protocol[OutcomeT, CoarseT]):
    """Protocol for mapping a fine outcome space onto a coarser one.

    A projection is a function ``f: Omega -> Omega'`` that assigns each fine
    outcome to a coarse equivalence class. Projecting a distribution aggregates
    (sums) the probability of all fine outcomes that map to the same coarse
    class, which preserves total probability mass for total functions.

    The protocol is deliberately HLA-agnostic: the statistical kernel knows
    nothing about GL Strings, allele resolution, or nomenclature. Biological
    adapters implement this protocol in a separate layer.

    A class implementing this protocol must be callable, mapping a single fine
    outcome to its coarse class, and must return ``None`` to indicate that an
    outcome is **dropped** (excluded from the coarse space), which makes the
    projection *partial*.
    """

    def __call__(self, outcome: OutcomeT) -> CoarseT | None:
        """Map a fine outcome to its coarse class, or ``None`` to drop it."""
        ...


class Distribution:
    """An immutable, validated finite discrete probability distribution.

    A distribution maps hashable *outcomes* to nonnegative, finite
    probabilities. Outcomes may be strings, integers, tuples, or any other
    hashable genotype identifier. The probabilities are validated at
    construction: they must be finite, nonnegative, and sum to one within an
    absolute tolerance (plus any declared :class:`Residual` mass).

    Parameters
    ----------
    probabilities : Mapping[Hashable, float] or Mapping[Hashable | Residual, float | None]
        A mapping from outcomes to probabilities. At most one key may be a
        :class:`Residual` instance; its value is ignored (use ``None`` by
        convention) because the mass is carried on the ``Residual`` itself.
    atol : float, optional
        Absolute tolerance for the sum-to-one check. Defaults to ``1e-9``.

    Raises
    ------
    ValueError
        If any probability is negative, non-finite, or if the probabilities
        (plus residual mass) do not sum to one within ``atol``, or if more than
        one residual entry is supplied.

    Examples
    --------
    A complete distribution over string genotype identifiers:

    >>> from immunomath import Distribution
    >>> d = Distribution({"G1": 0.80, "G2": 0.15, "G3": 0.05})
    >>> d["G1"]
    0.8
    >>> d.is_complete
    True

    Integer and tuple outcomes are equally valid:

    >>> Distribution({0: 0.5, 1: 0.5})["01"] if False else Distribution({0: 0.5, 1: 0.5})[1]
    0.5
    >>> Distribution({("A*01:01", "A*02:01"): 1.0})[("A*01:01", "A*02:01")]
    1.0
    """

    __slots__ = ("_atol", "_probs", "_residual")

    def __init__(
        self,
        probabilities: Mapping[Any, float | None],
        *,
        atol: float = DEFAULT_ATOL,
    ) -> None:
        if atol < 0:
            raise ValueError(f"atol must be nonnegative, got {atol!r}.")

        probs: dict[Hashable, float] = {}
        residual: Residual | None = None

        for key, value in probabilities.items():
            if isinstance(key, Residual):
                if residual is not None:
                    raise ValueError("A distribution may contain at most one Residual entry.")
                residual = key
                continue
            if value is None:
                raise ValueError(
                    f"Outcome {key!r} has probability None; only Residual keys may "
                    "carry a None value."
                )
            prob = float(value)
            if not math.isfinite(prob):
                raise ValueError(f"Probability for outcome {key!r} must be finite, got {prob!r}.")
            if prob < 0.0:
                raise ValueError(
                    f"Probability for outcome {key!r} must be nonnegative, got {prob!r}."
                )
            probs[key] = prob

        explicit_total = math.fsum(probs.values())
        residual_mass = residual.mass if residual is not None else 0.0
        total = explicit_total + residual_mass

        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=atol):
            detail = (
                f"explicit mass {explicit_total!r} + residual mass {residual_mass!r} = {total!r}"
            )
            raise ValueError(
                f"Probabilities must sum to 1 within atol={atol!r}; got {detail}. "
                "ImmunoMath never silently renormalizes. If renormalization is "
                "mathematically justified, call .renormalize() explicitly, or "
                "represent unaccounted mass with a Residual entry."
            )

        self._probs = probs
        self._residual = residual
        self._atol = atol

    # ------------------------------------------------------------------ #
    # Mapping-like access
    # ------------------------------------------------------------------ #
    def __getitem__(self, outcome: Hashable) -> float:
        """Return the explicit probability of ``outcome``.

        Raises
        ------
        KeyError
            If the outcome is not explicitly present. Use
            :meth:`probability` for closed-world lookup semantics that return
            ``0.0`` for omitted outcomes in a complete distribution.
        """
        return self._probs[outcome]

    def __contains__(self, outcome: object) -> bool:
        """Return whether ``outcome`` is explicitly listed."""
        return outcome in self._probs

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over explicitly listed outcomes (residual excluded)."""
        return iter(self._probs)

    def __len__(self) -> int:
        """Return the number of explicitly listed outcomes (residual excluded)."""
        return len(self._probs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Distribution):
            return NotImplemented
        return self._probs == other._probs and self._residual == other._residual

    def __hash__(self) -> int:
        return hash((frozenset(self._probs.items()), self._residual))

    def __repr__(self) -> str:
        body = dict(self._probs)
        suffix = f", residual={self._residual.mass!r}" if self._residual else ""
        return f"Distribution({body!r}{suffix})"

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def outcomes(self) -> tuple[Hashable, ...]:
        """The explicitly listed outcomes, in insertion order."""
        return tuple(self._probs)

    @property
    def probabilities(self) -> tuple[float, ...]:
        """The probabilities of the explicitly listed outcomes, in insertion order."""
        return tuple(self._probs.values())

    @property
    def residual_mass(self) -> float:
        """The unspecified residual probability mass (``0.0`` if complete)."""
        return self._residual.mass if self._residual is not None else 0.0

    @property
    def is_complete(self) -> bool:
        """Whether this is a complete (closed-world) distribution.

        A complete distribution carries no residual mass: every outcome not
        explicitly listed has probability exactly zero.
        """
        return self._residual is None

    @property
    def atol(self) -> float:
        """The absolute tolerance used for the sum-to-one check."""
        return self._atol

    @property
    def support(self) -> frozenset[Hashable]:
        """The set of outcomes with strictly positive explicit probability."""
        return frozenset(o for o, p in self._probs.items() if p > 0.0)

    # ------------------------------------------------------------------ #
    # Probability lookup with explicit residual semantics
    # ------------------------------------------------------------------ #
    def probability(self, outcome: Hashable) -> float:
        """Return the probability of ``outcome`` under closed-world semantics.

        This is the identifiable probability of a *specific* outcome:

        *   If ``outcome`` is explicitly listed, its assigned probability is
            returned (even when the distribution also carries residual mass).
        *   If ``outcome`` is **not** listed and the distribution is *complete*,
            it has probability ``0.0``.
        *   If ``outcome`` is **not** listed and the distribution is *open*
            (carries residual mass), its probability is **not identifiable** and
            a :class:`KeyError` is raised. The caller must decide how to handle
            unidentifiable probability; the library will not guess.

        Parameters
        ----------
        outcome : Hashable
            The outcome to look up.

        Returns
        -------
        float
            The probability of the outcome.

        Raises
        ------
        KeyError
            If the outcome is absent from an open distribution, in which case
            its probability cannot be determined from the reported distribution.

        Examples
        --------
        >>> from immunomath import Distribution, Residual
        >>> complete = Distribution({"G1": 0.8, "G2": 0.2})
        >>> complete.probability("G1")
        0.8
        >>> complete.probability("G99")   # unlisted in a complete distribution
        0.0
        >>> open_d = Distribution({"G1": 0.8, Residual(0.2): None})
        >>> open_d.probability("G1")
        0.8
        >>> open_d.probability("G99")      # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        KeyError: ...
        """
        if outcome in self._probs:
            return self._probs[outcome]
        if self.is_complete:
            return 0.0
        raise KeyError(
            f"Outcome {outcome!r} is not listed in an open distribution carrying "
            f"residual mass {self.residual_mass!r}. Its exact probability is not "
            "identifiable from the reported distribution. Exact-outcome scoring is "
            "undefined; consider event-level scoring instead."
        )

    def event_probability(self, outcomes: Iterable[Hashable]) -> float:
        """Return the aggregate probability of an *event* (a set of outcomes).

        The event probability is the sum of the explicit probabilities of all
        listed outcomes in the event. Unlike :meth:`probability`, this is always
        identifiable (it is a lower bound that becomes exact for complete
        distributions), because the residual mass is simply not counted toward
        the event. This supports event-level scoring, e.g. "is a given allele
        present?", even when exact-genotype probability is unidentifiable.

        Parameters
        ----------
        outcomes : Iterable[Hashable]
            The outcomes comprising the event.

        Returns
        -------
        float
            The summed probability of the listed event outcomes.

        Examples
        --------
        >>> from immunomath import Distribution, Residual
        >>> d = Distribution({"G1": 0.5, "G2": 0.3, "G3": 0.2})
        >>> d.event_probability({"G1", "G2"})
        0.8
        >>> open_d = Distribution({"G1": 0.5, "G2": 0.3, Residual(0.2): None})
        >>> open_d.event_probability({"G1", "G2"})
        0.8
        """
        wanted = set(outcomes)
        return math.fsum(p for o, p in self._probs.items() if o in wanted)

    # ------------------------------------------------------------------ #
    # Explicit, opt-in renormalization
    # ------------------------------------------------------------------ #
    def renormalize(self) -> Distribution:
        """Return a new complete distribution rescaled to sum to one.

        This is the **only** renormalization path in ImmunoMath, and it is
        explicit and opt-in by design. It divides every explicit probability by
        the current explicit total, discarding any residual mass and producing a
        *complete* distribution.

        Use this only when renormalization is mathematically justified (for
        example, after conditioning on a sub-event). Renormalizing a distribution
        that carried genuine residual mass **destroys information** about the
        unspecified outcomes; the resulting distribution asserts a closed world
        that the original did not.

        Returns
        -------
        Distribution
            A new, complete distribution whose explicit probabilities sum to one.

        Raises
        ------
        ValueError
            If the total explicit probability is zero (nothing to rescale).

        Examples
        --------
        >>> from immunomath import Distribution
        >>> raw = Distribution({"G1": 0.4, "G2": 0.1}, atol=0.5)
        >>> raw.renormalize()[ "G1"]
        0.8
        """
        total = math.fsum(self._probs.values())
        if total <= 0.0:
            raise ValueError("Cannot renormalize a distribution with zero total mass.")
        return Distribution({o: p / total for o, p in self._probs.items()})

    # ------------------------------------------------------------------ #
    # Projection / aggregation into a coarser outcome space
    # ------------------------------------------------------------------ #
    def project(self, projection: Projection[Any, Any]) -> Distribution:
        """Aggregate this distribution into a coarser outcome space.

        Each explicit outcome is mapped to a coarse class by ``projection``; the
        probabilities of all outcomes mapping to the same class are summed. This
        preserves total probability mass whenever the projection is *total*
        (every outcome maps to some class).

        A projection may return ``None`` for an outcome, which **drops** it. If
        any outcome is dropped, the projection is *partial* and the dropped mass
        is re-expressed as :class:`Residual` mass on the result, preserving the
        total. Residual mass on the input is carried through unchanged.

        Parameters
        ----------
        projection : Projection
            A callable mapping each fine outcome to a coarse class, or ``None``
            to drop it.

        Returns
        -------
        Distribution
            The aggregated distribution over coarse classes. Dropped mass (if
            any) is represented as residual mass, so the result always sums to
            one and remains valid.

        Examples
        --------
        Collapsing two-field alleles to one field (a complete projection):

        >>> from immunomath import Distribution
        >>> fine = Distribution({"A*01:01": 0.5, "A*01:02": 0.3, "A*02:01": 0.2})
        >>> coarse = fine.project(lambda a: a.split(":")[0])
        >>> coarse["A*01"]
        0.8
        >>> coarse["A*02"]
        0.2

        A partial projection drops mass into the residual:

        >>> partial = fine.project(lambda a: "A*01" if a.startswith("A*01") else None)
        >>> partial["A*01"]
        0.8
        >>> round(partial.residual_mass, 10)
        0.2
        """
        aggregated: dict[Hashable, float] = {}
        dropped = 0.0
        for outcome, prob in self._probs.items():
            coarse = projection(outcome)
            if coarse is None:
                dropped += prob
                continue
            aggregated[coarse] = aggregated.get(coarse, 0.0) + prob

        dropped_total = math.fsum([dropped]) if dropped else 0.0
        residual_mass = self.residual_mass + dropped_total

        entries: dict[Any, float | None] = dict(aggregated)
        if residual_mass > 0.0:
            entries[Residual(residual_mass)] = None
        # Allow slightly looser tolerance to absorb float summation error from
        # aggregation, while still rejecting genuinely unnormalized inputs.
        return Distribution(entries, atol=max(self._atol, 1e-9))

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict.

        Outcomes are preserved via a ``repr``-free, type-tagged encoding so that
        strings, integers, and tuples round-trip exactly. Residual mass is stored
        under a dedicated key.

        Returns
        -------
        dict
            A JSON-serializable representation with keys ``"version"``,
            ``"entries"`` (a list of ``[encoded_outcome, probability]`` pairs),
            and ``"residual"``.
        """
        return {
            "version": 1,
            "entries": [[_encode_outcome(o), p] for o, p in self._probs.items()],
            "residual": self.residual_mass if not self.is_complete else None,
            "atol": self._atol,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Distribution:
        """Reconstruct a :class:`Distribution` from :meth:`to_dict` output.

        Parameters
        ----------
        data : Mapping
            A mapping produced by :meth:`to_dict`.

        Returns
        -------
        Distribution
            The reconstructed distribution.
        """
        version = data.get("version")
        if version != 1:
            raise ValueError(f"Unsupported serialization version: {version!r}.")
        entries: dict[Any, float | None] = {
            _decode_outcome(enc): float(p) for enc, p in data["entries"]
        }
        residual = data.get("residual")
        if residual is not None:
            entries[Residual(float(residual))] = None
        atol = float(data.get("atol", DEFAULT_ATOL))
        return cls(entries, atol=atol)

    def to_json(self) -> str:
        """Serialize to a JSON string. See :meth:`to_dict`."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str) -> Distribution:
        """Reconstruct a :class:`Distribution` from a JSON string."""
        return cls.from_dict(json.loads(text))


# ---------------------------------------------------------------------- #
# Outcome encoding for serialization (strings, ints, tuples round-trip)
# ---------------------------------------------------------------------- #
def _encode_outcome(outcome: Hashable) -> dict[str, Any]:
    """Encode a hashable outcome into a type-tagged JSON-compatible dict."""
    if isinstance(outcome, str):
        return {"t": "str", "v": outcome}
    if isinstance(outcome, bool):
        return {"t": "bool", "v": outcome}
    if isinstance(outcome, int):
        return {"t": "int", "v": outcome}
    if isinstance(outcome, float):
        return {"t": "float", "v": outcome}
    if isinstance(outcome, tuple):
        return {"t": "tuple", "v": [_encode_outcome(x) for x in outcome]}
    raise TypeError(
        f"Outcome {outcome!r} of type {type(outcome).__name__} is not serializable. "
        "Supported types are str, int, float, bool, and (nested) tuples thereof."
    )


def _decode_outcome(encoded: Mapping[str, Any]) -> Hashable:
    """Invert :func:`_encode_outcome`."""
    tag = encoded["t"]
    value = encoded["v"]
    if tag == "str":
        return str(value)
    if tag == "bool":
        return bool(value)
    if tag == "int":
        return int(value)
    if tag == "float":
        return float(value)
    if tag == "tuple":
        return tuple(_decode_outcome(x) for x in value)
    raise ValueError(f"Unknown outcome type tag: {tag!r}.")
