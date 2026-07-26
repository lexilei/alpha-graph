"""Risk metrics for a book of binary contracts on one underlying.

Why this module exists
----------------------
The Kalshi FV-maker gated risk on ``net = sum(position.values())``, which
treats a long 64150-bracket and a short 64250-bracket as offsetting linear
exposure.  They do not offset: each is an indicator on a *different* region
of the settlement price, and both can pay against you.  In the 2026-07-25
production session that gate read a net of -7.8 contracts at a moment when
the true worst-case settlement loss was ~$74 on a $49.53 account
(``reports/audit_maker_2026-07-26.md``).

The correct risk variable for a book of binaries on a common underlying is
the worst settlement it can produce:

    worst_case_payoff = min over settlement price S of  sum_i payout_i(S)

where a long YES receives its size inside its own region and a short YES
(= long NO, fully prepaid) receives its size outside it.  Add cash and you
have a hard, mark-free floor under account equity.  It is exact and cheap:
the payout is piecewise constant in S with breakpoints only at the contract
boundaries, so one probe per region is exhaustive.  Contracts whose region
is unknown are assumed to pay nothing, so the result is always a valid
lower bound, never an optimistic one.

Everything here is pure: no network, no clock, no filesystem.  The maker
owns I/O; this module owns the arithmetic, so it can be tested against the
recorded session instead of against mocks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

__all__ = [
    "Region",
    "region_from_strikes",
    "worst_case_payoff",
    "gross_contracts",
    "HighWaterBudget",
    "parse_exchange_ts",
]


# --------------------------------------------------------------------------
# settlement regions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Region:
    """The set of settlement prices for which a contract's YES side pays 1.

    Kalshi convention (matches the maker's own fair-value formula
    ``P(S>floor) - P(S>cap)``): the region is the half-open interval
    ``(lo, hi]``.  A threshold market ("above K") has ``hi = inf``.
    """

    lo: float
    hi: float = math.inf

    def __post_init__(self) -> None:
        if not (self.lo < self.hi):
            raise ValueError(f"empty region ({self.lo}, {self.hi}]")

    def pays(self, s: float) -> bool:
        return self.lo < s <= self.hi


def _f(x) -> float | None:
    if x in (None, ""):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def region_from_strikes(floor_strike, cap_strike=None) -> Region | None:
    """Build a Region from an exchange market payload's strike fields.

    Kalshi ships three shapes: ``greater`` (floor only), ``less`` (cap only)
    and ``between`` (both).  Returns None when the payload cannot pin the
    region down, so the caller falls back to the adversarial treatment
    rather than inventing geometry.
    """
    lo, hi = _f(floor_strike), _f(cap_strike)
    # absent is a shape; unparseable is a fault. Treating a bracket whose cap
    # failed to parse as an unbounded threshold would widen the payout region
    # and make the risk gate optimistic, which is the one direction this
    # function must never fail in.
    if floor_strike not in (None, "") and lo is None:
        return None
    if cap_strike not in (None, "") and hi is None:
        return None
    if lo is None and hi is None:
        return None
    if lo is None:
        return Region(-math.inf, hi)      # "less": pays at or below the cap
    if hi is None:
        return Region(lo)                 # "greater": pays above the floor
    if hi <= lo:
        return None
    return Region(lo, hi)


# --------------------------------------------------------------------------
# worst-case settlement payoff
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WorstCase:
    value: float            # min over S of the book's settlement payout, in dollars
    at_price: float | None  # a settlement price attaining it
    n_unknown: int          # positions with no region, assumed to pay nothing
    gross: float            # sum of |position|, for logging only


def _payout(q: float, reg: Region | None, s: float) -> float:
    """Dollars this position receives if the underlying settles at ``s``.

    Kalshi is fully prepaid, so payouts are never negative: a long YES of q
    receives q when its region pays, a short YES of q (i.e. long |q| NO)
    receives |q| when it does not.  A position whose region is unknown is
    assumed to receive nothing -- the adversarial case, which keeps the
    result a valid lower bound.
    """
    if reg is None:
        return 0.0
    pays = reg.pays(s)
    if q > 0:
        return q if pays else 0.0
    return -q if not pays else 0.0


def worst_case_payoff(positions: dict[str, float],
                      regions: dict[str, Region]) -> WorstCase:
    """Lower bound on what the whole book pays out, minimised over settlement.

    ``positions`` maps ticker -> signed YES contracts (negative = short YES =
    long NO).  ``regions`` maps ticker -> Region wherever the geometry is
    known; missing entries are assumed to pay nothing.

    Add this to cash to get a hard, mark-free floor under account equity:
    whatever the underlying does, the account cannot settle below it.  That
    is the number the production maker never computed -- it gated on
    ``sum(positions.values())``, which at 17:01:31Z on 2026-07-25 read -7.8
    contracts while the book's worst settlement was $40.22 below the
    session's opening equity.

    Exact when every region is known: the payout is piecewise constant in the
    settlement price with breakpoints only at region boundaries, so one probe
    per region is exhaustive.
    """
    live = {t: float(q) for t, q in positions.items() if float(q) != 0.0}
    n_unknown = sum(1 for t in live if regions.get(t) is None)
    gross = sum(abs(float(q)) for q in positions.values())

    bounds = sorted({b for t in live
                     for b in ((regions[t].lo, regions[t].hi)
                               if regions.get(t) is not None else ())
                     if math.isfinite(b)})
    if not bounds:
        return WorstCase(sum(_payout(q, regions.get(t), 0.0)
                             for t, q in live.items()), None, n_unknown, gross)

    span = max(1.0, bounds[-1] - bounds[0])
    probes = [bounds[0] - span]
    probes += [(a + b) / 2.0 for a, b in zip(bounds, bounds[1:])]
    probes.append(bounds[-1] + span)

    best_v, best_s = math.inf, None
    for s in probes:
        v = sum(_payout(q, regions.get(t), s) for t, q in live.items())
        if v < best_v:
            best_v, best_s = v, s
    return WorstCase(best_v, best_s, n_unknown, gross)


def gross_contracts(positions: dict[str, float]) -> float:
    return sum(abs(float(q)) for q in positions.values())


# --------------------------------------------------------------------------
# loss budget
# --------------------------------------------------------------------------

@dataclass
class HighWaterBudget:
    """A loss budget anchored to the account's high-water mark.

    The production maker keyed its budget on the UTC calendar day.  When the
    day rolled at 00:05Z on 2026-07-26 the stop that had fired 71 minutes
    earlier silently re-armed with a fresh $15 on an account that had already
    lost 49% of its principal.  A budget that resets on a clock can be spent
    once per tick of that clock; this one can only be raised by actually
    making money.
    """

    budget: float
    high_water: float | None = None

    def observe(self, equity: float) -> None:
        """Record a (robust) equity reading; ratchets the high-water mark."""
        if equity is None or not math.isfinite(equity):
            return
        if self.high_water is None or equity > self.high_water:
            self.high_water = float(equity)

    def floor(self) -> float | None:
        """Equity level at which trading must stop, or None while unarmed."""
        if self.high_water is None:
            return None
        return self.high_water - self.budget

    def breached(self, equity: float) -> bool:
        f = self.floor()
        return f is not None and equity is not None and equity < f

    def headroom(self, equity: float) -> float:
        f = self.floor()
        return math.inf if f is None else equity - f

    def serialise(self) -> str:
        return "" if self.high_water is None else f"{self.high_water:.6f}"

    @classmethod
    def load(cls, budget: float, blob: str | None) -> "HighWaterBudget":
        hw = None
        if blob:
            try:
                hw = float(blob.strip().split()[-1])
            except (ValueError, IndexError):
                hw = None
            if hw is not None and not math.isfinite(hw):
                hw = None
        return cls(budget=budget, high_water=hw)


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?:\.(\d+))?"
                    r"(Z|[+-]\d{2}:?\d{2})?$")


def parse_exchange_ts(ts: str | None) -> datetime | None:
    """Parse an exchange timestamp with variable fractional-second width.

    Kalshi emits 3, 4, 5 and 6 fractional digits in the same fills payload
    (all four widths appear in the 2026-07-25 session).  The maker compared
    these as raw strings.  Decimal fractions do compare correctly
    lexicographically, but a timestamp with *no* fractional part does not:
    ``"...:59Z" > "...:59.148Z"`` because ``'Z' > '.'``, which would advance a
    high-water mark past a batch of fills and drop them silently.  Parse
    instead of trusting the string.
    """
    if not ts:
        return None
    m = _TS_RE.match(ts.strip())
    if not m:
        return None
    base, frac, tzs = m.groups()
    micro = int((frac or "0")[:6].ljust(6, "0"))
    try:
        dt = datetime.strptime(base.replace(" ", "T"), "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    dt = dt.replace(microsecond=micro, tzinfo=timezone.utc)
    if tzs in (None, "Z", "z"):
        return dt
    body = tzs[1:].replace(":", "")
    off = timedelta(hours=int(body[:2]), minutes=int(body[2:4]))
    return dt - off if tzs[0] == "+" else dt + off
