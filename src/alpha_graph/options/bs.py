"""Black-Scholes pricing and IV inversion.

Ported VERBATIM from vol_smile `src/pricing/black_scholes.py` (engine
v2.1, two adversarial review rounds) so that IVs computed here are
bit-comparable with vol_smile's audited artifacts. Do not "improve" in
place — divergences belong in a new function, not silent edits here.
"""
from __future__ import annotations

import math
from typing import Literal, Tuple

OptionType = Literal["C", "P"]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _d1_d2(spot: float, strike: float, t: float, rate: float, vol: float) -> Tuple[float, float]:
    if t <= 0 or vol <= 0 or spot <= 0 or strike <= 0:
        return 0.0, 0.0
    denom = vol * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t) / denom
    d2 = d1 - denom
    return d1, d2


def bs_price(option_type: OptionType, spot: float, strike: float, t: float, rate: float, vol: float) -> float:
    d1, d2 = _d1_d2(spot, strike, t, rate, vol)
    if option_type == "C":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2)
    if option_type == "P":
        return strike * math.exp(-rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    raise ValueError(f"unknown option_type: {option_type}")


def implied_vol_bisect(
    price: float,
    option_type: OptionType,
    spot: float,
    strike: float,
    t: float,
    rate: float,
    *,
    vol_low: float = 1e-4,
    vol_high: float = 5.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    if price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return None

    low = vol_low
    high = vol_high
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        mid_price = bs_price(option_type, spot, strike, t, rate, mid)
        if abs(mid_price - price) < tol:
            return mid
        if mid_price > price:
            high = mid
        else:
            low = mid
    # Non-convergence within [vol_low, vol_high] means the price is not
    # attainable in range: below intrinsic, above the vol=5 cap, or a
    # crossed/locked quote. Return None (a real solution hits the tol
    # check above long before max_iter) rather than a bracket bound that
    # would masquerade as IV≈0 or IV≈5. All callers are None-safe.
    return None
