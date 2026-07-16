"""Per-day ATM IV from a raw/clean EOD option chain, with parity-implied spot.

Built for the options-xs family (IDEAS.md I1/I2/I3). Two deliberate choices:

1. **Spot comes from put-call parity, not a price file.** The single-name
   store's spot.parquet closes are split-adjusted while strikes are
   as-traded (the 2026-07-15 QA finding: AAPL pre-2020 / AMZN pre-2022
   zero out under a price-file moneyness test). Per (quote_date,
   expiration): ATM strike = argmin |call_mid − put_mid| over strikes with
   both legs bid>0; spot ≈ K + C − P. This is exact under parity with
   zero rate/dividends over the horizon — good to a few tenths of a
   percent at 20-60 DTE, which is all moneyness banding and ATM selection
   need. IV inversion then uses the same implied spot, so rate/dividend
   error shows up symmetrically in both legs and mostly cancels in the
   call/put-averaged ATM IV.

2. **Method mirrors vol_smile's audited `build_atm_iv_series.py`** (expiry
   nearest target DTE within tolerance, strike nearest spot, bisect both
   legs' mids, average) so the SPY validation is apples-to-apples: the
   only intended difference on SPY is parity-spot vs spot-file.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .bs import implied_vol_bisect

RIGHT_MAP = {"CALL": "C", "PUT": "P", "C": "C", "P": "P"}


@dataclass(frozen=True)
class AtmIv:
    quote_date: str
    expiration: str
    dte: int
    strike: float
    spot_impl: float
    iv_call: float | None
    iv_put: float | None

    @property
    def iv(self) -> float | None:
        legs = [v for v in (self.iv_call, self.iv_put) if v is not None]
        return sum(legs) / len(legs) if legs else None


def paired_mids(day: pd.DataFrame) -> pd.DataFrame:
    """Keep-last dedup, bid>0 both legs, wide CALL/PUT mid table.

    Expects columns: quote_date, expiration, strike, right, bid, ask,
    created. Returns one row per (expiration, strike) that has BOTH legs
    live, with call_mid / put_mid columns.
    """
    df = day.copy()
    df = df[df["bid"] > 0]
    if df.empty:
        return pd.DataFrame(columns=["expiration", "strike", "call_mid", "put_mid"])
    df["created"] = pd.to_datetime(df["created"])
    df = (
        df.sort_values("created")
        .drop_duplicates(["expiration", "strike", "right"], keep="last")
    )
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["right"] = df["right"].map(RIGHT_MAP)
    wide = df.pivot_table(
        index=["expiration", "strike"], columns="right", values="mid", aggfunc="first"
    )
    if "C" not in wide.columns or "P" not in wide.columns:
        return pd.DataFrame(columns=["expiration", "strike", "call_mid", "put_mid"])
    wide = wide.dropna(subset=["C", "P"]).reset_index()
    return wide.rename(columns={"C": "call_mid", "P": "put_mid"})


def parity_spot(pairs: pd.DataFrame) -> pd.Series:
    """Implied spot per expiration: at the strike minimizing |C − P|,
    spot ≈ K + C − P. Input: paired_mids() output. Returns
    Series[expiration] -> spot."""
    if pairs.empty:
        return pd.Series(dtype=float)
    gap = (pairs["call_mid"] - pairs["put_mid"]).abs()
    idx = gap.groupby(pairs["expiration"]).idxmin()
    atm = pairs.loc[idx]
    return pd.Series(
        (atm["strike"] + atm["call_mid"] - atm["put_mid"]).values,
        index=atm["expiration"].values,
    )


def atm_iv_for_day(
    day: pd.DataFrame,
    quote_date: str,
    *,
    target_dte: int = 37,
    dte_tolerance: int = 9,
    rate: float = 0.04,
) -> AtmIv | None:
    """vol_smile-mirror ATM IV: expiry nearest target_dte within tolerance,
    strike nearest (parity-implied) spot, bisect both legs' mids, average."""
    pairs = paired_mids(day)
    if pairs.empty:
        return None
    qd = pd.Timestamp(quote_date)
    exp = pd.to_datetime(pairs["expiration"])
    pairs = pairs.assign(dte=(exp - qd).dt.days)
    cand = pairs[(pairs["dte"] - target_dte).abs() <= dte_tolerance]
    if cand.empty:
        return None
    # Nearest expiry to target; ties break to the LATER expiry — pinned
    # explicitly (the audited vol_smile series resolves ties by stable-sort
    # row order, which empirically lands on the later expiry; we match the
    # behavior with a deterministic rule instead of inheriting row order).
    per_exp = cand.groupby("expiration")["dte"].first().reset_index()
    per_exp["dist"] = (per_exp["dte"] - target_dte).abs()
    per_exp = per_exp.sort_values(["dist", "dte"], ascending=[True, False])
    best_exp = per_exp["expiration"].iloc[0]
    chain = cand[cand["expiration"] == best_exp]
    spots = parity_spot(chain)
    if spots.empty:
        return None
    spot = float(spots.iloc[0])
    if spot <= 0:
        return None
    row = chain.loc[(chain["strike"] - spot).abs().idxmin()]
    dte = int(row["dte"])
    t = dte / 365.0
    iv_c = implied_vol_bisect(float(row["call_mid"]), "C", spot, float(row["strike"]), t, rate)
    iv_p = implied_vol_bisect(float(row["put_mid"]), "P", spot, float(row["strike"]), t, rate)
    return AtmIv(
        quote_date=str(quote_date),
        expiration=str(row["expiration"]),
        dte=dte,
        strike=float(row["strike"]),
        spot_impl=spot,
        iv_call=iv_c,
        iv_put=iv_p,
    )
