"""Surgical repair of data/cache/market_data.parquet (2026-07-14 price audit).

Idempotent and reversible: re-running produces the same file; the pre-repair
panel is kept once at data/cache/market_data_prerepair_20260714.parquet.

The four repairs, exactly as pinned:

1. WRONG-IDENTITY PREFIXES dropped entirely (the yfinance symbol carried a
   different security's price line before the listed date):
   - SW    < 2024-07-08: Smurfit Kappa OTC line (never an S&P member; stale
     OTC fills), not Smurfit WestRock.
   - AMCR  < 2019-06-11: pre-NYSE-listing line, not Amcor plc.
   - VRT   < 2020-02-10: GS Acquisition Holdings SPAC shell, not Vertiv.
2. SYNTHETIC ROW dropped: GDDY 2015-03-31 (pre-IPO placeholder — volume 0,
   close = the $20 offer price; made the first panel return a fake +30.8%).
3. MISSING NAMES re-added: INTU ZBH ZBRA ZTS were silently absent (yfinance
   batch failure the old market.py logged at DEBUG). Re-downloaded over the
   full panel window via alpha_graph.data.market.download_prices, restricted
   to the panel's existing trading calendar, forward returns computed with
   the identical formula. New rows are adjusted as of the re-download date,
   not the original panel snapshot date — each ticker's series is internally
   consistent, which is all returns/momentum consume.
4. SPINOFF SEAM MASKING: six name-days carry a fake one-day "return" (a
   corporate-action adjustment artifact, not a price move). Prices are left
   UNTOUCHED; instead every forward-return cell whose window crosses the
   seam transition (prev close -> seam close) is set to NaN:
   ret_1d at pos-1; ret_5d at pos-5..pos-1; ret_21d at pos-21..pos-1
   (pos = the seam day's position in the ticker's calendar). The seam-day
   row itself keeps its forward returns (post-seam closes are internally
   consistent). The list is written to data/cache/market_data_seams.parquet
   (ticker, seam_date, reason) so consumers can exclude.

KNOWN LIMITATION (registered follow-up, not fixed here): close/open/high/low
stay untouched, so anything derived from CLOSE that crosses a seam remains
contaminated — panel lookback features (momentum_21d/252_21, volatility_21d),
the C10 customer-momentum builders, and the C16/C20 calendar-time books'
daily pct_change all read close. Fixing those requires splicing a documented
true seam-day return from external data. This repair fixes the CONSUMED
forward-return targets (ret_1d/5d/21d) only.

Usage:
    python scripts/repair_market_data.py [--skip-download]
"""

from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

MARKET_CACHE = CACHE_DIR / "market_data.parquet"
SEAMS_CACHE = CACHE_DIR / "market_data_seams.parquet"
BACKUP_CACHE = CACHE_DIR / "market_data_prerepair_20260714.parquet"

RET_HORIZONS = [(1, "ret_1d"), (5, "ret_5d"), (21, "ret_21d")]

# ticker -> first valid date; rows strictly BEFORE it are a different security.
WRONG_IDENTITY_PREFIXES = {
    "SW": "2024-07-08",
    "AMCR": "2019-06-11",
    "VRT": "2020-02-10",
}

# (ticker, date) rows dropped outright.
SYNTHETIC_ROWS = [("GDDY", "2015-03-31")]

MISSING_TICKERS = ["INTU", "ZBH", "ZBRA", "ZTS"]

# (ticker, seam_date, reason) — the six audited spinoff/separation seams.
SEAMS = [
    ("DHR", "2016-07-05",
     "Fortive spinoff ex-date; panel +61.2% one-day return is an "
     "adjustment artifact"),
    ("EXPE", "2011-12-21",
     "TripAdvisor spinoff + 1:2 reverse split; panel -33.6% one-day "
     "return is an adjustment artifact"),
    ("DD", "2019-06-03",
     "Corteva spinoff (DowDuPont breakup); panel +17.8% one-day return "
     "is an artifact (distribution leg missing)"),
    ("LDOS", "2013-09-30",
     "SAIC split-off + reverse split; panel +15.0% one-day return is an "
     "adjustment artifact"),
    ("HWM", "2020-04-01",
     "Arconic Corp spinoff over-adjustment; panel +7.2%, true return "
     "approx -9 to -13%"),
    ("HPQ", "2015-11-02",
     "HP/HPE separation; panel +13.0% one-day return suspected ~8pp "
     "overstated"),
]


# --------------------------------------------------------------------------- #
# Pure repair steps (importable; synthetic-testable)
# --------------------------------------------------------------------------- #

def drop_wrong_identity(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows before each ticker's first valid date. Idempotent."""
    dropped: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for ticker, first_valid in WRONG_IDENTITY_PREFIXES.items():
        bad = (df["ticker"] == ticker) & (df["date"] < pd.Timestamp(first_valid))
        dropped[ticker] = int(bad.sum())
        keep &= ~bad
    return df.loc[keep], dropped


def drop_synthetic_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop the pinned synthetic (ticker, date) rows. Idempotent."""
    keep = pd.Series(True, index=df.index)
    for ticker, date in SYNTHETIC_ROWS:
        keep &= ~((df["ticker"] == ticker) & (df["date"] == pd.Timestamp(date)))
    n = int((~keep).sum())
    return df.loc[keep], n


def recompute_returns_for(df: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Recompute ret_1d/5d/21d for `tickers` with market.compute_returns'
    exact formula (forward pct_change on close). Other tickers untouched."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    sel = df["ticker"].isin(tickers)
    sub = df.loc[sel]
    for horizon, col in RET_HORIZONS:
        df.loc[sel, col] = sub.groupby("ticker")["close"].transform(
            lambda s, h=horizon: s.pct_change(h).shift(-h)
        )
    return df


def mask_seam_returns(df: pd.DataFrame,
                      seams: list[tuple[str, str, str]] | None = None,
                      ) -> tuple[pd.DataFrame, int]:
    """NaN every forward-return cell whose window crosses a seam transition.

    ret_kd at row i covers close(i) -> close(i+k); it crosses the seam at
    position pos iff pos-k <= i <= pos-1. Returns the frame and the number
    of cells newly set to NaN (0 on a second run — idempotent).
    """
    seams = SEAMS if seams is None else seams
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    n_cells = 0
    for ticker, seam_date, _reason in seams:
        sub_idx = df.index[df["ticker"] == ticker]
        hit = np.flatnonzero(
            (df.loc[sub_idx, "date"] == pd.Timestamp(seam_date)).to_numpy())
        if len(hit) == 0:
            logger.warning(f"[{ticker}] seam {seam_date} not in panel — skipped")
            continue
        pos = int(hit[0])
        for h, col in RET_HORIZONS:
            rows = sub_idx[max(pos - h, 0):pos]
            n_cells += int(df.loc[rows, col].notna().sum())
            df.loc[rows, col] = np.nan
    return df, n_cells


def seams_frame(seams: list[tuple[str, str, str]] | None = None) -> pd.DataFrame:
    seams = SEAMS if seams is None else seams
    out = pd.DataFrame(seams, columns=["ticker", "seam_date", "reason"])
    out["seam_date"] = pd.to_datetime(out["seam_date"])
    return out


# --------------------------------------------------------------------------- #
# Missing-ticker re-add (network; retried once)
# --------------------------------------------------------------------------- #

def download_missing(missing: list[str], calendar: pd.DatetimeIndex,
                     ) -> pd.DataFrame | None:
    """Full-window download for `missing`, clipped to the panel calendar.

    The panel window is 2011-04-06 -> 2026-04-02; download_prices' end is
    exclusive, so end_date is the day after, and years_back=15 puts the
    start at 2011-04-03. One retry on failure; None if both attempts fail.
    """
    from alpha_graph.data.market import compute_returns, download_prices

    end_date = (calendar.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    new = pd.DataFrame()
    for attempt in (1, 2):
        try:
            new = download_prices(missing, years_back=15, end_date=end_date)
        except Exception as e:  # network/rate failures
            logger.warning(f"download attempt {attempt} raised: {e}")
            new = pd.DataFrame()
        if not new.empty and set(new["ticker"]) == set(missing):
            break
        logger.warning(f"download attempt {attempt} incomplete "
                       f"(got {sorted(set(new['ticker'])) if not new.empty else []})")
    if new.empty:
        return None

    n0 = len(new)
    new = new[new["date"].isin(calendar)].copy()   # never extend the calendar
    if len(new) < n0:
        logger.info(f"clipped {n0 - len(new)} rows outside the panel calendar")
    new = compute_returns(new)
    return new


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-download", action="store_true",
                        help="skip the missing-ticker re-add (offline mode)")
    args = parser.parse_args()

    df = pd.read_parquet(MARKET_CACHE)
    df["date"] = pd.to_datetime(df["date"])
    cols = list(df.columns)
    n_start, tickers_start = len(df), df["ticker"].nunique()

    if not BACKUP_CACHE.exists():
        shutil.copy2(MARKET_CACHE, BACKUP_CACHE)
        logger.info(f"pre-repair panel backed up to {BACKUP_CACHE}")

    # 1-2. drops
    df, dropped = drop_wrong_identity(df)
    df, n_synth = drop_synthetic_rows(df)
    logger.info(f"wrong-identity prefixes dropped: "
                + ", ".join(f"{t} {n}" for t, n in dropped.items())
                + f"; synthetic rows dropped: {n_synth}")

    # returns of the touched tickers recomputed on the post-drop series
    # (prefix drops leave kept rows' forward returns unchanged; recompute
    # anyway so the file is self-consistent by construction)
    touched = list(WRONG_IDENTITY_PREFIXES) + [t for t, _ in SYNTHETIC_ROWS]
    df = recompute_returns_for(df, touched)

    # 3. missing tickers
    calendar = pd.DatetimeIndex(np.sort(df["date"].unique()))
    missing = [t for t in MISSING_TICKERS if t not in set(df["ticker"])]
    if not missing:
        logger.info("missing tickers already present — no download needed")
    elif args.skip_download:
        logger.warning(f"--skip-download: proceeding WITHOUT {missing}")
    else:
        new = download_missing(missing, calendar)
        if new is None or new.empty:
            logger.error(f"RE-ADD FAILED after retry — proceeding WITHOUT "
                         f"{missing}; re-run this script to try again")
        else:
            got = sorted(set(new["ticker"]))
            still = sorted(set(missing) - set(got))
            if still:
                logger.error(f"RE-ADD PARTIAL: got {got}, still missing {still}")
            new["ticker"] = new["ticker"].astype(df["ticker"].dtype)
            new["volume"] = new["volume"].astype("float64")
            df = pd.concat([df[cols], new[cols]], ignore_index=True)
            logger.info(f"re-added {len(new)} rows for {got} "
                        f"({new.groupby('ticker')['date'].min().to_dict()} first dates)")

    # 4. seam masking (after any recompute/merge — order matters)
    df, n_cells = mask_seam_returns(df)
    logger.info(f"seam masking: {n_cells} forward-return cells set to NaN "
                f"across {len(SEAMS)} seams (0 = already masked)")

    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["date"] = df["date"].astype("datetime64[ms]")   # original panel dtype
    df[cols].to_parquet(MARKET_CACHE, index=False)
    logger.info(f"saved {len(df)} rows ({df['ticker'].nunique()} tickers) "
                f"to {MARKET_CACHE} [was {n_start} rows / {tickers_start}]")

    seams_frame().to_parquet(SEAMS_CACHE, index=False)
    logger.info(f"seam registry written to {SEAMS_CACHE} ({len(SEAMS)} rows)")


if __name__ == "__main__":
    main()
