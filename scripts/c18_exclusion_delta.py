"""C18 look 1/1: nt_filing_veto exclusion delta on the EW PIT-member long book.

The single pinned look (protocol row 2026-07-13 in
reports/factor_preregistration.md): WITH vs WITHOUT vetoed names on the
equal-weight PIT-member monthly-rebalanced long book.

Book (plain pandas, portfolio engine untouched):
  - rebalance at each month-end trading close (market_data.parquet calendar)
    where PIT membership resolves (pit_universe.membership_asof; its 60-day
    carry-forward bound past the last 2026-01-14 snapshot ends forming at
    2026-02-27, held to 2026-03-31 — no stub beyond the last full month);
  - holdings = all PIT members with a close at the forming date, equal
    weight, buy-and-hold to the next month-end (close-to-close daily
    compounding, drifting weights, halted days forward-filled at 0%);
  - WITHOUT book: identical, minus names inside a veto window
    (data/cache/nt_veto.parquet) AT THE REBALANCE DATE — exclusion acts at
    forming dates only, never mid-month.

Metrics: annualized return = geometric from the daily series ((1+r).prod()
** (252/n) - 1); Sharpe = daily mean/std(ddof=1) x sqrt(252), rf = 0.

Pinned bar, applied mechanically: exclusion improves annualized return by
>= 10 bp/yr AND does not degrade Sharpe -> adopted as a portfolio-layer veto
rule (not a factor); else rejected. Honest caveat (pinned): NT filings are
rare in this universe — if the delta rests on a handful of name-months it is
noise either way, and an underpowered pass still requires both
point-estimate conditions.

Usage:
    python scripts/c18_exclusion_delta.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.data.pit_universe import load_membership, membership_asof
from alpha_graph.signals.nt_veto import book_holdings, ew_book_returns

MARKET_PATH = CACHE_DIR / "market_data.parquet"
VETO_PATH = CACHE_DIR / "nt_veto.parquet"

MIN_DELTA_BP = 10.0     # pinned success bar, bp/yr of annualized return


def annualized_return(r: pd.Series) -> float:
    return float((1 + r).prod() ** (252 / len(r)) - 1)


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def main() -> None:
    md = pd.read_parquet(MARKET_PATH, columns=["date", "ticker", "close"])
    close_wide = md.pivot(index="date", columns="ticker", values="close")
    cal = close_wide.index
    windows = pd.read_parquet(VETO_PATH)
    snaps = load_membership()

    month_ends = [g.max() for _, g in cal.to_series().groupby([cal.year, cal.month])]
    valid = [membership_asof(snaps, d) is not None for d in month_ends]
    j = valid.index(False) if False in valid else len(month_ends)
    if not (all(valid[:j]) and not any(valid[j:])):
        raise ValueError("membership validity is not prefix-contiguous over month-ends")
    rebalances = month_ends[:min(j, len(month_ends) - 1)]
    end = month_ends[len(rebalances)]
    logger.info(f"{len(rebalances)} rebalances {rebalances[0].date()} -> "
                f"{rebalances[-1].date()}, last book held to {end.date()}")

    members_at = lambda d: membership_asof(snaps, d)  # noqa: E731
    h_with = book_holdings(close_wide, rebalances, members_at)
    h_without = book_holdings(close_wide, rebalances, members_at, windows)

    excl = {d: sorted(set(h_with[d]) - set(h_without[d])) for d in rebalances}
    name_months = [(d, t) for d in rebalances for t in excl[d]]
    hit = [w for w in windows.itertuples()
           if any(w.veto_from <= d <= w.veto_to and w.ticker in excl[d]
                  for d in rebalances)]

    print("\n" + "=" * 64)
    print("  C18 LOOK 1/1 — NT VETO EXCLUSION DELTA (EW PIT LONG BOOK)")
    print("=" * 64)
    sizes = [len(h_with[d]) for d in rebalances]
    print(f"\nbook: {len(rebalances)} rebalances {rebalances[0].date()} -> "
          f"{rebalances[-1].date()} (held to {end.date()}), "
          f"avg cross-section {np.mean(sizes):.1f}")
    print(f"veto windows: {len(windows)} ({windows['ticker'].nunique()} tickers); "
          f"{len(hit)} actually veto a held name" if len(windows)
          else "veto windows: 0")
    print(f"\nexcluded name-months: {len(name_months)} "
          f"({len(set(t for _, t in name_months))} distinct names), "
          f"avg {len(name_months) / len(rebalances):.3f} per rebalance")
    for w in hit:
        ds = [d for d in rebalances if w.veto_from <= d <= w.veto_to
              and w.ticker in excl[d]]
        print(f"  {w.ticker:6s} NT window {w.veto_from.date()} -> "
              f"{w.veto_to.date()}: excluded at {len(ds)} rebalances "
              f"({ds[0].date()} .. {ds[-1].date()})")
    for d, t in name_months:
        print(f"    {d.date()}  {t}")

    r_with = ew_book_returns(close_wide, rebalances, h_with, end)
    r_without = ew_book_returns(close_wide, rebalances, h_without, end)

    ann_w, ann_wo = annualized_return(r_with), annualized_return(r_without)
    sh_w, sh_wo = sharpe(r_with), sharpe(r_without)
    delta_bp = (ann_wo - ann_w) * 1e4
    delta_sh = sh_wo - sh_w

    print(f"\n{'':14s}{'WITH':>12s}{'WITHOUT':>12s}{'delta':>14s}")
    print(f"{'ann return':14s}{ann_w * 100:11.3f}%{ann_wo * 100:11.3f}%"
          f"{delta_bp:+13.2f}bp")
    print(f"{'Sharpe':14s}{sh_w:12.4f}{sh_wo:12.4f}{delta_sh:+14.4f}")
    print(f"{'daily obs':14s}{len(r_with):12d}{len(r_without):12d}")

    passed = (delta_bp >= MIN_DELTA_BP) and (sh_wo >= sh_w)
    print(f"\npinned bar: delta >= +{MIN_DELTA_BP:.0f} bp/yr "
          f"({'MET' if delta_bp >= MIN_DELTA_BP else 'NOT met'}: {delta_bp:+.2f}) "
          f"AND no Sharpe degradation "
          f"({'MET' if sh_wo >= sh_w else 'NOT met'}: {delta_sh:+.4f})")
    print(f"verdict: {'ADOPTED as portfolio-layer veto rule' if passed else 'REJECTED'}")
    if len(hit) < 30:
        print(f"caveat (pinned): the delta rests on {len(hit)} veto events "
              f"({len(name_months)} name-months) over {len(rebalances)} months — "
              "event-count noise either way; underpowered.")
    print("=" * 64)


if __name__ == "__main__":
    main()
