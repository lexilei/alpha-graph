"""C16 `insider_cluster_buy`: clustered insider purchases from Form 4.

Registered 2026-07-13; evaluation protocol pinned pre-computation the same day
(FACTORS.md C16; reports/factor_preregistration.md). Construction, exactly as
pinned:

- Qualifying buy: trans_code P (open-market purchase), non-amendment
  (is_amendment rows excluded), reporting owner flagged officer OR director.
- Person-vs-entity de-clustering. The registered rule excludes owner NAMES
  matching corporate markers (Corp/LLC/LP/Trust/Fund/...), but
  `form4_trans.parquet` carries owner_cik and role flags only — no owner
  name — so the marker rule cannot bind. BUILD-TIME PIN (ledgered
  2026-07-13, pre-evaluation; not a look): structural heuristic instead —
  a qualifying owner is a distinct owner_cik flagged officer/director and
  NOT (ten-percent-owner AND neither officer nor director). Within the
  officer/director filter the exclusion clause is redundant (it can only
  remove tenpct-ONLY owners, already outside the filter); it is kept
  explicit because it is the pinned rule. Known residual: joint-filing
  entity co-owners that carry a director flag are NOT excludable without
  names (80 multi-owner accessions of 9,982 qualifying ones repeat the
  same transaction lines per co-owner CIK).
- Cluster event: per ticker, an event fires at the SECOND distinct
  qualifying owner_cik's FILING date within 21 CALENDAR days of the
  first's filing date (inclusive: a gap of exactly 21 days fires, 22 does
  not; the pinned window is between the two FILING dates). Onset-collapsed:
  once fired, no new event for that ticker until the trailing-21d window
  holds <2 distinct qualifying owners again.
- Per event: ticker, event_date (= 2nd filing date), n_insiders (distinct
  qualifying owners in the trailing window at the fire date), sum_dollar
  (Σ dollar_value of qualifying buys in the window, NaN-skipped), and
  Σ$ / 20d-ADV (ADV = mean of close×volume over the 20 trading days ending
  at the last trading day ≤ event_date) — DIAGNOSTIC ONLY per the pin.
- Calendar-time portfolio (Mitchell–Stafford, long-only equal-weight): with
  pos(e) = index of the first trading day ≥ event_date on the market
  calendar (entry at the next close after the 2nd filing), a name is active
  on trading day t iff pos(e) + 1 <= pos(t) <= pos(e) + 63 for one of its
  events (t+1 convention: the first credited return is the next session's
  close-to-close return; names deduplicate across overlapping events).
  Daily return = equal-weight mean of active names' close-to-close returns
  (adjusted close). Days with zero active names — or none with a return —
  are 0.0, i.e. cash earning nothing, and stay IN the series/regression.
  BUILD-TIME PIN (ledgered 2026-07-13, pre-evaluation; not a look): the
  protocol did not state empty-day handling.
- Events dated outside the market calendar (before its first or after its
  last day) cannot be entered and are flagged tradeable=False; they stay in
  the events cache but contribute nothing to the series.

Caches:
- data/cache/insider_cluster_c16.parquet — daily series: ret, n_active
  (names contributing a return that day), n_active_events.
- data/cache/insider_cluster_c16_events.parquet — one row per event.

Evaluation (the 3 pinned looks, printed by main): FF5+MOM attribution
(`ff_attribution`, excess=False — long-only total returns) on the full
sample and on each half split at the midpoint DATE of the series. Success
bar pinned pre-computation: full-sample alpha HAC t >= +2.0 AND positive
point-estimate alpha in BOTH halves -> candidate; else rejected.

Usage:
    python -m alpha_graph.signals.insider_cluster   # build caches + 3 looks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR

WINDOW_CAL_DAYS = 21   # calendar days between the two filing dates, inclusive
HOLD_TDAYS = 63        # trading days held after entry
ADV_WINDOW = 20        # trading days in the ADV diagnostic
FULL_T_BAR = 2.0       # pinned success bar on the full-sample alpha HAC t

SERIES_CACHE = "insider_cluster_c16.parquet"
EVENTS_CACHE = "insider_cluster_c16_events.parquet"

TRANS_COLS = ["ticker", "owner_cik", "filing_date", "dollar_value"]
EVENT_COLS = ["ticker", "event_date", "n_insiders", "sum_dollar"]


# --------------------------------------------------------------------------- #
# Qualifying buys
# --------------------------------------------------------------------------- #

def qualifying_buys(trans: pd.DataFrame) -> pd.DataFrame:
    """Form 4 rows that count toward a cluster: P, non-amendment, person-like.

    Structural de-clustering pin (no owner names in the parquet): drop owners
    that are ten-percent holders with NEITHER an officer nor a director flag.
    The clause is redundant inside the officer/director filter but is the
    pinned fallback rule, kept verbatim.
    """
    person_like = ~(trans["is_tenpct"] & ~trans["is_officer"] & ~trans["is_director"])
    mask = (
        (trans["trans_code"] == "P")
        & ~trans["is_amendment"]
        & (trans["is_officer"] | trans["is_director"])
        & person_like
    )
    out = trans.loc[mask, TRANS_COLS].copy()
    out["filing_date"] = pd.DatetimeIndex(out["filing_date"]).normalize()
    return out.sort_values(["ticker", "filing_date"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Cluster events
# --------------------------------------------------------------------------- #

def build_events(qual: pd.DataFrame,
                 window_days: int = WINDOW_CAL_DAYS) -> pd.DataFrame:
    """Onset-collapsed cluster events per ticker.

    An event fires at the filing date where the trailing `window_days`
    (calendar, inclusive) window first holds >=2 distinct qualifying owners;
    no re-fire until the window drops below 2 distinct owners again.
    """
    events: list[dict] = []
    win = np.timedelta64(window_days, "D")
    for ticker, sub in qual.groupby("ticker", sort=True):
        sub = sub.sort_values("filing_date")
        dates = sub["filing_date"].to_numpy()
        owners = sub["owner_cik"].to_numpy()
        dollars = sub["dollar_value"].to_numpy(dtype=float)
        fired = False
        for t in np.unique(dates):
            in_win = (dates >= t - win) & (dates <= t)
            # Between filings the window only loses members, so its minimum
            # occupancy since the last filing is the pre-arrival count at t.
            if fired and len(np.unique(owners[in_win & (dates < t)])) < 2:
                fired = False
            n_owners = len(np.unique(owners[in_win]))
            if not fired and n_owners >= 2:
                events.append({
                    "ticker": ticker,
                    "event_date": pd.Timestamp(t),
                    "n_insiders": int(n_owners),
                    "sum_dollar": float(np.nansum(dollars[in_win])),
                })
                fired = True
    return pd.DataFrame(events, columns=EVENT_COLS)


# --------------------------------------------------------------------------- #
# Calendar-time portfolio
# --------------------------------------------------------------------------- #

def build_series(events: pd.DataFrame, market: pd.DataFrame,
                 hold_days: int = HOLD_TDAYS,
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily equal-weight series over the market calendar + flagged events.

    Active on trading day t: pos(event)+1 <= pos(t) <= pos(event)+hold_days,
    pos(event) = first trading day >= event_date. Empty days = 0.0 (cash).
    """
    px = market.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = px.pct_change(fill_method=None)
    cal = px.index
    n = len(cal)

    events = events.copy()
    events["tradeable"] = (
        (events["event_date"] >= cal[0]) & (events["event_date"] <= cal[-1])
        if len(events) else pd.Series(dtype=bool)
    )
    tick_idx = {tk: j for j, tk in enumerate(px.columns)}
    active = np.zeros((n, px.shape[1]), dtype=bool)
    n_active_events = np.zeros(n, dtype=int)
    for row in events.itertuples():
        if not row.tradeable:
            continue
        pos0 = int(cal.searchsorted(row.event_date, side="left"))
        start, stop = pos0 + 1, min(pos0 + hold_days, n - 1)
        if start > stop:
            continue
        n_active_events[start:stop + 1] += 1
        j = tick_idx.get(row.ticker)
        if j is not None:
            active[start:stop + 1, j] = True

    r = rets.to_numpy()
    contrib = active & ~np.isnan(r)
    cnt = contrib.sum(axis=1)
    sums = np.where(contrib, r, 0.0).sum(axis=1)
    ret = np.where(cnt > 0, sums / np.maximum(cnt, 1), 0.0)
    series = pd.DataFrame(
        {"ret": ret, "n_active": cnt, "n_active_events": n_active_events},
        index=pd.DatetimeIndex(cal, name="date"),
    )
    return series, events


def add_adv_diagnostic(events: pd.DataFrame, market: pd.DataFrame,
                       adv_window: int = ADV_WINDOW) -> pd.DataFrame:
    """Attach adv_20d (last trading day <= event_date) and dollar_over_adv."""
    if events.empty:
        return events.assign(adv_20d=pd.Series(dtype=float),
                             dollar_over_adv=pd.Series(dtype=float))
    dv = market.sort_values(["ticker", "date"]).copy()
    dv["date"] = dv["date"].astype("datetime64[ns]")
    dv["adv_20d"] = (
        (dv["close"] * dv["volume"])
        .groupby(dv["ticker"])
        .transform(lambda s: s.rolling(adv_window, min_periods=adv_window).mean())
    )
    events = events.copy()
    events["event_date"] = events["event_date"].astype("datetime64[ns]")
    out = pd.merge_asof(
        events.sort_values("event_date").reset_index(drop=True),
        dv[["ticker", "date", "adv_20d"]].sort_values("date"),
        left_on="event_date", right_on="date", by="ticker",
        direction="backward",
    ).drop(columns=["date"])
    out["dollar_over_adv"] = out["sum_dollar"] / out["adv_20d"]
    return out


def compute(trans: pd.DataFrame, market: pd.DataFrame,
            window_days: int = WINDOW_CAL_DAYS, hold_days: int = HOLD_TDAYS,
            adv_window: int = ADV_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full pipeline: transactions + market -> (daily series, events)."""
    qual = qualifying_buys(trans)
    events = build_events(qual, window_days=window_days)
    series, events = build_series(events, market, hold_days=hold_days)
    events = add_adv_diagnostic(events, market, adv_window=adv_window)
    return series, events


# --------------------------------------------------------------------------- #
# The 3 pinned looks
# --------------------------------------------------------------------------- #

def pinned_looks(series: pd.DataFrame,
                 factors: pd.DataFrame | None = None) -> list[dict]:
    """FF5+MOM attribution (excess=False): full sample + each midpoint half."""
    from alpha_graph.eval.ff_attribution import ff_attribution

    pnl = series["ret"]
    idx = pnl.index
    mid = idx[0] + (idx[-1] - idx[0]) / 2
    windows = [("full", pnl),
               ("first half", pnl.loc[idx <= mid]),
               ("second half", pnl.loc[idx > mid])]
    looks = []
    for name, s in windows:
        res = ff_attribution(s, factors=factors, excess=False)
        sub = series.loc[s.index]
        res.update(
            window=name,
            start=str(s.index[0].date()), end=str(s.index[-1].date()),
            avg_active=float(sub["n_active"].mean()),
            pct_days_active=float((sub["n_active"] >= 1).mean()),
        )
        looks.append(res)
    return looks


def decide(looks: list[dict], t_bar: float = FULL_T_BAR) -> str:
    """Apply the pinned bar: full HAC t >= +2.0 AND alpha > 0 in both halves."""
    full, h1, h2 = looks
    passed = (full["alpha_t_hac"] >= t_bar
              and h1["alpha"] > 0 and h2["alpha"] > 0)
    return "candidate" if passed else "rejected"


# --------------------------------------------------------------------------- #
# Build + evaluate on the cached data
# --------------------------------------------------------------------------- #

def main() -> None:
    trans = pd.read_parquet(CACHE_DIR / "form4_trans.parquet")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet",
                             columns=["date", "ticker", "close", "volume"])
    series, events = compute(trans, market)

    series.to_parquet(CACHE_DIR / SERIES_CACHE)
    events.to_parquet(CACHE_DIR / EVENTS_CACHE, index=False)

    per_year = events["event_date"].dt.year.value_counts().sort_index()
    logger.info(f"events: {len(events)} total, "
                f"{int(events['tradeable'].sum())} tradeable "
                f"({events['ticker'].nunique()} tickers)")
    logger.info("events per year:\n" + per_year.to_string())
    logger.info(f"n_insiders: mean {events['n_insiders'].mean():.2f}, "
                f"max {events['n_insiders'].max()}")
    logger.info(f"dollar_over_adv (diagnostic): median "
                f"{events['dollar_over_adv'].median():.4f}")

    looks = pinned_looks(series)
    for k, res in enumerate(looks, start=1):
        logger.info(
            f"look {k}/3 ({res['window']} {res['start']}->{res['end']}, "
            f"{res['n_days']}d): alpha {res['alpha_ann']:+.2%}/yr "
            f"(HAC t {res['alpha_t_hac']:+.2f}), mkt beta "
            f"{res['loadings']['mkt_rf']:+.3f}, R2 {res['r2']:.3f}, "
            f"avg active {res['avg_active']:.1f}, "
            f"days with >=1 name {res['pct_days_active']:.1%}"
        )
    logger.info(f"pinned bar (full t >= +{FULL_T_BAR}, both halves alpha > 0) "
                f"-> {decide(looks)}")


if __name__ == "__main__":
    main()
