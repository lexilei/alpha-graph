"""Who is better calibrated on Kalshi BTC hourly markets: the book, or our FV?

This is the question the whole FV-maker rests on and it has never been asked
against a correct settlement. The live session's "+$53.51 theoretical edge"
was Sum(our_fv - our_price), which measures only that we quoted around our
own model. Here both forecasters are scored against the settled outcome.

Forecasters compared, on identical rows:
  market   the book mid, (yes_bid + yes_ask) / 2
  gauss    the maker's own model: Phi(ln(S/floor)/sigma*sqrt(tau)) minus the
           cap leg, sigma = the same two-speed EWMA of 10s log returns with
           the same 3e-5 floor the live loop used
  base     the unconditional yes rate, as a floor on "did anyone learn anything"

Scoring is Brier (mean squared error on a 0/1 outcome), and the paired
difference is aggregated BY SETTLEMENT EVENT, not by row: every strike in one
hourly event shares a single settlement price, and snapshots ten seconds
apart are not independent draws. The effective sample size is the number of
hourly events, ~100 over the recorded window, and the reported t is computed
on that basis.

Usage: .venv/bin/python scripts/kalshi_fv_calibration.py [--tau-min 600]
                                                         [--tau-max 2700]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
BOOK_F = CACHE / "kalshi_replay_book.parquet"
SPOT_F = CACHE / "kalshi_replay_spot.parquet"

SIGMA_FLOOR = 3e-5      # same floor as the live maker
EWMA_HL_S = 300.0
SAMPLE_S = 10           # the live loop's cycle, so the estimator matches


def sigma_path(spot: pd.DataFrame) -> pd.DataFrame:
    """EWMA sigma-per-root-second on a 10s grid, reproducing the live Vol."""
    s = spot.set_index("ts_s").spot.sort_index()
    grid = np.arange(s.index.min(), s.index.max() + 1, SAMPLE_S)
    px = s.reindex(grid).ffill().dropna()
    lr = np.log(px / px.shift(1))
    r2 = (lr ** 2) / SAMPLE_S                      # per-second variance
    a = 1 - 0.5 ** (SAMPLE_S / EWMA_HL_S)
    var = r2.ewm(alpha=a, adjust=False).mean()
    out = pd.DataFrame({"ts_s": var.index, "spot": px.reindex(var.index).values,
                        "sigma": np.maximum(np.sqrt(var.values), SIGMA_FLOOR)})
    return out.dropna()


def gauss_fv(S, floor, cap, sigma, tau):
    st = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = np.where(np.isfinite(floor), norm.cdf(np.log(S / floor) / st), 1.0)
        hi = np.where(np.isfinite(cap), norm.cdf(np.log(S / cap) / st), 0.0)
    return np.clip(lo - hi, 0.0, 1.0)


def _paired(df, a, b, key="event"):
    """Per-event mean Brier difference (a - b) with a clustered t."""
    g = df.groupby(key).apply(
        lambda d: pd.Series({"a": ((d[a] - d.y) ** 2).mean(),
                             "b": ((d[b] - d.y) ** 2).mean()}),
        include_groups=False)
    d = g.a - g.b
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return d.mean(), n, float("nan")
    return d.mean(), n, d.mean() / (d.std(ddof=1) / math.sqrt(n))


def calib_table(df, col, label):
    b = df.assign(bin=np.minimum(9, (df[col] * 10).astype(int)))
    t = b.groupby("bin").agg(n=("y", "size"), pred=(col, "mean"),
                             realised=("y", "mean"))
    t["diff"] = t.realised - t.pred
    print(f"\n  {label}: n={len(df):,}  mean pred {df[col].mean():.3f}  "
          f"realised {df.y.mean():.3f}  Brier {((df[col]-df.y)**2).mean():.4f}")
    print(t.to_string(float_format=lambda v: f"{v:8.3f}"))


def main() -> None:
    argv = sys.argv[1:]
    tau_min = float(argv[argv.index("--tau-min") + 1]) if "--tau-min" in argv else 600.0
    tau_max = float(argv[argv.index("--tau-max") + 1]) if "--tau-max" in argv else 2700.0

    book = pd.read_parquet(BOOK_F)
    spot = pd.read_parquet(SPOT_F)
    sig = sigma_path(spot)
    print(f"book rows {len(book):,}  markets {book.ticker.nunique():,}")
    print(f"sigma grid {len(sig):,} points, sigma p50 "
          f"{sig.sigma.median():.2e} (annualised "
          f"{sig.sigma.median()*math.sqrt(365*24*3600):.1%}), "
          f"at floor {(sig.sigma <= SIGMA_FLOOR*1.0001).mean():.1%}")

    df = book[(book.tau_s >= tau_min) & (book.tau_s <= tau_max)].copy()
    df["ts_s"] = df.ts_ms // 1000
    df = df.merge(sig, on="ts_s", how="inner")
    df = df.dropna(subset=["yes_bid", "yes_ask"])
    df = df[(df.yes_ask > df.yes_bid)]
    df["market"] = (df.yes_bid + df.yes_ask) / 2
    df["y"] = (df.result == "yes").astype(float)
    df["event"] = df.ticker.str.split("-").str[1]
    df["kind"] = np.where(df.ticker.str.rsplit("-", n=1).str[-1].str.startswith("B"),
                          "bracket", "threshold")
    df["floor_f"] = df["floor"].astype(float).fillna(-np.inf)
    df["cap_f"] = df["cap"].astype(float).fillna(np.inf)
    df["gauss"] = gauss_fv(df.spot.values, df.floor_f.values, df.cap_f.values,
                           df.sigma.values, df.tau_s.values)
    df["base"] = df.y.mean()
    df["spread"] = df.yes_ask - df.yes_bid

    print(f"\nscored rows {len(df):,}  markets {df.ticker.nunique():,}  "
          f"settlement events {df.event.nunique()}  "
          f"yes rate {df.y.mean():.3f}  median spread {df.spread.median():.3f}")

    print("\n=== Brier, lower is better (paired by settlement event) ===")
    print(f"{'subset':<26}{'n rows':>9}{'events':>8}"
          f"{'market':>9}{'gauss':>9}{'base':>8}{'g-m':>9}{'t':>7}")
    for label, sel in (("all", df),
                       ("bracket", df[df.kind == "bracket"]),
                       ("threshold", df[df.kind == "threshold"]),
                       ("tau < 20min", df[df.tau_s < 1200]),
                       ("tau > 30min", df[df.tau_s > 1800]),
                       ("near-money (mkt .2-.8)",
                        df[(df.market > 0.2) & (df.market < 0.8)]),
                       ("maker's own filter",
                        df[(df.gauss > 0.10) & (df.gauss < 0.90)])):
        if len(sel) < 50:
            continue
        bm = ((sel.market - sel.y) ** 2).mean()
        bg = ((sel.gauss - sel.y) ** 2).mean()
        bb = ((sel.base - sel.y) ** 2).mean()
        d, n, t = _paired(sel, "gauss", "market")
        print(f"{label:<26}{len(sel):9,d}{sel.event.nunique():8d}"
              f"{bm:9.4f}{bg:9.4f}{bb:8.4f}{d:+9.4f}{t:+7.2f}")
    print("  g-m > 0 means the Gaussian model is WORSE than the book mid.")

    print("\n=== calibration ===")
    calib_table(df, "market", "market mid")
    calib_table(df, "gauss", "gaussian FV")
    for kind in ("bracket", "threshold"):
        calib_table(df[df.kind == kind], "gauss", f"gaussian FV, {kind}")

    # what a passive quote at the touch would have earned, before queue and
    # adverse selection: the ceiling on the market-making edge
    print("\n=== passive edge at the touch (settlement-marked, no fill model) ===")
    for label, sel in (("all", df),
                       ("bracket", df[df.kind == "bracket"]),
                       ("threshold", df[df.kind == "threshold"])):
        buy = (sel.y - sel.yes_bid).mean()
        sell = (sel.yes_ask - sel.y).mean()
        print(f"  {label:<12} rest-bid {buy:+.4f}  rest-ask {sell:+.4f}  "
              f"both {(buy + sell):+.4f}  half-spread {sel.spread.mean()/2:.4f}")
    print("  a fair market gives rest-bid ~= rest-ask ~= half the spread;")
    print("  a negative side is where the flow is picking that quote off.")


if __name__ == "__main__":
    main()
