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
    ann = sig.sigma.median() * math.sqrt(365 * 24 * 3600)
    print(f"sigma grid {len(sig):,} points, sigma p50 "
          f"{sig.sigma.median():.2e} (annualised {ann:.1%}), "
          f"at floor {(sig.sigma <= SIGMA_FLOOR*1.0001).mean():.1%}")
    # a poisoned price path is what a mixed BTC/ETH feed produced on the
    # first attempt here: sigma came out 1.7e5% annualised and the model
    # column was meaningless. Refuse to score rather than print a table
    # that looks like a result.
    if not 0.05 < ann < 5.0:
        raise SystemExit(
            f"realised sigma is {ann:.1%} annualised, which is not a BTC "
            f"vol. The price path is contaminated -- rebuild the substrate "
            f"before scoring anything on it.")

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

    # The money question. The maker only takes a position where its FV
    # disagrees with the book, so the test is not "is the model good" but
    # "when the model disagrees with the mid, which one is right?"
    #   lean = (y - market) * sign(gauss - market)
    # positive => the disagreement predicts the settlement, i.e. real edge.
    print("\n=== does the model's disagreement with the book predict truth? ===")
    print(f"{'subset':<26}{'n':>8}{'events':>8}{'lean/ct':>10}{'t':>8}"
          f"{'|disagree|':>12}")
    for label, sel in (("all", df),
                       ("bracket", df[df.kind == "bracket"]),
                       ("threshold", df[df.kind == "threshold"]),
                       ("disagreement > 5c",
                        df[(df.gauss - df.market).abs() > 0.05]),
                       ("disagreement > 10c",
                        df[(df.gauss - df.market).abs() > 0.10]),
                       ("maker's own filter",
                        df[(df.gauss > 0.10) & (df.gauss < 0.90)])):
        if len(sel) < 50:
            continue
        lean = (sel.y - sel.market) * np.sign(sel.gauss - sel.market)
        by_ev = lean.groupby(sel.event).mean()
        t = (by_ev.mean() / (by_ev.std(ddof=1) / math.sqrt(len(by_ev)))
             if len(by_ev) > 1 and by_ev.std(ddof=1) > 0 else float("nan"))
        print(f"{label:<26}{len(sel):8,d}{len(by_ev):8d}{lean.mean():+10.4f}"
              f"{t:+8.2f}{(sel.gauss - sel.market).abs().mean():12.4f}")
    print("  lean/ct is dollars per contract, before spread, fees and queue.")

    # NOT a P&L estimate. (y - bid) + (ask - y) is identically the spread,
    # so a "both sides" number is a tautology dressed as a measurement --
    # it would read positive on any market whatsoever. The only content in
    # a fill-free view is the mid's own error, and the asymmetry it induces.
    print("\n=== mid error, and the asymmetry it puts on the two quotes ===")
    print(f"{'subset':<12}{'mid err':>9}{'half-sp':>9}{'rest-bid':>10}"
          f"{'rest-ask':>10}{'events':>8}{'t(mid err)':>11}")
    for label, sel in (("all", df),
                       ("bracket", df[df.kind == "bracket"]),
                       ("threshold", df[df.kind == "threshold"])):
        err = sel.y - sel.market
        by_ev = err.groupby(sel.event).mean()
        t = (by_ev.mean() / (by_ev.std(ddof=1) / math.sqrt(len(by_ev)))
             if len(by_ev) > 1 else float("nan"))
        hs = sel.spread.mean() / 2
        print(f"{label:<12}{err.mean():+9.4f}{hs:9.4f}"
              f"{(err.mean() + hs):+10.4f}{(hs - err.mean()):+10.4f}"
              f"{len(by_ev):8d}{t:+11.2f}")
    print("  mid err > 0 means YES settled more often than the book priced it,")
    print("  which in a 3.4-day window is mostly the drift of one BTC path,")
    print("  not an edge: it moves both quotes in opposite directions and")
    print("  nets to zero for anyone quoting both sides.")


if __name__ == "__main__":
    main()
