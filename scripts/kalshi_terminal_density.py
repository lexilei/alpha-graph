"""B — does a better terminal density beat the Kalshi book?

A established that the maker's Gaussian fair value is worse-calibrated than
the book mid in every subset, and that leaning on their disagreement pays
nothing (`reports/kalshi_replay_A_2026-07-26.md`). The remaining question is
whether that is the *density's* fault. The maker assumes

    log(S_T / S_t) ~ Normal(0, sigma_t^2 * tau)

with sigma_t an EWMA of 10s returns extrapolated by sqrt(tau). Real BTC
terminal distributions over 10-45 minutes are neither Gaussian in shape nor
sqrt-scaled in level, so this fits several alternatives and scores them all
against settled outcomes, against each other, and against the book.

Fitting is strictly out of sample: the shape is estimated on Binance 1m
klines from BEFORE the recorded window, and scored on the Kalshi window.
Refitting on the scoring window would reproduce the circularity that made
the live session's "+$53.51 theoretical edge" meaningless.

Candidates
  mid       the book, the benchmark to beat
  gauss     the maker's own model, unchanged
  gauss_c   the same, with sigma multiplied by one fitted constant -- tests
            whether the problem is merely the level of vol
  student   standardised Student-t, nu fitted out of sample
  empirical the empirical CDF of standardised returns, no shape assumed
  blend     0.5*(empirical + mid), to see whether the model adds anything
            ON TOP of the price rather than instead of it

Usage
  .venv/bin/python scripts/kalshi_terminal_density.py [--fit-months 6]
"""

from __future__ import annotations

import io
import math
import sys
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kalshi_fv_calibration import (  # noqa: E402
    SAMPLE_S, SIGMA_FLOOR, sigma_path,
)

CACHE = ROOT / "data" / "cache"
BOOK_F = CACHE / "kalshi_replay_book.parquet"
SPOT_F = CACHE / "kalshi_replay_spot.parquet"
FIT_F = CACHE / "btc_1m_fit.parquet"

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
KLINE_COLS = ["open_time", "open", "high", "low", "close", "volume",
              "close_time", "quote_volume", "n_trades", "tb_vol", "tb_quote",
              "ignore"]
# horizons the maker actually quotes, in minutes
HORIZONS = (10, 15, 20, 30, 45)
EWMA_HL_S = 300.0


def fetch_1m(months: list[str]) -> pd.DataFrame:
    """BTCUSDT 1m klines from the public bucket, one monthly zip each."""
    frames = []
    for m in months:
        url = f"{BUCKET}/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{m}.zip"
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                blob = r.read()
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: {str(e)[:80]}", flush=True)
            continue
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = z.namelist()[0]
            d = pd.read_csv(z.open(name), header=None, names=KLINE_COLS)
        d = d[["open_time", "close"]]
        frames.append(d)
        print(f"  {m}: {len(d):,} bars", flush=True)
    if not frames:
        raise SystemExit("no fit data downloaded")
    out = pd.concat(frames).sort_values("open_time").drop_duplicates("open_time")
    # the bucket switched open_time from ms to us at some point in 2025
    unit = "us" if out.open_time.iloc[-1] > 1e15 else "ms"
    out["ts_s"] = (out.open_time // (1_000_000 if unit == "us" else 1_000))
    return out[["ts_s", "close"]].reset_index(drop=True)


def standardised_returns(px: pd.DataFrame, horizon_min: int) -> np.ndarray:
    """z = log(S_{t+h}/S_t) / (sigma_hat_t * sqrt(h)) on the fit history.

    sigma_hat is the maker's own estimator -- EWMA of squared log returns
    with a 300s halflife and the same 3e-5 floor -- so what is being tested
    is the SHAPE of the terminal distribution, holding the vol estimator
    fixed. If the shape is fine and only the level is wrong, gauss_c will
    say so.
    """
    s = px.set_index("ts_s").close
    step = 60
    lr = np.log(s / s.shift(1))
    r2 = (lr ** 2) / step
    a = 1 - 0.5 ** (step / EWMA_HL_S)
    var = r2.ewm(alpha=a, adjust=False).mean()
    sig = np.maximum(np.sqrt(var), SIGMA_FLOOR)
    h = horizon_min * 60
    fwd = np.log(s.shift(-horizon_min) / s)
    z = fwd / (sig * math.sqrt(h))
    return z.replace([np.inf, -np.inf], np.nan).dropna().values


def fit_shapes(z: np.ndarray) -> dict:
    """Fit the level scalar, a Student-t, and keep the empirical CDF."""
    c = float(np.std(z, ddof=1))          # one scalar: how wrong is the level
    nu, loc, scale = stats.t.fit(z, floc=0.0)
    grid = np.linspace(-8, 8, 4001)
    ecdf = np.searchsorted(np.sort(z), grid, side="right") / len(z)
    return {"n": len(z), "c": c, "skew": float(stats.skew(z)),
            "kurt": float(stats.kurtosis(z, fisher=False)),
            "nu": float(nu), "t_scale": float(scale),
            "grid": grid, "ecdf": ecdf}


def make_cdfs(fit: dict):
    def f_gauss(z):
        return stats.norm.cdf(z)

    def f_gauss_c(z):
        return stats.norm.cdf(z / fit["c"])

    def f_student(z):
        return stats.t.cdf(z / fit["t_scale"], df=fit["nu"])

    def f_emp(z):
        return np.interp(z, fit["grid"], fit["ecdf"], left=0.0, right=1.0)

    return {"gauss": f_gauss, "gauss_c": f_gauss_c,
            "student": f_student, "empirical": f_emp}


def fv_from_cdf(F, S, floor, cap, sigma, tau):
    """P(floor < S_T <= cap) under the standardised CDF F."""
    st = sigma * np.sqrt(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_lo = np.where(np.isfinite(floor), np.log(floor / S) / st, -np.inf)
        z_hi = np.where(np.isfinite(cap), np.log(cap / S) / st, np.inf)
    lo = np.where(np.isfinite(z_lo), 1.0 - F(np.nan_to_num(z_lo, neginf=-50)), 1.0)
    hi = np.where(np.isfinite(z_hi), 1.0 - F(np.nan_to_num(z_hi, posinf=50)), 0.0)
    return np.clip(lo - hi, 0.0, 1.0)


def clustered(series, groups):
    g = series.groupby(groups).mean()
    n = len(g)
    if n < 2 or g.std(ddof=1) == 0:
        return g.mean(), n, float("nan")
    return g.mean(), n, g.mean() / (g.std(ddof=1) / math.sqrt(n))


def main() -> None:
    argv = sys.argv[1:]
    n_months = int(argv[argv.index("--fit-months") + 1]) if "--fit-months" in argv else 6

    # ---- fit window: strictly before the recorded Kalshi window ---------
    if FIT_F.exists():
        px = pd.read_parquet(FIT_F)
        print(f"fit history from cache: {len(px):,} 1m bars")
    else:
        end = date(2026, 7, 1)          # the recorded window starts 07-21
        months = []
        y, m = end.year, end.month
        for _ in range(n_months):
            m -= 1
            if m == 0:
                y, m = y - 1, 12
            months.append(f"{y}-{m:02d}")
        months.reverse()
        print(f"fetching BTCUSDT 1m for {months}")
        px = fetch_1m(months)
        CACHE.mkdir(parents=True, exist_ok=True)
        px.to_parquet(FIT_F, index=False)
    span = pd.to_datetime(px.ts_s, unit="s", utc=True)
    print(f"fit history {len(px):,} bars, {span.min()} -> {span.max()}")
    if span.max() >= pd.Timestamp("2026-07-21", tz="UTC"):
        raise SystemExit("fit history overlaps the scoring window -- refusing "
                         "to fit and score on the same data")

    # ---- fit the standardised shape, pooled across the quoted horizons --
    zs = np.concatenate([standardised_returns(px, h) for h in HORIZONS])
    fit = fit_shapes(zs)
    print(f"\nstandardised terminal returns z, pooled over {HORIZONS} min:")
    print(f"  n {fit['n']:,}  sd {fit['c']:.3f}  skew {fit['skew']:+.3f}  "
          f"kurtosis {fit['kurt']:.2f}  fitted Student-t nu {fit['nu']:.2f}")
    if abs(fit["c"] - 1.0) > 0.02:
        word = "understates" if fit["c"] > 1 else "overstates"
        print(f"  -> the maker's sigma {word} the {min(HORIZONS)}-"
              f"{max(HORIZONS)} min move by {fit['c']:.2f}x in level "
              f"(sd of z would be 1.00 if sqrt(tau) scaling held)")
    print(f"  note: horizons overlap at 1m steps, so the effective sample is "
          f"~{len(px)//max(HORIZONS):,} independent windows, not {fit['n']:,}")
    if fit["kurt"] > 3.2:
        print(f"  -> and the shape is fat-tailed (Gaussian kurtosis is 3.00)")

    # ---- score every candidate on the recorded window -------------------
    book = pd.read_parquet(BOOK_F)
    spot = pd.read_parquet(SPOT_F)
    sig = sigma_path(spot)
    df = book[(book.tau_s >= 600) & (book.tau_s <= 2700)].copy()
    df["ts_s"] = df.ts_ms // 1000
    df = df.merge(sig, on="ts_s", how="inner").dropna(subset=["yes_bid", "yes_ask"])
    df = df[df.yes_ask > df.yes_bid]
    df["mid"] = (df.yes_bid + df.yes_ask) / 2
    df["y"] = (df.result == "yes").astype(float)
    df["event"] = df.ticker.str.split("-").str[1]
    df["kind"] = np.where(
        df.ticker.str.rsplit("-", n=1).str[-1].str.startswith("B"),
        "bracket", "threshold")
    fl = df["floor"].astype(float).fillna(-np.inf).values
    cp = df["cap"].astype(float).fillna(np.inf).values

    for name, F in make_cdfs(fit).items():
        df[name] = fv_from_cdf(F, df.spot.values, fl, cp,
                               df.sigma.values, df.tau_s.values)
    df["blend"] = 0.5 * (df.empirical + df.mid)

    # ORACLE: the same empirical density, but handed the realised volatility
    # of the forward window it is forecasting. Not tradeable -- it reads the
    # future -- but it is the ceiling on what any amount of volatility
    # forecasting could ever buy. If the oracle does not beat the book, no
    # sigma model will.
    g = sig.set_index("ts_s")
    r2 = (np.log(g.spot / g.spot.shift(1)) ** 2).fillna(0.0)
    cum = r2.cumsum()
    idx = cum.index.values
    start = np.searchsorted(idx, df.ts_s.values)
    stop = np.searchsorted(idx, df.ts_s.values + df.tau_s.values)
    ok = (stop < len(idx)) & (start < stop)
    cv = cum.values
    fwd_var = np.where(ok, (cv[np.clip(stop, 0, len(cv) - 1)]
                            - cv[np.clip(start, 0, len(cv) - 1)]), np.nan)
    df["sigma_oracle"] = np.maximum(
        np.sqrt(np.clip(fwd_var, 0, None) / df.tau_s.values), SIGMA_FLOOR)
    F_emp = make_cdfs(fit)["empirical"]
    df["oracle"] = np.where(
        ok, fv_from_cdf(F_emp, df.spot.values, fl, cp,
                        df.sigma_oracle.values, df.tau_s.values), np.nan)
    print(f"  oracle sigma available on {ok.mean():.1%} of rows; "
          f"median oracle/EWMA sigma ratio "
          f"{np.nanmedian(df.sigma_oracle / df.sigma):.2f}")
    df = df[ok].copy()

    models = ["mid", "gauss", "gauss_c", "student", "empirical", "blend",
              "oracle"]

    print(f"\nscored rows {len(df):,}  events {df.event.nunique()}  "
          f"yes rate {df.y.mean():.3f}")
    print("\n=== Brier vs the book, paired by settlement event ===")
    print(f"{'subset':<22}{'events':>7}" + "".join(f"{m:>10}" for m in models))
    for label, sel in (("all", df),
                       ("bracket", df[df.kind == "bracket"]),
                       ("threshold", df[df.kind == "threshold"]),
                       ("near-money .2-.8", df[(df["mid"] > 0.2) & (df["mid"] < 0.8)])):
        row = "".join(f"{((sel[m] - sel.y) ** 2).mean():10.4f}" for m in models)
        print(f"{label:<22}{sel.event.nunique():7d}{row}")

    print("\n=== each model minus the book (negative = BEATS the book) ===")
    print(f"{'model':<12}{'dBrier':>10}{'t':>8}   {'lean/ct':>9}{'t':>8}")
    for m in models[1:]:
        d, n, t = clustered(((df[m] - df.y) ** 2) - ((df["mid"] - df.y) ** 2),
                            df.event)
        lean = (df.y - df["mid"]) * np.sign(df[m] - df["mid"])
        ld, _, lt = clustered(lean, df.event)
        print(f"{m:<12}{d:+10.4f}{t:+8.2f}   {ld:+9.4f}{lt:+8.2f}")
    print("  dBrier < 0 with t < -2 would be a real forecasting edge.")
    print("  lean/ct is what leaning on the disagreement pays per contract.")


if __name__ == "__main__":
    main()
