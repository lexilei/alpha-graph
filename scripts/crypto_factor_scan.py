"""Evaluate pre-registered crypto factor batch K1-K14.

Definitions, signs, decision stats and bars are frozen in
reports/crypto_factor_prereg.md (committed before this run). Protocol is the
feasibility-scan protocol: top-100 liquidity universe, quintile L/S, EW,
1.0 gross/side, close-t signal earns t -> t+1, funding included, fee ladder.

Output: reports/crypto_factor_scan_k1.csv + printed table.

Usage: .venv/bin/python scripts/crypto_factor_scan.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "binance"

UNIVERSE_N = 100
QUANTILE = 5
MIN_HISTORY_D = 30
STABLE_PAIRS = {"USDCUSDT", "TUSDUSDT", "BUSDUSDT", "USDPUSDT", "FDUSDUSDT",
                "AEURUSDT", "USD1USDT"}
TAKER = 5e-4
MAKER = 2e-4
ANN = 365


# pre-registered judgment window: numbers of record are computed on the
# panel truncated here. Run with --full for live monitoring; that mode
# writes *_full.csv and never touches the frozen artifact.
PANEL_END = "2026-06-30"
FULL_PANEL = "--full" in sys.argv


def load() -> dict[str, pd.DataFrame]:
    k = pd.read_parquet(RAW / "perp_klines_1d.parquet")
    k = k[~k.symbol.isin(STABLE_PAIRS)]
    k["date"] = pd.to_datetime(k["date"])
    k = k.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"])
    cols = [c for c in ("close", "quote_volume", "volume", "taker_buy_volume",
                        "n_trades")
            if c in k.columns]
    p = {c: k.pivot(index="date", columns="symbol", values=c) for c in cols}
    # zombie bars: Binance keeps emitting flat 0-volume daily bars for
    # halted contracts (7.9% of raw rows, some 398 days long); mask them so
    # frozen prices can never enter returns or the universe
    if "volume" in p and "n_trades" in p:
        zombie = (p["volume"] == 0) & (p["n_trades"] == 0)
        for c in ("close", "quote_volume", "volume", "taker_buy_volume"):
            if c in p:
                p[c] = p[c].mask(zombie)
    f = pd.read_parquet(RAW / "perp_funding.parquet")
    f["date"] = f["funding_time"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    fund = f.groupby(["date", "symbol"]).funding_rate.sum().unstack()
    # events/day per symbol from recent truth: 4h-funding symbols are the
    # 2026 majority (72% of symbol-days) and a flat x3 halves their carry
    n_ev = (f.groupby(["date", "symbol"]).size().unstack()
            .tail(14).median().clip(1, 6))
    fund = fund.reindex(index=p["close"].index, columns=p["close"].columns)
    # months past the last monthly-truth file: premiumIndexKlines proxy.
    # Per interval, funding ~= premium + clamp(interest - premium, +-0.05%)
    # with interest = 0.03%/day split across intervals; then x events/day.
    # Flagged: rank-level proxy - refresh truth funding before any verdict.
    proxy_f = RAW / "premium_proxy_daily.parquet"
    if proxy_f.exists():
        pr = pd.read_parquet(proxy_f)
        pr["date"] = pd.to_datetime(pr["date"])
        prem = pr.pivot(index="date", columns="symbol", values="premium_close")
        n = n_ev.reindex(prem.columns).fillna(3.0)
        adj = (0.0003 / n - prem).clip(lower=-0.0005, upper=0.0005)
        proxy = (prem + adj).mul(n, axis=1).reindex(
            index=p["close"].index, columns=p["close"].columns)
        fund = fund.combine_first(proxy)
    p["fund"] = fund.fillna(0.0)
    if not FULL_PANEL:
        p = {c: df.loc[:PANEL_END] for c, df in p.items()}
    return p


def build_signals(p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    btc = ret["BTCUSDT"]
    cov = ret.rolling(60).cov(btc)
    beta = cov.div(btc.rolling(60).var(), axis=0)
    sigs = {
        "mom30s1": close.shift(1) / close.shift(31) - 1,
        "K1_carry_1d": -fund,
        "K2_carry_7d": -fund.rolling(7).mean(),
        "K3_carry_30d": -fund.rolling(30).mean(),
        "K4_rev_1d": -ret,
        "K5_rev_3d": -(close / close.shift(3) - 1),
        "K6_lowvol_30d": -ret.rolling(30).std(),
        "K7_bab_60d": -beta,
        "K8_max_30d": -ret.rolling(30).max(),
        "K9_skew_30d": -ret.rolling(30).skew(),
        "K10_size_qv": -np.log(qv.rolling(30).median()),
        "K11_amihud_30d": (ret.abs() / qv).replace([np.inf, -np.inf], np.nan)
                          .rolling(30).mean(),
        "K13_volz_7d": -(qv.rolling(7).mean() / qv.rolling(90).mean()),
        "K14_high_90d": close / close.rolling(90).max(),
    }
    if "taker_buy_volume" in p:  # column absent until klines re-download lands
        tbr = (p["taker_buy_volume"] / p["volume"]).replace(
            [np.inf, -np.inf], np.nan)
        sigs["K12_tbr_7d"] = tbr.rolling(7).mean()
    return sigs


def ls_returns(sig: pd.DataFrame, universe: pd.DataFrame, ret: pd.DataFrame,
               fund: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Daily gross pnl, traded notional, live mask under the frozen protocol."""
    s = sig.where(universe)
    r = s.rank(axis=1)
    n = r.max(axis=1)
    top = r.ge(n * (1 - 1 / QUANTILE), axis=0)
    bot = r.le(n / QUANTILE, axis=0)
    w = top.div(top.sum(axis=1), axis=0).fillna(0.0) \
        - bot.div(bot.sum(axis=1), axis=0).fillna(0.0)
    w[n < QUANTILE * 4] = 0.0
    pnl = (w * ret.shift(-1)).sum(axis=1) - (w * fund.shift(-1)).sum(axis=1)
    traded = (w - (w.shift(1) * (1 + ret)).fillna(0.0)).abs().sum(axis=1)
    live = w.abs().sum(axis=1) > 0
    return pnl, traded, live


def sharpe(x: pd.Series) -> float:
    return float(x.mean() / x.std() * np.sqrt(ANN)) if x.std() > 0 else np.nan


def main() -> None:
    print(f"PANEL: {'FULL (live monitoring, proxy-funded tail)' if FULL_PANEL else f'frozen <= {PANEL_END} (judgment window)'}")
    p = load()
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    med_qv = qv.rolling(MIN_HISTORY_D, min_periods=MIN_HISTORY_D).median()
    universe = (med_qv.where(close.notna())
                .rank(axis=1, ascending=False) <= UNIVERSE_N) & close.notna()

    sigs = build_signals(p)
    base_pnl, base_traded, base_live = ls_returns(
        sigs["mom30s1"], universe, ret, fund)
    base_net = (base_pnl - base_traded * TAKER)[base_live]
    base_sr = sharpe(base_net)
    yrs = len(base_net) / ANN

    rows = []
    for name, sig in sigs.items():
        pnl, traded, live = ls_returns(sig, universe, ret, fund)
        net = (pnl - traded * TAKER)[live]
        net23 = net[net.index >= "2023-01-01"]
        corr = float(pnl[live].corr(base_pnl[live]))
        # equal-risk 50/50 combo with baseline (full-sample vols, diagnostic)
        a = net / net.std()
        b = base_net / base_net.std()
        combo = (a.add(b, fill_value=np.nan)).dropna()
        # compare on the joint subsample: full-sample base_sr vs a
        # short-history factor's combo window mixes Sharpe regimes
        base_sub_sr = sharpe(base_net.reindex(combo.index))
        sr = sharpe(net)
        rows.append({
            "id": name,
            "turnover_1s": float(traded[live].mean() / 2),
            "gross_sr": sharpe(pnl[live]),
            "maker_sr": sharpe((pnl - traded * MAKER)[live]),
            "taker_sr": sr,
            "t": sr * np.sqrt(len(net) / ANN) if np.isfinite(sr) else np.nan,
            "taker_sr_2023on": sharpe(net23),
            "corr_mom": corr,
            "combo_dSR": sharpe(combo) - base_sub_sr,
        })
    res = pd.DataFrame(rows)

    def verdict(r: pd.Series) -> str:
        if r["id"] == "mom30s1":
            return "baseline"
        standalone = (r.taker_sr >= 0.5 and r.t >= 2.0
                      and r.taker_sr_2023on > 0)
        divers = (abs(r.corr_mom) <= 0.4 and r.combo_dSR >= 0.10
                  and r.taker_sr >= 0.3)
        if r.taker_sr < 0:
            return "REJECT wrong-sign"
        if standalone and divers:
            return "PASS standalone+div"
        if standalone:
            return "PASS standalone"
        if divers:
            return "PASS diversifier"
        return "reject"

    res["verdict"] = res.apply(verdict, axis=1)
    tag = "_full" if FULL_PANEL else ""
    out = ROOT / "reports" / f"crypto_factor_scan_k1{tag}.csv"
    res.to_csv(out, index=False, float_format="%.4f")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res.round(3).to_string(index=False))
    print(f"baseline mom30s1 net-taker SR = {base_sr:.3f} over {yrs:.1f}y")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
