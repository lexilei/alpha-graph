#!/usr/bin/env python3
"""QA gate for the options-xs family (IDEAS.md I1/I2/I3): per-name usable-quote
coverage of the single-name options store, BEFORE any IV computation.

Data engineering, not a look (pinned in the 2026-07-15 rule-6 amendment):
member 1's registration requires this report to pass first.

Usable quote (per name-day): after keep-last dedup on (quote_date, expiration,
strike, right), a CALL and PUT at the SAME strike/expiration with bid > 0 on
both legs, 20-60 calendar DTE, strike within ±10% of the PARITY-IMPLIED spot.
That is the minimal raw material for the I2 call-put IV spread construction;
I1 (OTM put skew) and I3 (ATM IV) are less demanding on pairing but need the
same near-money quote liveness.

Spot is inferred per (quote_date, expiration) from put-call parity — the
strike minimizing |call_mid - put_mid| gives spot ~= K + C - P. The store's
spot.parquet `close` is SPLIT-ADJUSTED (back-adjusted) while option strikes
are as-traded, so banding against it zeroes every pre-split year (caught
2026-07-15: AAPL 0% pre-2020-08 4:1 split, AMZN 0% pre-2022-06 20:1 split);
spot.parquet is used only for the trading-day calendar.

Output: reports/singles_iv_qa.csv (per name-year day counts + coverage) and
a stdout summary with the gate verdict inputs (names with >=80% day coverage,
monthly cross-section curve).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

RAW = Path("/Users/leizhihan/Lex/data/vol_smile/raw_singles")
OUT = Path(__file__).resolve().parents[1] / "reports" / "singles_iv_qa.csv"

DTE_MIN, DTE_MAX = 20, 60
MONEY_BAND = 0.10


def qa_ticker(tdir: Path) -> pd.DataFrame | None:
    spot_f = tdir / "spot.parquet"
    if not spot_f.exists():
        return None
    spot = pd.read_parquet(spot_f)[["quote_date", "close"]]
    spot["quote_date"] = pd.to_datetime(spot["quote_date"])
    spot = spot.set_index("quote_date")["close"]

    parts = sorted(tdir.glob("expiration=*/*.parquet"))
    if not parts:
        return None
    days_ok: dict[pd.Timestamp, bool] = {}
    for p in parts:
        df = pd.read_parquet(
            p,
            columns=["quote_date", "expiration", "strike", "right", "bid", "ask", "created"],
        )
        if df.empty:
            continue
        df["quote_date"] = pd.to_datetime(df["quote_date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        dte = (df["expiration"] - df["quote_date"]).dt.days
        df = df[(dte >= DTE_MIN) & (dte <= DTE_MAX) & (df["bid"] > 0)]
        if df.empty:
            continue
        df = (
            df.sort_values("created")
            .drop_duplicates(["quote_date", "expiration", "strike", "right"], keep="last")
        )
        df["mid"] = (df["bid"] + df["ask"]) / 2
        wide = df.pivot_table(
            index=["quote_date", "expiration", "strike"], columns="right",
            values="mid", aggfunc="first",
        )
        if "CALL" not in wide.columns or "PUT" not in wide.columns:
            continue
        wide = wide.dropna(subset=["CALL", "PUT"]).reset_index()
        if wide.empty:
            continue
        # parity-implied spot per (date, expiration): ATM = argmin |C - P|
        cp = (wide["CALL"] - wide["PUT"]).abs()
        atm = wide.loc[cp.groupby([wide["quote_date"], wide["expiration"]]).idxmin()]
        atm_spot = (
            (atm["strike"] + atm["CALL"] - atm["PUT"])
            .groupby([atm["quote_date"], atm["expiration"]])
            .first()
            .rename("spot_impl")
        )
        wide = wide.join(atm_spot, on=["quote_date", "expiration"])
        near = wide[(wide["strike"] / wide["spot_impl"] - 1).abs() <= MONEY_BAND]
        for d in near["quote_date"].unique():
            days_ok[d] = True

    trading_days = spot.index
    rows = []
    for year, days in pd.Series(trading_days, index=trading_days).groupby(
        trading_days.year
    ):
        ok = sum(1 for d in days if days_ok.get(d, False))
        rows.append(
            {
                "ticker": tdir.name,
                "year": year,
                "trading_days": len(days),
                "paired_quote_days": ok,
                "coverage": round(ok / len(days), 4) if len(days) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="default: every ticker directory present in the store")
    args = ap.parse_args()

    tickers = args.tickers or sorted(
        d.name for d in RAW.iterdir() if d.is_dir() and not d.name.startswith("_")
    )
    frames = []
    for t in tickers:
        r = qa_ticker(RAW / t)
        if r is None:
            print(f"{t}: NO DATA", file=sys.stderr)
            continue
        frames.append(r)
        tot = r["paired_quote_days"].sum()
        alldays = r["trading_days"].sum()
        print(f"{t}: {tot}/{alldays} paired-quote days ({tot/alldays:.1%})")
    if not frames:
        print("no tickers with data", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    per_name = out.groupby("ticker").apply(
        lambda g: g["paired_quote_days"].sum() / g["trading_days"].sum()
    )
    print(f"names >=80% coverage: {(per_name >= 0.8).sum()}/{len(per_name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
