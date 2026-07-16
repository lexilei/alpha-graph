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

Output: reports/singles_iv_qa.csv (per name-year day counts + coverage,
plus per-name first/last option-quote date and expiration-partition count)
and a stdout summary with the gate inputs: names with >=80% LIFE-ADJUSTED
coverage — denominator restricted to trading days within [first, last]
option-quote date present for that name, which is immune to mid-window
IPOs and to spot.parquet's fixed back-filled calendar (audit F1,
reports/qa_gate_audit_2026-07-15.md). Full-calendar coverage is also
reported; the (full, life, first/last, n_partitions) tuple disambiguates
unfinished downloads from bad data (audit F2).

Known accepted floors (audit F3/F4): the ±10% band can exceed the strike
grid for sub-$5 price histories (understatement only, quantified at 1 day
on AMD), and "one near-money pair marks the day usable" measures liveness
(I2's minimal need), not chain richness.
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
    qdates_present: set[pd.Timestamp] = set()
    for p in parts:
        df = pd.read_parquet(
            p,
            columns=["quote_date", "expiration", "strike", "right", "bid", "ask", "created"],
        )
        if df.empty:
            continue
        df["quote_date"] = pd.to_datetime(df["quote_date"])
        qdates_present.update(df["quote_date"].unique())
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

    if not qdates_present:
        return None
    first_q, last_q = min(qdates_present), max(qdates_present)
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
                "first_quote": str(first_q.date()),
                "last_quote": str(last_q.date()),
                "n_partitions": len(parts),
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
    life_cov: dict[str, float] = {}
    for t in tickers:
        r = qa_ticker(RAW / t)
        if r is None:
            print(f"{t}: NO DATA", file=sys.stderr)
            continue
        frames.append(r)
        tot = r["paired_quote_days"].sum()
        alldays = r["trading_days"].sum()
        # life-adjusted: denominator restricted to the name's own
        # [first, last] option-quote window (audit F1)
        spot = pd.read_parquet(RAW / t / "spot.parquet")[["quote_date"]]
        cal = pd.to_datetime(spot["quote_date"])
        f0, l0 = pd.Timestamp(r["first_quote"].iloc[0]), pd.Timestamp(r["last_quote"].iloc[0])
        life_days = int(((cal >= f0) & (cal <= l0)).sum())
        life = tot / life_days if life_days else 0.0
        life_cov[t] = life
        print(
            f"{t}: {tot}/{alldays} full-calendar ({tot/alldays:.2%}) | "
            f"life-adjusted {tot}/{life_days} ({life:.2%}) | "
            f"quotes {r['first_quote'].iloc[0]}..{r['last_quote'].iloc[0]}, "
            f"{r['n_partitions'].iloc[0]} partitions"
        )
    if not frames:
        print("no tickers with data", file=sys.stderr)
        return 1
    out = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}")
    n_pass = sum(1 for v in life_cov.values() if v >= 0.8)
    print(f"names >=80% life-adjusted coverage: {n_pass}/{len(life_cov)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
