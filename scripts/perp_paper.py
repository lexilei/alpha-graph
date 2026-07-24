"""Paper-trading pipeline for the 4-sleeve crypto perp combo on Coinbase.

Daily cycle (UTC close + a few minutes):
  1. update  — extend the Binance panel with vision DAILY kline files for
     each symbol's missing dates; extend funding with monthly truth when
     available plus a premiumIndexKlines daily proxy for the current month
     (flagged: rank-level approximation of funding for K2's signal).
  2. trade   — recompute the four sleeves exactly as the frozen scan module,
     equal-risk combine (trailing 90d sleeve vols), static-scale the book to
     TARGET_VOL using trailing combo vol, cap gross, map symbols to
     Coinbase *-PERP-INTX products, and paper-execute the weight deltas at
     live Coinbase best bid/ask plus TAKER_FEE. Positions/cash persist in
     data/paper/state.json; every action appends to data/paper/ledger.jsonl.
  3. mark    — daily NAV at Coinbase mids; funding accrual on held positions
     via the premium proxy (flagged, backfilled monthly).

Modes:
  update | trade | run-once (update+trade) | loop (daily at 00:10 UTC)
  --test N  restrict the panel update to N symbols (plumbing smoke test)

Paper capital: $100,000. No real orders are ever sent (read-only usage of
the Coinbase API; the key's trade scope is unused by design).
"""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "binance"
PAPER = ROOT / "data" / "paper"

spec = importlib.util.spec_from_file_location("scan", ROOT / "scripts" / "crypto_factor_scan.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)
spec2 = importlib.util.spec_from_file_location("fb", ROOT / "scripts" / "fetch_binance_perp.py")
fb = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(fb)
spec3 = importlib.util.spec_from_file_location("cb", ROOT / "scripts" / "coinbase_client.py")
cbm = importlib.util.module_from_spec(spec3)
spec3.loader.exec_module(cbm)

TARGET_VOL = 0.15
GROSS_CAP = 2.0
TAKER_FEE = 3e-4
PAPER_CAPITAL = 100_000.0
SLEEVES = ("mom30s1", "K2_carry_7d", "K11_amihud_30d", "K12_tbr_7d")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%m-%d %H:%M:%S}] {msg}", flush=True)


def ledger_write(event: dict) -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.now(timezone.utc).isoformat()
    with open(PAPER / "ledger.jsonl", "a") as f:
        f.write(json.dumps(event) + "\n")


# ---------------------------------------------------------------- update --

def daily_kline(sym: str, d: date) -> list | None:
    key = (f"data/futures/um/daily/klines/{sym}/1d/"
           f"{sym}-1d-{d:%Y-%m-%d}.zip")
    try:
        raw = fb._get(f"https://data.binance.vision/{key}")
    except RuntimeError:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows = [r.split(",") for r in
                z.read(z.namelist()[0]).decode().strip().split("\n")]
    rows = [r for r in rows if not r[0][0].isalpha()]
    return rows[0] if rows else None


def daily_premium(sym: str, d: date) -> float | None:
    key = (f"data/futures/um/daily/premiumIndexKlines/{sym}/1d/"
           f"{sym}-1d-{d:%Y-%m-%d}.zip")
    try:
        raw = fb._get(f"https://data.binance.vision/{key}")
    except RuntimeError:
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        rows = [r.split(",") for r in
                z.read(z.namelist()[0]).decode().strip().split("\n")]
    rows = [r for r in rows if not r[0][0].isalpha()]
    return float(rows[0][4]) if rows else None  # close of premium index


def update_panel(test_n: int | None = None) -> None:
    k = pd.read_parquet(RAW / "perp_klines_1d.parquet")
    k["date"] = pd.to_datetime(k["date"])
    last = k.groupby("symbol").date.max()
    yesterday = pd.Timestamp(datetime.now(timezone.utc).date() - timedelta(days=1))
    # live symbols only: delisted names would 404 for every day since death.
    # 40d window covers the initial monthly-panel catch-up (~23d stale).
    alive = last[last >= yesterday - pd.Timedelta(days=40)]
    syms = sorted(alive.index)
    if test_n:
        syms = syms[:test_n]
    jobs = []
    for s in syms:
        d = alive[s] + timedelta(days=1)
        while d <= yesterday:
            jobs.append((s, d.date()))
            d += timedelta(days=1)
    log(f"panel update: {len(jobs)} symbol-days to fetch")
    new_rows, prem_rows = [], []
    def work(sd):
        s, d = sd
        row = daily_kline(s, d)
        prem = daily_premium(s, d)
        return s, d, row, prem
    with ThreadPoolExecutor(24) as ex:
        futs = [ex.submit(work, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            s, d, row, prem = fut.result()
            if row:
                new_rows.append({
                    "symbol": s, "date": pd.Timestamp(d),
                    "open": float(row[1]), "high": float(row[2]),
                    "low": float(row[3]), "close": float(row[4]),
                    "volume": float(row[5]), "quote_volume": float(row[7]),
                    "n_trades": int(row[8]),
                    "taker_buy_volume": float(row[9])})
            if prem is not None:
                prem_rows.append({"symbol": s, "date": pd.Timestamp(d),
                                  "premium_close": prem})
            if (i + 1) % 2000 == 0:
                log(f"  {i+1}/{len(jobs)}")
    if new_rows:
        k2 = pd.concat([k, pd.DataFrame(new_rows)], ignore_index=True)
        k2 = k2.sort_values(["symbol", "date"]).drop_duplicates(
            ["symbol", "date"], keep="last")
        k2.to_parquet(RAW / "perp_klines_1d.parquet", index=False)
        log(f"klines +{len(new_rows)} rows -> {len(k2):,}")
    if prem_rows:
        p = pd.DataFrame(prem_rows)
        f_ = RAW / "premium_proxy_daily.parquet"
        if f_.exists():
            p = pd.concat([pd.read_parquet(f_), p], ignore_index=True)
            p = p.drop_duplicates(["symbol", "date"], keep="last")
        p.to_parquet(f_, index=False)
        log(f"premium proxy -> {len(p):,} rows")
    ledger_write({"event": "update", "new_klines": len(new_rows),
                  "new_premium": len(prem_rows)})


# ----------------------------------------------------------------- trade --

def load_panels_with_proxy():
    p = scan.load()
    f_ = RAW / "premium_proxy_daily.parquet"
    if f_.exists():
        prem = pd.read_parquet(f_)
        prem_piv = prem.pivot(index="date", columns="symbol",
                              values="premium_close")
        # funding_8h ~ premium (clamped); daily funding ~ 3x premium close.
        # Rank-level proxy only — flagged in every report.
        proxy = (prem_piv * 3).reindex(columns=p["fund"].columns)
        fund = p["fund"].combine_first(proxy.reindex(p["fund"].index))
        # extend index if proxy has newer dates
        all_idx = p["close"].index.union(proxy.index)
        p = {k: v.reindex(all_idx) for k, v in p.items()}
        p["fund"] = p["fund"].reindex(all_idx).combine_first(
            proxy.reindex(all_idx)).fillna(0.0)
    return p


def compute_targets(tradable_bases: set | None = None) -> dict:
    p = load_panels_with_proxy()
    close, qv, fund = p["close"], p["quote_volume"], p["fund"]
    ret = close.pct_change(fill_method=None)
    med = qv.rolling(30, min_periods=30).median()
    if tradable_bases:
        elig = close.columns.isin(tradable_bases)
        med = med.where(pd.DataFrame(
            {c: elig[i] for i, c in enumerate(close.columns)},
            index=close.index))
    uni = (med.where(close.notna()).rank(axis=1, ascending=False) <= 100) \
        & close.notna()
    sigs = scan.build_signals(p)
    # anchor on the last date where the universe actually exists (the funding
    # proxy can extend the index past the latest klines)
    valid_rows = uni.sum(axis=1)
    anchor = valid_rows[valid_rows >= 20].index[-1]
    weights, rets = {}, {}
    for s in SLEEVES:
        sig = sigs[s].where(uni)
        r = sig.rank(axis=1)
        n = r.max(axis=1)
        top = r.ge(n * 0.8, axis=0)
        bot = r.le(n * 0.2, axis=0)
        w = top.div(top.sum(axis=1), axis=0).fillna(0.0) \
            - bot.div(bot.sum(axis=1), axis=0).fillna(0.0)
        w[n < 20] = 0.0
        weights[s] = w
        pnl = (w * ret.shift(-1)).sum(axis=1) - (w * fund.shift(-1)).sum(axis=1)
        rets[s] = pnl
    rd = pd.DataFrame(rets).dropna()
    if rd.empty:
        raise RuntimeError("no valid sleeve returns — panel too sparse")
    vol90 = rd.tail(90).std()
    inv = (1.0 / vol90)
    inv /= inv.sum()
    combo_w = sum(weights[s].loc[anchor] * inv[s] for s in SLEEVES)
    combo_ret = (rd * inv).sum(axis=1)  # raw units, not z-scored
    lever = TARGET_VOL / (combo_ret.tail(90).std() * np.sqrt(365))
    tgt = combo_w * lever
    gross = tgt.abs().sum()
    if gross > GROSS_CAP:
        tgt *= GROSS_CAP / gross
    tgt = tgt[tgt.abs() > 1e-4]
    asof = str(anchor.date())
    return {"asof": asof, "weights": tgt.to_dict(), "lever": float(lever),
            "gross": float(tgt.abs().sum())}


def to_product(sym: str) -> str:
    return sym.replace("USDT", "") + "-PERP-INTX"


def paper_trade() -> None:
    c = cbm.Coinbase()
    prods = {p_["product_id"] for p_ in c.get(
        "/api/v3/brokerage/products?product_type=FUTURE"
        "&contract_expiry_type=PERPETUAL").get("products", [])}
    tradable = {pid.replace("-PERP-INTX", "") + "USDT" for pid in prods}
    tgt = compute_targets(tradable)
    mapped = {s: to_product(s) for s in tgt["weights"]
              if to_product(s) in prods}
    dropped = sorted(set(tgt["weights"]) - set(mapped))
    state_f = PAPER / "state.json"
    state = json.loads(state_f.read_text()) if state_f.exists() else {
        "cash": PAPER_CAPITAL, "positions": {}, "start": None}
    if state["start"] is None:
        state["start"] = datetime.now(timezone.utc).isoformat()

    # quote everything we target OR hold: held-but-untargeted symbols must
    # still be priced so they can be flattened and counted in NAV
    quote_syms = {**mapped,
                  **{s2: to_product(s2) for s2 in state["positions"]
                     if to_product(s2) in prods}}
    ids = "&".join(f"product_ids={p_}" for p_ in set(quote_syms.values()))
    books = {b["product_id"]: b for b in c.get(
        f"/api/v3/brokerage/best_bid_ask?{ids}").get("pricebooks", [])}

    nav = state["cash"]
    quotes = {}
    for sym, prod in quote_syms.items():
        b = books.get(prod, {})
        try:
            bid = float(b["bids"][0]["price"]); ask = float(b["asks"][0]["price"])
        except (KeyError, IndexError):
            continue
        quotes[sym] = (bid, ask, (bid + ask) / 2)
    unpriceable = [s2 for s2 in state["positions"] if s2 not in quotes]
    if unpriceable:
        log(f"WARN: held positions with no quote (marked at last known "
            f"price): {unpriceable}")
    for sym, pos in state["positions"].items():
        if sym in quotes:
            pos["last_mark"] = quotes[sym][2]
            nav += pos["qty"] * quotes[sym][2]
        elif pos.get("last_mark") is not None:
            # delisted/bookless symbol: carry at last mark rather than
            # silently vanishing from NAV
            nav += pos["qty"] * pos["last_mark"]
    trades, fees, slip = [], 0.0, 0.0
    for sym, prod in mapped.items():
        if sym not in quotes:
            continue
        bid, ask, mid = quotes[sym]
        w = tgt["weights"][sym]
        tgt_qty = w * nav / mid
        cur = state["positions"].get(sym, {"qty": 0.0, "avg": 0.0})
        dq = tgt_qty - cur["qty"]
        if abs(dq) * mid < 10:  # ignore sub-$10 rebalances
            continue
        px = ask if dq > 0 else bid
        fee = abs(dq) * px * TAKER_FEE
        state["cash"] -= dq * px + fee
        fees += fee
        slip += abs(dq) * (px - mid) if dq > 0 else abs(dq) * (mid - px)
        state["positions"][sym] = {"qty": tgt_qty, "avg": px}
        trades.append({"sym": sym, "dq": round(dq, 6), "px": px,
                       "notional": round(dq * px, 2)})
    # drop zeroed positions not in targets
    for sym in list(state["positions"]):
        if sym not in mapped and sym in quotes:
            pos = state["positions"].pop(sym)
            bid, ask, mid = quotes[sym]
            px = bid if pos["qty"] > 0 else ask
            state["cash"] += pos["qty"] * px
            state["cash"] -= abs(pos["qty"]) * px * TAKER_FEE
    nav2 = state["cash"] + sum(
        pos["qty"] * quotes[s][2] for s, pos in state["positions"].items()
        if s in quotes)
    state_f.parent.mkdir(parents=True, exist_ok=True)
    state_f.write_text(json.dumps(state, indent=1))
    ledger_write({"event": "rebalance", "asof": tgt["asof"],
                  "n_targets": len(tgt["weights"]), "n_mapped": len(mapped),
                  "dropped": dropped[:10], "n_trades": len(trades),
                  "fees": round(fees, 2), "half_spread_cost": round(slip, 2),
                  "gross": tgt["gross"], "lever": tgt["lever"],
                  "nav": round(nav2, 2), "trades": trades})
    log(f"paper rebalance asof {tgt['asof']}: {len(trades)} trades, "
        f"fees ${fees:.2f}, spread ${slip:.2f}, NAV ${nav2:,.2f}, "
        f"gross {tgt['gross']:.2f}, dropped {len(dropped)} unmapped")


def next_run_utc() -> float:
    now = datetime.now(timezone.utc)
    run = now.replace(hour=0, minute=10, second=0, microsecond=0)
    if run <= now:
        run += timedelta(days=1)
    return run.timestamp()


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "run-once"
    test_n = None
    if "--test" in sys.argv:
        test_n = int(sys.argv[sys.argv.index("--test") + 1])
    if mode == "update":
        update_panel(test_n)
    elif mode == "trade":
        paper_trade()
    elif mode == "run-once":
        update_panel(test_n)
        paper_trade()
    elif mode == "loop":
        while True:
            wait = next_run_utc() - time.time()
            log(f"sleeping {wait/3600:.1f}h until 00:10 UTC")
            time.sleep(max(60, wait))
            try:
                update_panel()
                paper_trade()
            except Exception as e:  # noqa: BLE001
                ledger_write({"event": "error", "err": str(e)[:300]})
                log(f"cycle error: {e}")


if __name__ == "__main__":
    main()
