"""FV-maker v0 on Kalshi DEMO — BTC hourly ladders, wide safe quotes.

The equilibrium test the replay sim cannot provide: real resting orders in a
real (demo) book, measuring fill rate, adverse selection, and whether other
makers undercut. Fake money; risk caps anyway (habits transfer to live).

Policy each cycle (10s):
  spot   Coinbase BTC-USD last trade (BRTI-constituent proxy; Kalshi settles
         on the 60s BRTI TWAP)
  sigma  EWMA of 1s log returns from the in-process spot buffer
  for each KXBTC hourly T-strike within +/-0.8% of spot, 10-50min to close:
    FV = Phi( ln(S/K) / (sigma*sqrt(tau)) )
    delta = PHI0/(sigma*sqrt(tau)) * 3*sigma*sqrt(CYCLE_S)   [q99~3sigma]
    quote post_only bid/ask at FV -/+ delta (sticky: requote only if the
    target moved > STICKY from the resting price), 10 contracts/side
  inventory cap 50/market -> quote reduce-only side; skip last 10min (TWAP
  gamma); cancel everything on exit or on daily loss > $25.

Logs: data/raw/polymarket/demomaker_*.jsonl.gz (sink format).

Usage: .venv/bin/python scripts/kalshi_demo_maker.py [--cycles N]
"""

from __future__ import annotations

import importlib.util
import json
import calendar
import math
import sys
from datetime import datetime, timedelta, timezone
import time
import urllib.request
import uuid
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "kc", ROOT / "scripts" / "kalshi_client.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)
import polymarket_recorder as rec  # noqa: E402  (Sink)

PHI0 = 0.3989
CYCLE_S = 10.0
STICKY = 0.02
SIZE = 5
INV_CAP = 50
BAND = 0.05   # strike band vs spot; nearest N picked below

# environment is FILE-driven (data/private/maker_env.txt: "demo"|"prod"),
# never argv: the watchdog relaunches with bare argv, and an argv switch
# would silently revert prod->demo on the first restart (panel review B1).
# All durable state (baseline/stop-flag/sink) is namespaced per env so a
# demo baseline ($962) can never disarm the prod ($49) loss stop (B2).
_ENV_F = rec.OUT.parent.parent / "private" / "maker_env.txt"
try:
    ENV = _ENV_F.read_text().strip().lower()
except (OSError, ValueError):  # ValueError: non-UTF-8 file must fail safe
    ENV = "demo"               # to demo, not crash-loop at import (r2 LOW)
if ENV not in ("demo", "prod"):
    ENV = "demo"
PROD = ENV == "prod"
DAILY_LOSS_STOP = 15.0 if PROD else 25.0  # prod: ~30% of the $49.53 bankroll
MAX_STRIKES = 5 if PROD else 12  # $44 budget can't two-side 12 strikes;
# 5 keeps every quoted strike deterministically two-sided (panel 4)
NET_CAP = 20 if PROD else 3 * INV_CAP  # prod: with the HARD projection in
# the placement loop, worst reachable net ~= NET_CAP + in-flight SIZE, so
# (20+5) x $0.60 = $15 = the stop; the old soft cap overshot ~2x (r2 F1)
SINK_NAME = "makerprod" if PROD else "demomaker"


def spot_now() -> tuple[float, float]:
    """(last price, print age seconds). The ticker's own timestamp was
    fetched-and-ignored before — a stuck feed passed the jump guard from
    cycle 2 on and anchored fv to a stale price (panel 6 drill)."""
    req = urllib.request.Request(
        "https://api.exchange.coinbase.com/products/BTC-USD/ticker",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        tk = json.load(r)
    age = 0.0
    t = tk.get("time")
    if t:
        try:
            ts = datetime.strptime(t[:19], "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc).timestamp()
            age = max(0.0, time.time() - ts)
        except ValueError:
            pass
    return float(tk["price"]), age


class Vol:
    """Two-speed EWMA variance. Audit: a single 300s-halflife estimator
    lagged a real spike 4.6x for minutes. The slow leg centers fv (stable);
    the fast leg drives the width so quotes widen INTO a spike."""

    def __init__(self, halflife_s: float = 300.0, fast_s: float = 30.0):
        self.hl, self.fast_hl = halflife_s, fast_s
        self.var = None
        self.fvar = None
        self.last = None
        self.t = None

    def update(self, px: float, t: float) -> None:
        if self.last is not None and t > self.t:
            dt = t - self.t
            r2 = math.log(px / self.last) ** 2 / dt  # per-second variance
            a = 1 - 0.5 ** (dt / self.hl)
            self.var = r2 if self.var is None else (1 - a) * self.var + a * r2
            af = 1 - 0.5 ** (dt / self.fast_hl)
            self.fvar = r2 if self.fvar is None else (1 - af) * self.fvar + af * r2
        self.last, self.t = px, t

    @property
    def sigma_1s(self) -> float:
        s = math.sqrt(self.var) if self.var else 3e-4
        return max(s, 3e-5)  # floor: quiet-period collapse quoted too tight

    @property
    def sigma_fast(self) -> float:
        s = math.sqrt(self.fvar) if self.fvar else 3e-4
        return max(s, self.sigma_1s)  # width driver: never below the slow leg


def main() -> None:
    max_cycles = None
    if "--cycles" in sys.argv:
        max_cycles = int(sys.argv[sys.argv.index("--cycles") + 1])
    # flock singleton (r2 ops F2): the maker was the only long-lived process
    # without one — a watchdog false-negative plus a manual start meant two
    # makers double-quoting one account and interleaving one gzip sink
    import fcntl
    lock_dir = rec.OUT.parent.parent / "logs" / "launchd"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_f = open(lock_dir / f"maker_{ENV}.lock", "a")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"another {ENV} maker holds the lock; exiting", flush=True)
        return
    k = kc.Kalshi(ENV)
    sink = rec.Sink(SINK_NAME)
    vol = Vol()
    start_bal = None  # set from full equity (cash+positions) on first cycle
    # minus 5min: a process STARTING inside the 00:00-00:05 pre-roll window
    # must treat "today" as yesterday, or it bypasses the held stop flag and
    # warms its baseline off settlement-distorted equity (r2 logic MED-3)
    stop_day = (datetime.now(timezone.utc) - timedelta(minutes=5)).date()
    stop_armed = False  # loss stop fires only on two consecutive breaches
    # settlement makes portfolio_value spike toward $1-marks or read 0 for
    # minutes at a time, so the stop AND the daily baseline both use a
    # 30-cycle (~5min) median of equity, never an instantaneous read
    eq_hist: list[float] = []
    seen_fills: dict[str, bool] = {}
    last_targets: dict = {}
    prev_S: float | None = None
    flag_dir = rec.OUT.parent.parent / "logs" / "launchd"
    flag_dir.mkdir(parents=True, exist_ok=True)
    base_f = flag_dir / f"maker_baseline_{ENV}.txt"  # "<day> <start_bal>"

    def _write_baseline(path, day, val):
        tmp = path.with_suffix(".tmp")  # atomic: a torn write would defeat
        tmp.write_text(f"{day} {val}")  # the durability this file exists for
        tmp.replace(path)

    # fill telemetry (nightly r2 data F1 + logic MED-1): the in-memory-only
    # dedup dropped every fill that landed during a restart gap, and on a
    # FRESH account the first real batch was mis-classified as history. A
    # durable per-env watermark (max recorded created_time) fixes both; a
    # restart may re-record the single boundary fill (dup > loss; offline
    # dedupe by trade_id).
    fillmark_f = flag_dir / f"maker_fillmark_{ENV}.txt"
    try:
        fill_wm = fillmark_f.read_text().strip()
    except OSError:
        fill_wm = None  # first-ever session: snapshot on the first poll

    def _write_fillmark():
        tmp = fillmark_f.with_suffix(".tmp")
        tmp.write_text(fill_wm or "")
        tmp.replace(fillmark_f)

    def record_fills(base_ctx: dict, tgt_map: dict | None = None) -> None:
        nonlocal fill_wm
        try:
            fl = k.get("/portfolio/fills?limit=100").get("fills", [])
        except Exception:  # noqa: BLE001
            return
        if fill_wm is None:
            # first successful poll EVER (empty or not): everything present
            # is pre-session history — snapshot the watermark, record none
            fill_wm = max((f_.get("created_time") or "" for f_ in fl),
                          default="")
            _write_fillmark()
            return
        new_max = fill_wm
        for f_ in fl:
            ct = f_.get("created_time") or ""
            if ct < fill_wm:
                continue  # strictly older than the watermark: history
            fid = str(f_.get("trade_id") or f_.get("fill_id")
                      or str(f_.get("order_id", "")) + ct)
            if fid in seen_fills:
                continue
            seen_fills[fid] = True
            ctx_t = (tgt_map or {}).get(f_.get("ticker")) or {}
            sink.write("fill", {"raw": f_, "ctx": {
                **base_ctx, "fv": ctx_t.get("fv"),
                "our_bid": ctx_t.get("bid"), "our_ask": ctx_t.get("ask")}})
            if ct > new_max:
                new_max = ct
        if new_max != fill_wm:
            fill_wm = new_max
            _write_fillmark()
        while len(seen_fills) > 5000:
            seen_fills.pop(next(iter(seen_fills)))

    print(f"maker start env={ENV} stop=${DAILY_LOSS_STOP} "
          f"strikes={MAX_STRIKES} net_cap={NET_CAP}", flush=True)

    n = 0
    while max_cycles is None or n < max_cycles:
        t0 = time.time()
        try:
            # equity and the loss-stop are evaluated FIRST: a spot-feed
            # failure must skip quoting, never blind the stop (panel 6)
            bal_resp = k.get("/portfolio/balance")
            bal = float(bal_resp["balance_dollars"])
            # portfolio_value is in CENTS (no _dollars twin in the payload)
            equity = bal + float(bal_resp.get("portfolio_value", 0) or 0) / 100.0
            eq_hist = (eq_hist + [equity])[-30:]
            # lower median: robust to settlement spikes AND conservative
            # (leans toward stopping) on even-length windows
            eq_med = sorted(eq_hist)[(len(eq_hist) - 1) // 2]
            now_utc = datetime.now(timezone.utc)
            today = now_utc.date()
            # roll the baseline only after 00:05: the roll instant coincides
            # with daily settlement, whose portfolio_value distortions can
            # outlast the 30-cycle median window (audit rank-3)
            if today != stop_day and now_utc.minute >= 5:
                stop_day = today
                if len(eq_hist) >= 10:
                    start_bal = eq_med
                    _write_baseline(base_f, today, start_bal)
                else:
                    start_bal = None  # process started seconds before the
                    # roll: warm up like a cold start, don't latch 1 sample
            if start_bal is None:
                # baseline must survive restarts, else every restart re-arms
                # a fresh $25 budget (daily loss becomes restarts x $25); and
                # a first-cycle instantaneous read can latch a settlement
                # spike/zero. Adopt today's persisted baseline, else warm up
                # ~100s and set it from the median.
                try:
                    day_s, bal_s = base_f.read_text().split()
                    if day_s == str(today):
                        start_bal = float(bal_s)
                except (OSError, ValueError):
                    pass
                if start_bal is None and len(eq_hist) >= 10:
                    start_bal = eq_med
                    _write_baseline(base_f, today, start_bal)
            if start_bal is None:
                # warming up: the stop is not armed yet, so DO NOT QUOTE —
                # round-4 review caught this window placing orders with the
                # loss stop disarmed (and crashing telemetry on round(None))
                sink.write("maker_warmup", {"balance": bal, "equity": equity,
                                            "n_hist": len(eq_hist)})
                sink.flush()
                n += 1
                time.sleep(max(0.0, CYCLE_S - (time.time() - t0)))
                continue
            # keyed by stop_day, not today: in the 00:00-00:05 pre-roll
            # window yesterday's halt must keep holding under yesterday's
            # flag, not re-fire under today's and halt the whole new day
            # (fix-review M3)
            stop_flag = flag_dir / f"maker_stop_{ENV}_{stop_day}.flag"
            if stop_flag.exists():
                # halted for the day; the flag survives watchdog restarts so
                # a restart cannot silently defeat the stop. Auto-resumes at
                # the next UTC roll (new flag name) with a fresh baseline.
                # Fills are still polled: the fills that CAUSED the stop are
                # the most diagnostic records of all (r2 data F2).
                record_fills({"halted": True}, last_targets)
                sink.flush()
                n += 1
                time.sleep(45)
                continue
            breached = (start_bal is not None
                        and eq_med < start_bal - DAILY_LOSS_STOP)
            if breached and not stop_armed:
                stop_armed = True
                sink.write("stop_armed", {"equity": equity, "eq_med": eq_med,
                                          "bal": bal})
                breached = False
            elif not breached:
                stop_armed = False
            if breached:
                resting = k.get("/portfolio/orders?status=resting"
                                ).get("orders", [])
                n_fail = 0
                for o in resting:
                    try:
                        k.delete(f"/portfolio/events/orders/{o['order_id']}")
                    except Exception:  # noqa: BLE001
                        n_fail += 1
                try:
                    stop_flag.write_text(
                        f"eq_med {eq_med:.2f} baseline {start_bal:.2f}\n")
                except OSError:
                    pass  # disk-full: halting still holds via per-cycle
                    # breach re-entry; only restart persistence is lost
                sink.write("maker_stop", {"balance": bal, "equity": equity,
                                          "eq_med": eq_med,
                                          "baseline": start_bal,
                                          "canceled": len(resting) - n_fail,
                                          "cancel_failed": n_fail})
                print(f"loss stop hit: canceled {len(resting)-n_fail}, "
                      f"failed {n_fail}; halted until next UTC day",
                      flush=True)
                sink.flush()
                n += 1
                continue

            # spot AFTER the stop: a glitched/stale/failed print skips only
            # quoting for this cycle (fix-review M2 + panel 6 staleness)
            S, spot_age = spot_now()
            glitch = ((prev_S is not None and abs(S / prev_S - 1) > 0.03)
                      or spot_age > 30)
            if glitch:
                sink.write("spot_glitch", {"prev": prev_S, "now": S,
                                           "age_s": round(spot_age, 1)})
            prev_S = S
            if not glitch:
                vol.update(S, t0)
            sig = vol.sigma_1s
            if glitch:  # stop evaluated above; only quoting sits out
                sink.flush()
                n += 1
                time.sleep(max(0.0, CYCLE_S - (time.time() - t0)))
                continue
            pos = {p["ticker"]: float(p.get("position_fp") or p.get("position", 0))
                   for p in k.get("/portfolio/positions").get(
                       "market_positions", [])}
            net_total = sum(pos.values())  # all markets share the BTC underlying
            long_capped = net_total > NET_CAP
            short_capped = net_total < -NET_CAP

            # audit CRITICAL: open markets exceed one page and Kalshi puts
            # the near-money near-expiry strikes on page 2 — without cursor
            # pagination the maker silently quoted NOTHING for hours
            mkts = []
            for series in ("KXBTC", "KXBTCD"):
                cursor = ""
                for _page in range(12):
                    resp = k.get(f"/markets?series_ticker={series}"
                                 f"&status=open&limit=200"
                                 + (f"&cursor={cursor}" if cursor else ""))
                    mkts += resp.get("markets", [])
                    cursor = resp.get("cursor") or ""
                    if not cursor:
                        break
                else:
                    if cursor:  # fix-review M1: never truncate SILENTLY —
                        sink.write("mkts_truncated",  # that class of bug
                                   {"series": series, "n": len(mkts)})
            now = time.time()
            cands = []
            for m in mkts:
                fs = m.get("floor_strike")
                if not fs or abs(fs / S - 1) > BAND:
                    continue
                close = calendar.timegm(time.strptime(
                    m["close_time"], "%Y-%m-%dT%H:%M:%SZ"))
                tau = close - now
                if not 600 < tau < 16 * 3600:
                    continue
                st = sig * math.sqrt(tau)
                # T market = P(S>floor); B market = P(floor<S<cap)
                fv = norm.cdf(math.log(S / fs) / st)
                cap = m.get("cap_strike")
                if cap:
                    fv -= norm.cdf(math.log(S / cap) / st)
                if not 0.10 < fv < 0.90:
                    continue
                cands.append((abs(fv - 0.5), m, fv, tau))
            cands.sort(key=lambda x: x[0])
            targets = {}
            sig_f = vol.sigma_fast
            for _, m, fv, tau in cands[:MAX_STRIKES]:
                st = sig * math.sqrt(tau)
                # audit: with one sigma the width was 3.784/sqrt(tau) — the
                # sigmas CANCELLED and quotes never widened into a spike.
                # Sensitivity uses the slow leg, the expected move the fast
                # leg, so delta scales with sigma_fast/sigma_slow.
                delta = min(0.25, PHI0 / st * 3 * sig_f * math.sqrt(CYCLE_S))
                inv = pos.get(m["ticker"], 0.0)
                # clamp to the live book: post-only join-or-behind, never cross
                try:
                    ob = k.get(f"/markets/{m['ticker']}/orderbook?depth=1"
                               ).get("orderbook_fp", {})
                    yb = ob.get("yes_dollars") or []
                    nb = ob.get("no_dollars") or []
                    book_bid = float(yb[-1][0]) if yb else None
                    book_ask = 1.0 - float(nb[-1][0]) if nb else None
                except Exception:  # noqa: BLE001
                    continue  # no book read -> do not quote blind this cycle
                if book_bid is None and book_ask is None:
                    # empty book = no clamp possible: never rest naked fv+-d
                    # (half the near-money demo brackets; audit rank-1c)
                    continue
                q = {}
                if inv < INV_CAP and not long_capped:
                    b = max(0.01, round(fv - delta, 2))
                    if book_ask is not None:
                        b = min(b, round(book_ask - 0.01, 2))
                    if b >= 0.01:
                        q["bid"] = b
                if inv > -INV_CAP and not short_capped:
                    a = min(0.99, round(fv + delta, 2))
                    if book_bid is not None:
                        a = max(a, round(book_bid + 0.01, 2))
                    if a <= 0.99:
                        q["ask"] = a
                targets[m["ticker"]] = {"fv": fv, "delta": delta, **q}

            # reconcile: cancel stale/off-target orders, place missing ones
            open_orders = k.get("/portfolio/orders?status=resting"
                                ).get("orders", [])
            live = {}
            for o in open_orders:
                tick, oid = o["ticker"], o["order_id"]
                o_side, o_act = o.get("side"), o.get("action")
                if o_side == "yes" and o_act == "buy":
                    side = "bid"
                elif o_side == "yes" and o_act == "sell":
                    side = "ask"
                elif o_side == "no" and o_act == "buy":
                    side = "ask"   # buy-no == sell-yes economically
                elif o_side == "no" and o_act == "sell":
                    side = "bid"
                else:
                    sink.write("order_warn", {"ticker": tick, "oid": oid,
                                              "side": o_side, "action": o_act,
                                              "note": "unrecognized, skipping"})
                    continue
                px = o.get("yes_price_dollars")
                if px is None and o.get("no_price_dollars") is not None:
                    px = 1.0 - float(o["no_price_dollars"])
                px = float(px) if px is not None else None
                created = o.get("created_time")
                live[(tick, side)] = (oid, px, created)
            canceled_any = False
            now_utc = time.time()
            for (tick, side), (oid, px, created) in list(live.items()):
                tgt = targets.get(tick, {}).get(side)
                age = 0.0
                if created:
                    try:
                        age = now_utc - calendar.timegm(time.strptime(
                            created[:19], "%Y-%m-%dT%H:%M:%S"))
                    except ValueError:
                        age = 0.0
                stale = tgt is None or px is None or abs(px - tgt) > STICKY
                near_expiry = age > 90  # renew before the 120s self-expiry
                if stale or near_expiry:
                    try:
                        k.delete(f"/portfolio/events/orders/{oid}")
                        canceled_any = True
                        live[(tick, side)] = None
                    except Exception:  # noqa: BLE001
                        pass  # keep slot occupied: order may still be live
            if canceled_any:
                time.sleep(1.0)  # let cancels land before re-quoting: avoids
                # post-only crossing our own in-flight canceled orders
            # collateral budget: placing what cash can't back just 400-spams
            # insufficient_balance (10k+ rejects on the first low-cash run).
            # balance_dollars is GROSS cash (verified: constant as orders
            # rest), so first subtract collateral already committed by the
            # resting orders we are keeping this cycle.
            reserved = sum(
                (v[1] * SIZE if side == "bid" else (1.0 - v[1]) * SIZE)
                for (t_, side), v in live.items()
                if v is not None and v[1] is not None)
            budget = bal * 0.9 - reserved
            # HARD net-inventory projection (nightly r2 money F1): the old
            # per-cycle-start check let one adverse cycle overshoot NET_CAP
            # ~2x (25 fresh contracts on 5 strikes) — settlement loss past
            # the stop. Count every kept resting order and every new order
            # as if it fills; never project past the cap.
            net_proj = net_total
            for (t_, side), v in live.items():
                if v is not None:
                    net_proj += SIZE if side == "bid" else -SIZE
            n_skip_budget = n_skip_clamp = n_skip_net = 0
            for tick, tgt in targets.items():
                fresh = None
                for side in ("bid", "ask"):
                    if side not in tgt or live.get((tick, side)):
                        continue
                    if side == "bid" and net_proj + SIZE > NET_CAP:
                        n_skip_net += 1
                        continue
                    if side == "ask" and net_proj - SIZE < -NET_CAP:
                        n_skip_net += 1
                        continue
                    # v2 API: side bid=buy yes, ask=sell yes; price is yes price
                    price = tgt[side]
                    if canceled_any:
                        # the clamp book is >=1s stale after the cancel settle
                        # (that staleness was the residual post-only-cross
                        # source); re-clamp against a fresh top level
                        if fresh is None:
                            try:
                                ob = k.get(f"/markets/{tick}/orderbook?depth=1"
                                           ).get("orderbook_fp", {})
                                yb = ob.get("yes_dollars") or []
                                nb = ob.get("no_dollars") or []
                                fresh = {
                                    "bid": float(yb[-1][0]) if yb else None,
                                    "ask": 1.0 - float(nb[-1][0]) if nb else None}
                            except Exception:  # noqa: BLE001
                                fresh = {"bid": None, "ask": None}
                        if side == "bid" and fresh["ask"] is not None:
                            price = min(price, round(fresh["ask"] - 0.01, 2))
                        if side == "ask" and fresh["bid"] is not None:
                            price = max(price, round(fresh["bid"] + 0.01, 2))
                        if not 0.01 <= price <= 0.99:
                            n_skip_clamp += 1
                            continue
                    cost = price * SIZE if side == "bid" else (1 - price) * SIZE
                    if cost > budget:
                        n_skip_budget += 1
                        continue
                    try:
                        k.post("/portfolio/events/orders", {
                            "ticker": tick,
                            "client_order_id": str(uuid.uuid4()),
                            "side": side, "count": f"{SIZE}.00",
                            "price": f"{price:.4f}",
                            "time_in_force": "good_till_canceled",
                            "self_trade_prevention_type": "maker",
                            "post_only": True,
                            "cancel_order_on_pause": True,
                            # poor-man's dead-man switch: orders self-expire;
                            # a live loop re-places them, a dead loop leaves
                            # nothing resting beyond 120s
                            "expiration_time": int(time.time()) + 120})
                        budget -= cost
                        net_proj += SIZE if side == "bid" else -SIZE
                    except Exception as e:  # noqa: BLE001
                        sink.write("order_err", {"ticker": tick, "side": side,
                                                 "err": str(e)[:200]})
            record_fills({"spot": S, "sigma": sig, "sigma_fast": sig_f,
                          "net_before": net_total,
                          "n_open": len(open_orders)},
                         {**last_targets, **targets})
            last_targets = targets
            sink.write("maker_cycle", {
                "spot": S, "sigma_1s": sig, "balance": bal,
                "equity": round(equity, 2), "eq_med": round(eq_med, 2),
                "baseline": round(start_bal, 2) if start_bal is not None else None,
                "skip_budget": n_skip_budget, "skip_clamp": n_skip_clamp,
                "skip_net": n_skip_net,
                "targets": {t: {kk: round(v, 4) if isinstance(v, float) else v
                                for kk, v in d.items()}
                            for t, d in targets.items()},
                "n_open": len(open_orders), "pos": pos})
        except Exception as e:  # noqa: BLE001
            try:
                sink.write("maker_err", str(e)[:300])
            except OSError:
                pass  # ENOSPC here must degrade, not crash-loop (panel 6)
        try:
            sink.flush()
        except OSError:
            pass
        n += 1
        time.sleep(max(0.0, CYCLE_S - (time.time() - t0)))
    print("maker loop ended", flush=True)


if __name__ == "__main__":
    main()
