"""FV-maker on Kalshi BTC hourly ladders (demo or production).

STATUS 2026-07-26: the pricing side of this is NOT validated. The first
production session lost 88% of a $49.53 account and the forensics
(`reports/audit_maker_2026-07-26.md`) found (a) the Kalshi demo environment
settles every market "no", so nothing here has ever been validated anywhere,
and (b) a suspected bias in the Gaussian terminal density on bracket
(interval) markets. This file is the CONTROL layer rebuilt; the pricing
layer is unchanged and still on trial. Do not point it at production again
before the offline replay (A) and the density study (B) return.

Policy each cycle (10s):
  spot   Coinbase BTC-USD last trade (BRTI-constituent proxy; Kalshi settles
         on the 60s BRTI TWAP)
  sigma  two-speed EWMA of log returns from the in-process spot buffer
  for each KXBTC/KXBTCD strike within BAND of spot and TAU_MIN_S..TAU_MAX_S
  from close:
    FV = Phi( ln(S/K) / (sigma*sqrt(tau)) )        [minus the cap leg for
                                                    bracket markets]
    delta = PHI0/(sigma*sqrt(tau)) * 3*sigma_fast*sqrt(CYCLE_S)   [q99~3sigma]
    quote post_only bid/ask at FV -/+ delta, skewed against inventory,
    clamped to the live book, SIZE contracts a side, sticky within STICKY

Risk (rebuilt 2026-07-26, see `alpha_graph.trading.binary_risk`):
  the gate is `cash + worst-case settlement payout` against a high-water
  loss budget -- not net position, which reads flat on a book of disjoint
  brackets that can all lose at once. Every candidate order is projected
  as if it fills before it is sent. Breaching cancels AND flattens; the
  budget has no date in it, so a calendar roll cannot re-arm it.

Logs: data/raw/polymarket/{makerprod,demomaker}_*.jsonl.gz (sink format).

Usage: .venv/bin/python scripts/kalshi_demo_maker.py [--cycles N]
"""

from __future__ import annotations

import importlib.util
import json
import calendar
import math
import signal
import sys
from datetime import datetime, timezone
import time
import urllib.request
import uuid
from pathlib import Path

from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location(
    "kc", ROOT / "scripts" / "kalshi_client.py")
kc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kc)
import polymarket_recorder as rec  # noqa: E402  (Sink)
from alpha_graph.trading.binary_risk import (  # noqa: E402
    HighWaterBudget, gross_contracts, parse_exchange_ts, region_from_strikes,
    worst_case_payoff,
)

PHI0 = 0.3989
CYCLE_S = 10.0
STICKY = 0.02
SIZE = 5
BAND = 0.05          # strike band vs spot; nearest MAX_STRIKES picked below
TAU_MIN_S = 600      # skip the last 10 min: TWAP gamma, and no time to flatten
TAU_MAX_S = 45 * 60  # audit: the old bound was 16h. Quoting markets that
# settle four hours out locked every dollar of collateral in them, and the
# account then sat unable to quote for four straight hours (skip_budget fired
# 14,013 times against skip_net's 16). The docstring always claimed 10-50min;
# the code never did.
INV_SKEW = 0.02      # dollars of quote shift per SKEW_REF contracts of inventory
SKEW_REF = 20.0
MAX_SKEW = 0.05
HALT_SLEEP_S = 15.0   # halted cadence; also the shutdown-signal response time
CANCEL_SETTLE_S = 1.0
_sleep = time.sleep   # seam: CYCLE_S is the quote-refresh horizon the width
# formula is built on, so tests cannot zero it to skip waiting

# environment is FILE-driven (data/private/maker_env.txt: "demo"|"prod"),
# never argv: the watchdog relaunches with bare argv, and an argv switch
# would silently revert prod->demo on the first restart (panel review B1).
# All durable state (budget/stop-flag/sink) is namespaced per env so a
# demo budget can never disarm the prod one (B2).
_ENV_F = rec.OUT.parent.parent / "private" / "maker_env.txt"
try:
    ENV = _ENV_F.read_text().strip().lower()
except (OSError, ValueError):  # ValueError: non-UTF-8 file must fail safe
    ENV = "demo"               # to demo, not crash-loop at import (r2 LOW)
if ENV not in ("demo", "prod"):
    ENV = "demo"
PROD = ENV == "prod"

# LOSS_BUDGET is measured against the account's high-water mark, not against
# a daily baseline. The 2026-07-25 stop fired at 22:54Z and the UTC day roll
# re-armed it 71 minutes later with a fresh $15 on an account already down
# 49%. A budget with a date in it can be spent once per tick of that date.
LOSS_BUDGET = 15.0 if PROD else 25.0
MAX_STRIKES = 5 if PROD else 12
MAX_GROSS = 60 if PROD else 400   # coarse backstop; the real gate is the
# worst-case settlement projection below
SINK_NAME = "makerprod" if PROD else "demomaker"

_STOPPING = {"flag": False, "signal": None}


def _on_signal(signum, _frame) -> None:
    """Ask the loop to wind down. The old build had no exit path at all:
    its docstring promised "cancel everything on exit" while a TERM simply
    killed the process, leaving resting orders to the 120s self-expiry and
    every open position untouched. That is how the 2026-07-25 session ended
    -- 96 gross contracts left on the account, unattended, for 16 minutes."""
    _STOPPING["flag"] = True
    _STOPPING["signal"] = signum


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
        ts = parse_exchange_ts(t)
        if ts is not None:
            age = max(0.0, time.time() - ts.timestamp())
    return float(tk["price"]), age


class Vol:
    """Two-speed EWMA variance. Audit: a single 300s-halflife estimator
    lagged a real spike 4.6x for minutes. The slow leg centers fv (stable);
    the fast leg drives the width so quotes widen INTO a spike.

    Caveat carried forward from the audit: this estimates *instantaneous*
    vol from 10s returns and extrapolates by sqrt(tau). It cannot see the
    kurtosis of the hourly terminal distribution, which is what the bracket
    markets actually pay on. That is question (B), not fixed here."""

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


class Regions:
    """Settlement geometry per ticker, cached on disk.

    The risk gate needs the payout region of every position, including
    positions in markets that have already left the open list. Missing
    geometry is never guessed: `worst_case_payoff` assumes an unknown
    contract pays nothing, so a cache miss makes the gate stricter, not
    looser."""

    def __init__(self, path: Path, fetch):
        self.path, self.fetch = path, fetch
        self.raw: dict[str, dict] = {}
        self.map: dict = {}
        self.missing: set[str] = set()
        try:
            self.raw = json.loads(path.read_text())
        except (OSError, ValueError):
            self.raw = {}
        self._rebuild()

    def _rebuild(self) -> None:
        self.map = {}
        for t, v in self.raw.items():
            reg = region_from_strikes(v.get("floor"), v.get("cap"))
            if reg is not None:
                self.map[t] = reg

    def learn(self, markets) -> None:
        new = False
        for m in markets:
            t = m.get("ticker")
            if not t or t in self.raw:
                continue
            self.raw[t] = {"floor": m.get("floor_strike"), "cap": m.get("cap_strike")}
            self.missing.discard(t)
            new = True
        if new:
            self._rebuild()
            self._save()

    def ensure(self, tickers) -> int:
        """Fetch geometry for held tickers we have never seen. Returns the
        number still unknown after trying."""
        unknown = 0
        for t in tickers:
            if t in self.map:
                continue
            if t in self.missing:
                unknown += 1
                continue
            try:
                m = self.fetch(t)
            except Exception:  # noqa: BLE001
                self.missing.add(t)   # negative-cache: do not hammer per cycle
                unknown += 1
                continue
            self.raw[t] = {"floor": m.get("floor_strike"), "cap": m.get("cap_strike")}
            self._rebuild()
            self._save()
            if t not in self.map:
                self.missing.add(t)
                unknown += 1
        return unknown

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.raw))
            tmp.replace(self.path)
        except OSError:
            pass  # a cold cache only makes the gate stricter


def read_positions(k) -> dict[str, float]:
    return {p["ticker"]: float(p.get("position_fp") or p.get("position", 0))
            for p in k.get("/portfolio/positions").get("market_positions", [])}


def top_of_book(k, ticker) -> tuple[float | None, float | None]:
    ob = k.get(f"/markets/{ticker}/orderbook?depth=1").get("orderbook_fp", {})
    yb = ob.get("yes_dollars") or []
    nb = ob.get("no_dollars") or []
    return (float(yb[-1][0]) if yb else None,
            1.0 - float(nb[-1][0]) if nb else None)


def cancel_all(k, sink) -> tuple[int, int]:
    try:
        resting = k.get("/portfolio/orders?status=resting").get("orders", [])
    except Exception as e:  # noqa: BLE001
        sink.write("cancel_all_err", str(e)[:200])
        return 0, -1
    ok = fail = 0
    for o in resting:
        try:
            k.delete(f"/portfolio/events/orders/{o['order_id']}")
            ok += 1
        except Exception:  # noqa: BLE001
            fail += 1
    return ok, fail


def flatten(k, sink) -> tuple[int, int, float]:
    """Close every open position with marketable limits.

    The 2026-07-25 stop cancelled orders and stopped quoting, which reduced
    risk by exactly $0: there were no resting orders, and the 101 contracts
    that carried the exposure were untouched. A stop that cannot neutralise
    the exposure it exists to bound is not a stop. Crossing the spread here
    is expensive on purpose -- it is the price of the control working.

    Returns (closed, failed, residual_gross)."""
    try:
        pos = read_positions(k)
    except Exception as e:  # noqa: BLE001
        sink.write("flatten_err", {"stage": "positions", "err": str(e)[:200]})
        return 0, -1, math.nan
    closed = failed = 0
    for tick, q in pos.items():
        if abs(q) < 0.01:
            continue
        try:
            book_bid, book_ask = top_of_book(k, tick)
        except Exception as e:  # noqa: BLE001
            sink.write("flatten_err", {"ticker": tick, "stage": "book",
                                       "err": str(e)[:200]})
            failed += 1
            continue
        if q > 0:                       # long YES -> sell YES into the bid
            if book_bid is None:
                failed += 1
                sink.write("flatten_err", {"ticker": tick, "stage": "no_bid"})
                continue
            side, price = "ask", max(0.01, min(0.99, round(book_bid, 2)))
        else:                           # long NO -> buy YES at the ask
            if book_ask is None:
                failed += 1
                sink.write("flatten_err", {"ticker": tick, "stage": "no_ask"})
                continue
            side, price = "bid", max(0.01, min(0.99, round(book_ask, 2)))
        try:
            k.post("/portfolio/events/orders", {
                "ticker": tick,
                "client_order_id": str(uuid.uuid4()),
                "side": side, "count": f"{abs(q):.2f}",
                "price": f"{price:.4f}",
                "time_in_force": "good_till_canceled",
                "post_only": False,        # crossing is the point
                "cancel_order_on_pause": True,
                "expiration_time": int(time.time()) + 60})
            closed += 1
            sink.write("flatten_order", {"ticker": tick, "qty": q,
                                         "side": side, "price": price})
        except Exception as e:  # noqa: BLE001
            failed += 1
            sink.write("flatten_err", {"ticker": tick, "stage": "order",
                                       "err": str(e)[:200]})
    residual = gross_contracts(pos)
    sink.write("flatten", {"closed": closed, "failed": failed,
                           "gross_before": residual})
    return closed, failed, residual


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
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    k = kc.Kalshi(ENV)
    sink = rec.Sink(SINK_NAME)
    vol = Vol()
    flag_dir = rec.OUT.parent.parent / "logs" / "launchd"
    flag_dir.mkdir(parents=True, exist_ok=True)
    regions = Regions(flag_dir / f"maker_regions_{ENV}.json",
                      lambda t: k.get(f"/markets/{t}").get("market", {}))

    # the stop and the budget both read a 30-cycle (~5min) lower median of
    # RISK equity: settlement makes balance and portfolio_value jump for
    # minutes at a time, and the lower median is both robust to that and
    # conservative (leans toward stopping) on even-length windows
    eq_hist: list[float] = []
    seen_fills: dict[str, bool] = {}
    last_targets: dict = {}
    prev_S: float | None = None

    budget_f = flag_dir / f"maker_highwater_{ENV}.txt"
    try:
        budget = HighWaterBudget.load(LOSS_BUDGET, budget_f.read_text())
    except OSError:
        budget = HighWaterBudget(LOSS_BUDGET)

    def _save_budget():
        try:
            tmp = budget_f.with_suffix(".tmp")   # atomic: a torn write would
            tmp.write_text(budget.serialise())   # defeat the durability this
            tmp.replace(budget_f)                # file exists for
        except OSError:
            pass

    # fill telemetry (nightly r2 data F1 + logic MED-1): the in-memory-only
    # dedup dropped every fill that landed during a restart gap, and on a
    # FRESH account the first real batch was mis-classified as history. A
    # durable per-env watermark (max recorded created_time) fixes both; a
    # restart may re-record the single boundary fill (dup > loss; offline
    # dedupe by trade_id). Timestamps are PARSED, never string-compared:
    # Kalshi mixes 3..6 fractional digits and a fraction-free stamp sorts
    # after a fractional one as a raw string.
    fillmark_f = flag_dir / f"maker_fillmark_{ENV}.txt"
    try:
        fill_wm = parse_exchange_ts(fillmark_f.read_text().strip())
    except OSError:
        fill_wm = None  # first-ever session: snapshot on the first poll
    fill_wm_seen = fill_wm is not None

    def _write_fillmark():
        try:
            tmp = fillmark_f.with_suffix(".tmp")
            tmp.write_text(fill_wm.isoformat().replace("+00:00", "Z")
                           if fill_wm else "")
            tmp.replace(fillmark_f)
        except OSError:
            pass

    def record_fills(base_ctx: dict, tgt_map: dict | None = None) -> None:
        nonlocal fill_wm, fill_wm_seen
        try:
            fl = k.get("/portfolio/fills?limit=100").get("fills", [])
        except Exception:  # noqa: BLE001
            return
        stamps = [parse_exchange_ts(f_.get("created_time")) for f_ in fl]
        if not fill_wm_seen:
            # first successful poll EVER (empty or not): everything present
            # is pre-session history — snapshot the watermark, record none
            known = [s for s in stamps if s]
            fill_wm = max(known) if known else None
            fill_wm_seen = True
            _write_fillmark()
            return
        new_max = fill_wm
        for f_, ct in zip(fl, stamps):
            if fill_wm is not None and ct is not None and ct < fill_wm:
                continue  # strictly older than the watermark: history
            fid = str(f_.get("trade_id") or f_.get("fill_id")
                      or str(f_.get("order_id", "")) + str(f_.get("created_time")))
            if fid in seen_fills:
                continue
            seen_fills[fid] = True
            ctx_t = (tgt_map or {}).get(f_.get("ticker")) or {}
            sink.write("fill", {"raw": f_, "ctx": {
                **base_ctx, "fv": ctx_t.get("fv"),
                "our_bid": ctx_t.get("bid"), "our_ask": ctx_t.get("ask")}})
            if ct is not None and (new_max is None or ct > new_max):
                new_max = ct
        if new_max != fill_wm:
            fill_wm = new_max
            _write_fillmark()
        while len(seen_fills) > 5000:
            seen_fills.pop(next(iter(seen_fills)))

    print(f"maker start env={ENV} budget=${LOSS_BUDGET} strikes={MAX_STRIKES} "
          f"tau={TAU_MIN_S}-{TAU_MAX_S}s max_gross={MAX_GROSS} "
          f"high_water={budget.high_water}", flush=True)

    n = 0
    try:
        while (max_cycles is None or n < max_cycles) and not _STOPPING["flag"]:
            t0 = time.time()
            try:
                # equity and the loss stop are evaluated FIRST: a spot-feed
                # failure must skip quoting, never blind the stop (panel 6)
                bal_resp = k.get("/portfolio/balance")
                bal = float(bal_resp["balance_dollars"])
                # portfolio_value is in CENTS (no _dollars twin in the payload)
                mark_eq = bal + float(bal_resp.get("portfolio_value", 0) or 0) / 100.0
                pos = read_positions(k)
                n_unknown = regions.ensure(
                    [t for t, q in pos.items() if abs(q) >= 0.01])
                wc = worst_case_payoff(pos, regions.map)
                # THE risk number: what the account is worth after every open
                # contract settles, minimised over the settlement price. Free
                # of marks, so no mid-price artefact can move it.
                risk_eq = bal + wc.value
                eq_hist = (eq_hist + [risk_eq])[-30:]
                risk_med = sorted(eq_hist)[(len(eq_hist) - 1) // 2]

                # always warm up ~100s after a restart: a single balance
                # read taken mid-settlement is not a risk number, and the
                # high-water mark persists across restarts anyway
                warm = len(eq_hist) < 10
                if not warm:
                    before = budget.high_water
                    budget.observe(risk_med)
                    if budget.high_water != before:
                        _save_budget()
                floor = budget.floor()

                stop_flag = flag_dir / f"maker_stop_{ENV}.flag"
                if stop_flag.exists():
                    # halted; the flag survives restarts so a restart cannot
                    # silently defeat the stop. It is cleared by hand, never
                    # by a clock: the old build's flag name carried the UTC
                    # date and expired itself 71 minutes after firing.
                    # Fills are still polled: the fills that CAUSED the stop
                    # are the most diagnostic records of all (r2 data F2).
                    record_fills({"halted": True}, last_targets)
                    sink.write("maker_halted", {          # heartbeat: the old
                        "balance": bal, "risk_eq": round(risk_eq, 2),
                        "mark_eq": round(mark_eq, 2),     # halted branch wrote
                        "gross": wc.gross,                # nothing at all and
                        "floor": floor})                  # left a 71-min hole
                    sink.flush()                          # in the telemetry
                    n += 1
                    _sleep(HALT_SLEEP_S)
                    continue

                if warm:
                    # not armed yet, so DO NOT QUOTE — round-4 review caught
                    # this window placing orders with the loss stop disarmed
                    sink.write("maker_warmup", {"balance": bal, "risk_eq": risk_eq,
                                                "n_hist": len(eq_hist)})
                    sink.flush()
                    n += 1
                    _sleep(max(0.0, CYCLE_S - (time.time() - t0)))
                    continue

                breached = budget.breached(risk_med) or wc.gross > MAX_GROSS
                if breached:
                    n_ok, n_fail = cancel_all(k, sink)
                    closed, cfail, residual = flatten(k, sink)
                    try:
                        stop_flag.write_text(
                            f"risk_eq {risk_med:.2f} floor {floor:.2f} "
                            f"gross {wc.gross:.1f} at {datetime.now(timezone.utc)}\n")
                    except OSError:
                        pass  # disk-full: halting still holds via per-cycle
                        # breach re-entry; only restart persistence is lost
                    sink.write("maker_stop", {
                        "balance": bal, "mark_eq": round(mark_eq, 2),
                        "risk_eq": round(risk_eq, 2), "risk_med": round(risk_med, 2),
                        "floor": floor, "high_water": budget.high_water,
                        "worst_at": wc.at_price, "gross": wc.gross,
                        "n_unknown": n_unknown,
                        "canceled": n_ok, "cancel_failed": n_fail,
                        "flatten_sent": closed, "flatten_failed": cfail,
                        "residual_gross": residual})
                    print(f"STOP risk_eq {risk_med:.2f} < floor {floor:.2f}: "
                          f"cancelled {n_ok}, flatten sent {closed}; halted "
                          f"until {stop_flag} is removed by hand", flush=True)
                    sink.flush()
                    n += 1
                    _sleep(CYCLE_S)
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
                    _sleep(max(0.0, CYCLE_S - (time.time() - t0)))
                    continue

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
                regions.learn(mkts)
                now = time.time()
                cands = []
                for m in mkts:
                    fs = m.get("floor_strike")
                    if not fs or abs(fs / S - 1) > BAND:
                        continue
                    close = calendar.timegm(time.strptime(
                        m["close_time"], "%Y-%m-%dT%H:%M:%SZ"))
                    tau = close - now
                    if not TAU_MIN_S < tau < TAU_MAX_S:
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
                    tick = m["ticker"]
                    st = sig * math.sqrt(tau)
                    # audit: with one sigma the width was 3.784/sqrt(tau) — the
                    # sigmas CANCELLED and quotes never widened into a spike.
                    # Sensitivity uses the slow leg, the expected move the fast
                    # leg, so delta scales with sigma_fast/sigma_slow.
                    delta = min(0.25, PHI0 / st * 3 * sig_f * math.sqrt(CYCLE_S))
                    inv = pos.get(tick, 0.0)
                    # lean the whole quote against inventory instead of only
                    # cutting a side at a hard cap: the old build had no skew
                    # at all between 0 and INV_CAP, which it never reached
                    skew = max(-MAX_SKEW, min(MAX_SKEW, INV_SKEW * inv / SKEW_REF))
                    # clamp to the live book: post-only join-or-behind, never cross
                    try:
                        book_bid, book_ask = top_of_book(k, tick)
                    except Exception:  # noqa: BLE001
                        continue  # no book read -> do not quote blind this cycle
                    if book_bid is None and book_ask is None:
                        # empty book = no clamp possible: never rest naked fv+-d
                        # (half the near-money demo brackets; audit rank-1c)
                        continue
                    q = {}
                    b = max(0.01, round(fv - delta - skew, 2))
                    if book_ask is not None:
                        b = min(b, round(book_ask - 0.01, 2))
                    if b >= 0.01:
                        q["bid"] = b
                    a = min(0.99, round(fv + delta - skew, 2))
                    if book_bid is not None:
                        a = max(a, round(book_bid + 0.01, 2))
                    if a <= 0.99:
                        q["ask"] = a
                    targets[tick] = {"fv": fv, "delta": delta, "skew": skew, **q}

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
                    live[(tick, side)] = (oid, px, o.get("created_time"))
                canceled_any = False
                now_utc = time.time()
                for (tick, side), v in list(live.items()):
                    if v is None:
                        continue
                    oid, px, created = v
                    tgt = targets.get(tick, {}).get(side)
                    age = 0.0
                    cts = parse_exchange_ts(created)
                    if cts is not None:
                        age = now_utc - cts.timestamp()
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
                    _sleep(CANCEL_SETTLE_S)  # let cancels land before
                    # re-quoting: avoids
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
                cash_budget = bal * 0.9 - reserved
                # HARD worst-case projection: assume every kept resting order
                # AND the order about to be sent fills, then ask what the book
                # settles to at its worst. The old build projected the NET
                # position against a net cap, which is blind to a ladder of
                # disjoint brackets that can all lose at once.
                proj_pos = dict(pos)
                for (t_, side), v in live.items():
                    if v is not None:
                        proj_pos[t_] = proj_pos.get(t_, 0.0) + (
                            SIZE if side == "bid" else -SIZE)
                proj_cash = bal - reserved
                n_skip_budget = n_skip_clamp = n_skip_risk = 0
                for tick, tgt in targets.items():
                    fresh = None
                    for side in ("bid", "ask"):
                        if side not in tgt or live.get((tick, side)):
                            continue
                        price = tgt[side]
                        if canceled_any:
                            # the clamp book is >=1s stale after the cancel
                            # settle (that staleness was the residual
                            # post-only-cross source); re-clamp against a
                            # fresh top level
                            if fresh is None:
                                try:
                                    bb, ba = top_of_book(k, tick)
                                    fresh = {"bid": bb, "ask": ba}
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
                        if cost > cash_budget:
                            n_skip_budget += 1
                            continue
                        cand = dict(proj_pos)
                        cand[tick] = cand.get(tick, 0.0) + (
                            SIZE if side == "bid" else -SIZE)
                        cand_wc = worst_case_payoff(cand, regions.map)
                        if (proj_cash - cost) + cand_wc.value < floor:
                            n_skip_risk += 1
                            continue
                        if cand_wc.gross > MAX_GROSS:
                            n_skip_risk += 1
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
                            cash_budget -= cost
                            proj_cash -= cost
                            proj_pos = cand
                        except Exception as e:  # noqa: BLE001
                            sink.write("order_err", {"ticker": tick, "side": side,
                                                     "err": str(e)[:200]})
                record_fills({"spot": S, "sigma": sig, "sigma_fast": sig_f,
                              "risk_eq": round(risk_eq, 2),
                              "n_open": len(open_orders)},
                             {**last_targets, **targets})
                last_targets = targets
                sink.write("maker_cycle", {
                    "spot": S, "sigma_1s": sig, "balance": bal,
                    "mark_eq": round(mark_eq, 2), "risk_eq": round(risk_eq, 2),
                    "risk_med": round(risk_med, 2),
                    "high_water": budget.high_water,
                    "floor": round(floor, 2) if floor is not None else None,
                    "worst_at": wc.at_price, "gross": wc.gross,
                    "n_unknown": n_unknown,
                    "skip_budget": n_skip_budget, "skip_clamp": n_skip_clamp,
                    "skip_risk": n_skip_risk,
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
            _sleep(max(0.0, CYCLE_S - (time.time() - t0)))
    finally:
        # the exit path the old build promised in its docstring and never had
        n_ok, n_fail = cancel_all(k, sink)
        try:
            left = read_positions(k)
        except Exception:  # noqa: BLE001
            left = {}
        residual = gross_contracts(left)
        sink.write("maker_exit", {"signal": _STOPPING["signal"],
                                  "canceled": n_ok, "cancel_failed": n_fail,
                                  "residual_gross": residual,
                                  "residual": left})
        try:
            sink.flush()
        except OSError:
            pass
        if residual:
            print(f"EXIT: cancelled {n_ok} orders; {residual:.1f} contracts "
                  f"STILL OPEN across {len(left)} markets — they will settle "
                  f"unattended. Flatten by hand if that is not intended.",
                  flush=True)
        else:
            print(f"EXIT: cancelled {n_ok} orders; flat.", flush=True)


if __name__ == "__main__":
    main()
