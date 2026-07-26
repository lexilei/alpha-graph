"""End-to-end tests for the Kalshi FV-maker control loop.

These drive ``main()`` against a fake exchange that holds real state
(balance, positions, resting orders) and record what the maker *decides*.
They exist because the 2026-07-25 production session shipped with zero
coverage of this file: 25 rounds of review all read the code, none ran it,
and the audit found a stop that cancelled nothing, a budget that a calendar
roll re-armed, and an exit path that the docstring promised and the code
never had.

No network, no credentials, no real clock.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def maker():
    """Load scripts/kalshi_demo_maker.py by path (scripts/ is not a package)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    try:
        spec = importlib.util.spec_from_file_location(
            "kalshi_maker_under_test", ROOT / "scripts" / "kalshi_demo_maker.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    return mod


# ---------------------------------------------------------------- fake venue

def _ladder(spot: float, minutes: int = 30, n: int = 4, width: float = 800.0):
    """A KXBTC bracket ladder centred on spot, closing `minutes` from now.

    ``width`` is chosen so the near-money brackets clear the maker's own
    ``0.10 < fv < 0.90`` filter at the cold-start sigma (3e-4 per root-second
    => ~$815 of price sigma over 30 minutes). Narrower brackets are simply
    never quoted, which is itself the behaviour the real session showed.
    """
    close = (datetime.now(timezone.utc) + timedelta(minutes=minutes)
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    base = round(spot / width) * width
    out = []
    for i in range(-(n // 2), n // 2 + 1):
        lo = base + i * width
        out.append({"ticker": f"KXBTC-TEST-B{int(lo + width / 2)}",
                    "floor_strike": float(lo),
                    "cap_strike": float(lo + width - 0.01),
                    "close_time": close})
    return out


class FakeKalshi:
    """Stateful stand-in for the exchange. Records every mutation."""

    def __init__(self, cash=100.0, positions=None, markets=None,
                 book=(0.30, 0.70), portfolio_value_cents=0):
        self.cash = cash
        self.positions = dict(positions or {})
        self.markets = list(markets if markets is not None else _ladder(64000.0))
        self.book = book
        self.portfolio_value_cents = portfolio_value_cents
        self.orders: list[dict] = []      # every POST
        self.cancels: list[str] = []      # every DELETE
        self.resting: list[dict] = []
        self.fills: list[dict] = []
        self.gets: list[str] = []

    # -- transport ---------------------------------------------------------
    def get(self, path):
        self.gets.append(path)
        if path.startswith("/portfolio/balance"):
            return {"balance_dollars": f"{self.cash:.4f}",
                    "portfolio_value": self.portfolio_value_cents}
        if path.startswith("/portfolio/positions"):
            return {"market_positions": [{"ticker": t, "position_fp": f"{q:.2f}"}
                                         for t, q in self.positions.items()]}
        if path.startswith("/portfolio/fills"):
            return {"fills": list(self.fills)}
        if path.startswith("/portfolio/orders"):
            return {"orders": list(self.resting)}
        if path.startswith("/markets?"):
            series = path.split("series_ticker=")[1].split("&")[0]
            if series != "KXBTC":
                return {"markets": [], "cursor": ""}
            return {"markets": list(self.markets), "cursor": ""}
        if "/orderbook" in path:
            bid, ask = self.book
            return {"orderbook_fp": {
                "yes_dollars": [[f"{bid:.2f}", "100"]] if bid is not None else [],
                "no_dollars": [[f"{1 - ask:.2f}", "100"]] if ask is not None else []}}
        if path.startswith("/markets/"):
            tick = path.split("/markets/")[1]
            for m in self.markets:
                if m["ticker"] == tick:
                    return {"market": m}
            raise RuntimeError(f"unknown market {tick}")
        raise AssertionError(f"unexpected GET {path}")

    def post(self, path, body):
        assert path == "/portfolio/events/orders"
        self.orders.append(dict(body))
        oid = f"o{len(self.orders)}"
        if body.get("post_only"):          # resting quotes stay on the book
            self.resting.append({
                "ticker": body["ticker"], "order_id": oid, "side": "yes",
                "action": "buy" if body["side"] == "bid" else "sell",
                "yes_price_dollars": body["price"],
                "created_time": datetime.now(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ")})
        return {"order": {"order_id": oid}}

    def delete(self, path):
        oid = path.rsplit("/", 1)[-1]
        self.cancels.append(oid)
        self.resting = [o for o in self.resting if o["order_id"] != oid]
        return {}

    def premium_committed(self):
        """Cash the resting quotes would consume if they all filled."""
        tot = 0.0
        for o in self.orders:
            if not o.get("post_only"):
                continue
            px = float(o["price"])
            tot += (px if o["side"] == "bid" else 1 - px) * float(o["count"])
        return tot


class FakeSink:
    def __init__(self, *_a, **_k):
        self.records: list[tuple[str, object]] = []

    def write(self, src, msg):
        self.records.append((src, msg))

    def flush(self):
        pass

    def of(self, src):
        return [m for s, m in self.records if s == src]


@pytest.fixture
def harness(maker, tmp_path, monkeypatch):
    """Wire the maker to a fake venue, a temp state dir and a zero clock."""
    sink = FakeSink()
    state = tmp_path / "logs" / "launchd"
    state.mkdir(parents=True)
    monkeypatch.setattr(maker.rec, "OUT", tmp_path / "raw" / "polymarket")
    monkeypatch.setattr(maker.rec, "Sink", lambda *a, **k: sink)
    monkeypatch.setattr(maker, "spot_now", lambda: (64000.0, 0.0))
    monkeypatch.setattr(maker, "_sleep", lambda _s: None)
    monkeypatch.setattr(maker, "CANCEL_SETTLE_S", 0.0)
    monkeypatch.setattr(maker, "LOSS_BUDGET", 15.0)
    monkeypatch.setattr(maker, "MAX_GROSS", 60)
    monkeypatch.setattr(maker, "MAX_STRIKES", 5)
    maker._STOPPING["flag"] = False
    maker._STOPPING["signal"] = None

    def run(fake, cycles=12, high_water=None, stop_flag=False):
        if high_water is not None:
            (state / f"maker_highwater_{maker.ENV}.txt").write_text(str(high_water))
        if stop_flag:
            (state / f"maker_stop_{maker.ENV}.flag").write_text("halted\n")
        monkeypatch.setattr(maker.kc, "Kalshi", lambda env: fake)
        monkeypatch.setattr(sys, "argv", ["maker", "--cycles", str(cycles)])
        maker.main()
        return sink, state

    return run


# ------------------------------------------------------------- static shape

def test_tau_window_is_not_sixteen_hours(maker):
    """The docstring always said 10-50min; the code said 10min-16h, and the
    collateral lock-up that caused cost four hours of the session."""
    assert maker.TAU_MIN_S == 600
    assert maker.TAU_MAX_S <= 60 * 60


def test_no_net_position_cap_survives(maker):
    """A net cap on a book of disjoint brackets is the defect, not the fix."""
    assert not hasattr(maker, "NET_CAP")
    assert not hasattr(maker, "INV_CAP")


def test_stop_flag_name_carries_no_date(maker, harness):
    """The old flag was maker_stop_{env}_{utc_date}.flag, so the calendar
    cleared the halt 71 minutes after it fired."""
    fake = FakeKalshi(cash=20.0)
    sink, state = harness(fake, cycles=12, high_water=50.0)
    flags = list(state.glob("maker_stop_*"))
    assert flags, "expected the stop to fire"
    for f in flags:
        assert not any(ch.isdigit() for ch in f.stem.split("_")[-1])


# ------------------------------------------------------------------ warmup

def test_warmup_quotes_nothing(maker, harness):
    fake = FakeKalshi(cash=100.0)
    sink, _ = harness(fake, cycles=5, high_water=100.0)
    assert len(sink.of("maker_warmup")) == 5
    assert fake.orders == []


def test_quotes_once_armed(maker, harness):
    fake = FakeKalshi(cash=100.0)
    sink, _ = harness(fake, cycles=12, high_water=100.0)
    assert fake.orders, "expected quotes after warmup"
    assert all(o["post_only"] is True for o in fake.orders)
    assert all(int(o["expiration_time"]) > time.time() for o in fake.orders)
    cyc = sink.of("maker_cycle")
    assert cyc and cyc[-1]["floor"] == pytest.approx(85.0)


# --------------------------------------------------- the pre-trade risk gate

def test_never_commits_more_premium_than_the_headroom(maker, harness):
    """Cash $36 against a $35 floor leaves $1 of headroom.

    For a fully prepaid binary the worst case is that it pays nothing, so
    premium outstanding IS the loss exposure. The maker may commit exactly
    the headroom and not a cent more, however many strikes it likes.
    """
    fake = FakeKalshi(cash=36.0)
    sink, _ = harness(fake, cycles=12, high_water=50.0)
    cyc = sink.of("maker_cycle")
    assert cyc and sum(c["skip_risk"] for c in cyc) > 0, "gate never engaged"
    assert fake.premium_committed() <= 1.0 + 1e-9


def test_quotes_are_two_sided_and_straddle_fair_value(maker, harness):
    """Regression on the harness itself: zeroing the sleep must not zero the
    quote width (CYCLE_S is the horizon the width formula is built on)."""
    fake = FakeKalshi(cash=100.0)
    sink, _ = harness(fake, cycles=12, high_water=100.0)
    tgts = [t for c in sink.of("maker_cycle") for t in c["targets"].values()]
    assert tgts
    assert all(t["delta"] > 0.0 for t in tgts)
    straddling = [t for t in tgts if "bid" in t and "ask" in t]
    assert straddling
    assert all(t["bid"] < t["ask"] for t in straddling)


def test_headroom_lets_orders_through(maker, harness):
    """Same setup, more headroom: the gate must not be unconditionally shut."""
    fake = FakeKalshi(cash=60.0)
    sink, _ = harness(fake, cycles=12, high_water=50.0)
    assert fake.orders


def test_gate_sees_through_a_flat_net_position(maker, harness):
    """Two disjoint brackets netting to zero: the old gate reads flat.

    Long the low bracket, short the high one. Net is 0, gross is 60, and
    there is a settlement price at which neither pays. The maker must not
    treat that as riskless capacity.
    """
    mkts = _ladder(64000.0)
    lo_t, hi_t = mkts[0]["ticker"], mkts[-1]["ticker"]
    fake = FakeKalshi(cash=36.0, positions={lo_t: 30.0, hi_t: -30.0},
                      markets=mkts)
    assert sum(fake.positions.values()) == 0.0        # the old metric: flat
    sink, _ = harness(fake, cycles=12, high_water=50.0)
    cyc = sink.of("maker_cycle")
    assert cyc
    assert cyc[-1]["gross"] == 60.0
    assert cyc[-1]["risk_eq"] == pytest.approx(36.0)  # nothing is guaranteed
    assert fake.orders == []


# ------------------------------------------------------------------- the stop

def test_stop_cancels_and_flattens(maker, harness):
    """The 2026-07-25 stop cancelled 0 orders and closed 0 contracts."""
    mkts = _ladder(64000.0)
    held = mkts[0]["ticker"]
    fake = FakeKalshi(cash=20.0, positions={held: 30.0}, markets=mkts)
    fake.resting = [{"ticker": held, "order_id": "abc",
                     "side": "yes", "action": "buy",
                     "yes_price_dollars": "0.30",
                     "created_time": "2026-07-25T16:00:00Z"}]
    sink, state = harness(fake, cycles=12, high_water=50.0)

    stops = sink.of("maker_stop")
    assert stops, "expected a stop"
    assert "abc" in fake.cancels
    closing = [o for o in fake.orders if o["post_only"] is False]
    assert closing, "the stop must send closing orders, not just cancels"
    assert closing[0]["ticker"] == held
    assert closing[0]["side"] == "ask"          # long YES -> sell YES
    assert float(closing[0]["count"]) == pytest.approx(30.0)
    assert stops[0]["flatten_sent"] >= 1
    assert (state / f"maker_stop_{maker.ENV}.flag").exists()


def test_stop_closes_a_short_by_buying_yes(maker, harness):
    mkts = _ladder(64000.0)
    held = mkts[0]["ticker"]
    fake = FakeKalshi(cash=20.0, positions={held: -12.0}, markets=mkts)
    sink, _ = harness(fake, cycles=12, high_water=50.0)
    closing = [o for o in fake.orders if o["post_only"] is False]
    assert closing and closing[0]["side"] == "bid"
    assert float(closing[0]["count"]) == pytest.approx(12.0)


def test_max_gross_backstop_fires(maker, harness):
    mkts = _ladder(64000.0)
    fake = FakeKalshi(cash=100.0,
                      positions={m["ticker"]: 40.0 for m in mkts}, markets=mkts)
    sink, _ = harness(fake, cycles=12, high_water=100.0)
    assert sink.of("maker_stop"), "gross 200 must trip the backstop"


# ------------------------------------------------------------------- halted

def test_halted_writes_a_heartbeat_and_quotes_nothing(maker, harness):
    """The old halted branch wrote nothing at all: 71 minutes of silence
    across the worst part of the loss."""
    fake = FakeKalshi(cash=20.0)
    sink, _ = harness(fake, cycles=6, high_water=50.0, stop_flag=True)
    beats = sink.of("maker_halted")
    assert len(beats) == 6
    assert all("risk_eq" in b and "gross" in b for b in beats)
    assert fake.orders == []


def test_halt_survives_a_restart(maker, harness):
    fake = FakeKalshi(cash=20.0)
    sink, state = harness(fake, cycles=3, high_water=50.0, stop_flag=True)
    assert (state / f"maker_stop_{maker.ENV}.flag").exists()
    assert not sink.of("maker_cycle")


# --------------------------------------------------------------------- exit

def test_exit_cancels_orders_and_reports_residual(maker, harness, capsys):
    mkts = _ladder(64000.0)
    held = mkts[0]["ticker"]
    fake = FakeKalshi(cash=100.0, positions={held: 7.0}, markets=mkts)
    fake.resting = [{"ticker": held, "order_id": "zzz", "side": "yes",
                     "action": "buy", "yes_price_dollars": "0.30",
                     "created_time": "2026-07-25T16:00:00Z"}]
    sink, _ = harness(fake, cycles=2, high_water=100.0)
    exits = sink.of("maker_exit")
    assert len(exits) == 1
    assert "zzz" in fake.cancels
    assert exits[0]["residual_gross"] == pytest.approx(7.0)
    assert "STILL OPEN" in capsys.readouterr().out


def test_signal_breaks_the_loop(maker, harness):
    fake = FakeKalshi(cash=100.0)
    maker._on_signal(15, None)                 # SIGTERM
    sink, _ = harness(fake, cycles=1000, high_water=100.0)
    assert sink.of("maker_exit")
    assert len(sink.of("maker_warmup")) == 0   # never entered the loop body


# ------------------------------------------------- the recorded session book

@pytest.mark.skipif(not (FIXTURES / "kalshi_btc_strikes_2026-07-25.json").exists(),
                    reason="strike fixture absent")
def test_recorded_book_would_not_be_added_to(maker, harness):
    """Replay the real 20:02:47Z book. Cash $0.06, marked equity $57.64.

    The old build quoted five fresh strikes into this. The rebuilt gate must
    place nothing and must halt.
    """
    strikes = json.loads((FIXTURES / "kalshi_btc_strikes_2026-07-25.json").read_text())
    book = {"KXBTC-26JUL2517-B64125": -30.0,
            "KXBTC-26JUL2517-B64375": +5.0,
            "KXBTCD-26JUL2517-T63999.99": -14.0,
            "KXBTCD-26JUL2517-T64249.99": +35.0}
    close = (datetime.now(timezone.utc) + timedelta(minutes=30)
             ).strftime("%Y-%m-%dT%H:%M:%SZ")
    mkts = [{"ticker": t, "floor_strike": strikes[t]["floor"],
             "cap_strike": strikes[t]["cap"], "close_time": close}
            for t in book] + _ladder(64200.0)
    fake = FakeKalshi(cash=0.0599, positions=book, markets=mkts,
                      portfolio_value_cents=5758)
    sink, _ = harness(fake, cycles=12, high_water=49.527)

    assert not [o for o in fake.orders if o["post_only"] is True]
    stops = sink.of("maker_stop")
    assert stops, "risk equity $0.06 against a $34.53 floor must halt"
    assert stops[0]["risk_eq"] == pytest.approx(0.0599, abs=1e-3)
    assert stops[0]["mark_eq"] == pytest.approx(57.64, abs=0.01)
    assert 64100.0 < stops[0]["worst_at"] <= 64249.99
