"""Unit tests for alpha_graph.options: BS inversion round-trip, parity spot,
dedup/crossed-quote semantics, ATM selection, and the Black-76 no-wedge
property on synthetic chains."""
import math

import pandas as pd
import pytest

from alpha_graph.options.bs import bs_price, implied_vol_bisect
from alpha_graph.options.iv import atm_iv_for_day, paired_mids, parity_spot

HALF_SPREAD = 0.005  # locked/crossed quotes are filtered, so quotes need
# width; kept below the ~0.01 deep-OTM wing prices so bids stay positive


def synth_chain(spot=100.0, vol=0.25, rate=0.04, qd="2024-01-02", exp="2024-02-09"):
    """Chain of BS-priced quotes (symmetric spread, mid == model price)."""
    t = (pd.Timestamp(exp) - pd.Timestamp(qd)).days / 365.0
    rows = []
    for k in range(80, 121, 5):
        for right, ot in (("CALL", "C"), ("PUT", "P")):
            px = bs_price(ot, spot, float(k), t, rate, vol)
            rows.append(
                {"quote_date": qd, "expiration": exp, "strike": float(k),
                 "right": right, "bid": px - HALF_SPREAD, "ask": px + HALF_SPREAD,
                 "created": "2024-01-02 17:00:00"}
            )
    return pd.DataFrame(rows)


def test_bs_roundtrip():
    px = bs_price("C", 100.0, 100.0, 0.1, 0.04, 0.3)
    iv = implied_vol_bisect(px, "C", 100.0, 100.0, 0.1, 0.04)
    assert iv is not None and abs(iv - 0.3) < 1e-4


def test_bisect_none_on_unattainable():
    assert implied_vol_bisect(0.0, "C", 100.0, 100.0, 0.1, 0.04) is None
    # below intrinsic: deep ITM call priced under S-K
    assert implied_vol_bisect(1.0, "C", 100.0, 50.0, 0.1, 0.04) is None


def test_paired_mids_keep_last_and_both_legs():
    chain = synth_chain()
    stale = chain.iloc[[0]].copy()
    stale["bid"] = 998.0
    stale["ask"] = 1000.0
    stale["created"] = "2024-01-02 09:00:00"  # earlier snapshot must lose
    lonely = pd.DataFrame(
        [{"quote_date": "2024-01-02", "expiration": "2024-02-09",
          "strike": 200.0, "right": "CALL", "bid": 0.05, "ask": 0.10,
          "created": "2024-01-02 17:00:00"}]
    )
    pairs = paired_mids(pd.concat([stale, chain, lonely], ignore_index=True))
    assert 200.0 not in pairs["strike"].values  # no put leg -> dropped
    k80 = pairs[pairs["strike"] == 80.0]
    assert len(k80) == 1 and float(k80["call_mid"].iloc[0]) < 100  # stale lost


def test_crossed_quotes_dropped():
    chain = synth_chain()
    crossed = pd.DataFrame(
        [{"quote_date": "2024-01-02", "expiration": "2024-02-09",
          "strike": 100.0, "right": right, "bid": 50.0, "ask": 1.0,
          "created": "2024-01-02 18:00:00"}  # latest -> would win dedup
         for right in ("CALL", "PUT")]
    )
    clean = paired_mids(chain)
    with_crossed = paired_mids(pd.concat([chain, crossed], ignore_index=True))
    # the crossed pair must not survive to poison the K=100 mids
    pd.testing.assert_frame_equal(
        clean.reset_index(drop=True), with_crossed.reset_index(drop=True)
    )


def test_parity_spot_recovers_forward():
    pairs = paired_mids(synth_chain(spot=100.0))
    s = parity_spot(pairs)
    # parity recovers the FORWARD: S*e^{rt} = 100*e^{0.04*38/365} ~ 100.42
    t = 38 / 365.0
    fwd = 100.0 * math.exp(0.04 * t)
    assert abs(float(s.iloc[0]) - fwd) / fwd < 0.001


def test_expiry_tie_breaks_later():
    qd = "2024-01-02"
    early = synth_chain(vol=0.20, qd=qd, exp="2024-02-07")  # 36 dte
    late = synth_chain(vol=0.30, qd=qd, exp="2024-02-09")   # 38 dte, same |Δ|
    r = atm_iv_for_day(pd.concat([early, late], ignore_index=True), qd)
    assert r is not None and r.dte == 38
    assert r.iv == pytest.approx(0.30, abs=0.005)


def test_atm_iv_recovers_vol_and_prefers_target_dte():
    qd = "2024-01-02"
    near = synth_chain(vol=0.25, qd=qd, exp="2024-02-09")   # 38 dte
    far = synth_chain(vol=0.60, qd=qd, exp="2024-06-21")    # out of window
    r = atm_iv_for_day(pd.concat([near, far], ignore_index=True), qd)
    assert r is not None and r.dte == 38
    assert r.iv == pytest.approx(0.25, abs=0.005)


def test_flat_vol_no_call_put_wedge():
    """Black-76 audit fix (2026-07-17 finding 1): under a flat true vol and
    nonzero carry, leg IVs must match each other and the truth — the old
    spot-BS-on-forward path manufactured a -3.2 vol-pt call-put wedge."""
    r = atm_iv_for_day(synth_chain(vol=0.25, rate=0.05), "2024-01-02", rate=0.05)
    assert r is not None and r.iv_call is not None and r.iv_put is not None
    assert r.iv_call == pytest.approx(0.25, abs=2e-3)
    assert r.iv_put == pytest.approx(0.25, abs=2e-3)
    assert abs(r.iv_call - r.iv_put) < 2e-3
