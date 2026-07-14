"""C23 `sc13d_drift`: activist-stake (SC 13D) announcement drift.

Registered 2026-07-14 (commit 77ffe85; family=event-paired, roles=sel).
Construction and the three pinned looks, executed exactly as registered
(reports/factor_preregistration.md, row "C23 registration").

Events
------
Original SC 13D filings from `sc13_filings.parquet` (is_amendment=False, form
canonical "SC 13D"; 230 rows / 136 subjects). `acceptance_ts` is native ET,
SGML-sourced. Availability of the announcement:

  - acceptance strictly before 16:00 ET on a trading day  -> that day's close;
  - acceptance at/after 16:00 ET, or on a non-trading day  -> the next trading
    close.

ENTRY is the NEXT close after availability (you trade the close after the close
at which the news is known). CAR is the cumulative simple (holding-period)
return close(entry) -> close(entry + k td): CAR_i(k) = close_i(entry+k) /
close_i(entry) - 1. The headline horizon is +63 td; the descriptive curve is
read at +1, +5, +21, +42, +63. Equal weight across events (each event is one
observation; there is no cross-sectional weighting). Prices are the adjusted
closes in `market_data.parquet`; the market calendar is that file's trading
days. An event is dropped when its subject lacks prices over the window
(subject absent from the panel, window before the subject's first listed day,
or the window runs past the calendar end); the drop count and reasons are
reported.

Control (a) — same-ticker, one year earlier
-------------------------------------------
The same subject's window shifted back 252 trading days (control entry =
entry_pos - 252). Skipped if the control window COLLIDES within +/-63 td of ANY
original 13D or 13G event for that ticker — collision anchor-to-anchor,
|filing_pos - control_ref| <= 63, where control_ref = entry_pos - 252 and
filing_pos is the first trading day >= the filing's acceptance date. "Event"
here is an ORIGINAL filing (is_amendment=False, form SC 13D or SC 13G) drawn
from the FULL parquet for that ticker (not just the 230 studied 13Ds); routine
amendments (13D/A, 13G/A) are not collision events. This isolates a clean same-
stock baseline uncontaminated by a fresh 5%-stake disclosure. Match rate
reported.

Control (b) — contemporaneous passive stake, different stock
------------------------------------------------------------
For each (priced) 13D, an original SC 13G (is_amendment=False, form SC 13G) on
a DIFFERENT subject, chosen as the nearest by `acceptance_ts`, via GREEDY
nearest-first assignment: over all eligible (13D, 13G) pairs sorted by
|acceptance difference| ascending, assign a pair when the 13D is still
unmatched and the 13G is usable. A 13G filing is used at most once (no reuse),
and no 13G subject supplies two controls whose entries are within +/-63 td of
each other ("not reused within +/-63 td"). Same availability / entry / CAR
rules on the 13G's subject; only 13Gs whose subject is priced over the control
window are eligible. This holds the mechanical 5%-stake-disclosure demand
fixed and isolates the activist INTENT unique to a 13D. Match rate reported.

The three pinned looks
----------------------
1. Paired real - control(a) mean 63-td CAR, HAC (Newey-West, Bartlett) t on the
   per-event differences ordered by entry date. NW lag = 5: neighbouring
   events' 63-td windows overlap (mean spacing ~16 td over the sample), so
   adjacent paired differences are serially dependent out to ~4 events; lag 5
   covers it. (The registration says "~5"; stated here as the choice.)
2. Paired real - control(b) mean 63-td CAR, same HAC t, same lag, same ordering.
3. The descriptive CAR curve: mean cumulative excess-over-control(a) at
   +1, +5, +21, +42, +63 (one look). Its +63 point equals look 1's mean by
   construction (same paired set).

Bar pinned pre-computation: candidate iff BOTH look 1 and look 2 HAC t >= +2.0;
else rejected. Applied mechanically in `decide`.

Cache
-----
data/cache/sc13d_drift_events.parquet — one row per original 13D: the event
id, availability/entry dates, real CAR curve, both controls' CAR curves and
identities, and the match / drop flags.

Usage:
    python -m alpha_graph.signals.sc13d_drift   # build cache + the 3 looks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from alpha_graph.config import CACHE_DIR
from alpha_graph.eval.ic_tools import hac_tstat

MARKET_CLOSE_HOUR = 16          # 16:00 ET availability boundary
HOLD_TDAYS = 63                 # CAR horizon (trading days)
CONTROL_LOOKBACK_TDAYS = 252    # control (a) shift, ~1 year
COLLISION_TDAYS = 63            # control (a) clean-window buffer, +/- td
REUSE_TDAYS = 63                # control (b) no-reuse window, +/- td
NW_LAGS = 5                     # HAC lag on the event-ordered paired diffs
CURVE_OFFSETS = (1, 5, 21, 42, 63)
PAIRED_T_BAR = 2.0              # pinned success bar (BOTH controls)
BSEARCH_K = 1000                # candidate 13Gs per 13D for greedy matching

EVENTS_CACHE = "sc13d_drift_events.parquet"


# --------------------------------------------------------------------------- #
# Calendar / availability / entry
# --------------------------------------------------------------------------- #

def availability_pos(acc_ts, cal: pd.DatetimeIndex) -> int | None:
    """Calendar index of the availability close for an acceptance timestamp.

    acc_ts is native ET. Returns None when the acceptance precedes the calendar
    (cannot be availability-mapped) or leaves no trading session after it.
    """
    acc = pd.Timestamp(acc_ts)
    acc_date = acc.normalize()
    if acc_date < cal[0]:
        return None
    right = int(cal.searchsorted(acc_date, side="right"))   # first idx > date
    le = right - 1
    is_trading = le >= 0 and cal[le] == acc_date
    if is_trading and acc.hour < MARKET_CLOSE_HOUR:          # strictly < 16:00
        return le
    # >= 16:00 on a trading day, or any time on a non-trading day: next close
    if right >= len(cal):
        return None
    return right


def entry_pos(acc_ts, cal: pd.DatetimeIndex) -> int | None:
    """Entry = the NEXT close after availability. None if out of calendar."""
    ap = availability_pos(acc_ts, cal)
    if ap is None:
        return None
    ep = ap + 1
    if ep >= len(cal):
        return None
    return ep


def filing_grid_pos(acc_ts, cal: pd.DatetimeIndex) -> int:
    """First trading day >= the acceptance date (collision anchor). Filings
    after the last session map to len(cal) (never collide with in-range
    controls)."""
    acc_date = pd.Timestamp(acc_ts).normalize()
    return int(cal.searchsorted(acc_date, side="left"))


# --------------------------------------------------------------------------- #
# CAR
# --------------------------------------------------------------------------- #

def car_curve(close_mat: np.ndarray, tick_idx: dict, ticker: str,
              ep: int | None, offsets=CURVE_OFFSETS) -> dict | None:
    """{offset: close(ep+offset)/close(ep) - 1}, or None if any needed price
    (subject column absent, or NaN at entry or any offset, or offset past the
    calendar) is missing — the "lacks prices in the window" drop rule."""
    j = tick_idx.get(ticker)
    if j is None or ep is None or ep < 0:
        return None
    if ep + max(offsets) > close_mat.shape[0] - 1:
        return None
    base = close_mat[ep, j]
    if not np.isfinite(base):
        return None
    out = {}
    for k in offsets:
        p = close_mat[ep + k, j]
        if not np.isfinite(p):
            return None
        out[k] = float(p / base - 1.0)
    return out


def collides(control_ref: int, positions: np.ndarray,
             tol: int = COLLISION_TDAYS) -> bool:
    """Any collision-event position within +/- tol of the control anchor."""
    if positions.size == 0:
        return False
    return bool(np.any(np.abs(positions - control_ref) <= tol))


# --------------------------------------------------------------------------- #
# Event set assembly
# --------------------------------------------------------------------------- #

def _orig(sc13: pd.DataFrame, form: str) -> pd.DataFrame:
    return sc13[(sc13["form"] == form) & (~sc13["is_amendment"])].copy()


def collision_positions(sc13: pd.DataFrame,
                        cal: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Per-ticker sorted calendar positions of ORIGINAL 13D and 13G events
    (is_amendment=False, form SC 13D or SC 13G) from the full parquet."""
    orig = sc13[(~sc13["is_amendment"])
                & (sc13["form"].isin(["SC 13D", "SC 13G"]))]
    pos = orig["acceptance_ts"].map(lambda t: filing_grid_pos(t, cal))
    out: dict[str, np.ndarray] = {}
    for tk, g in pos.groupby(orig["subject_ticker"]):
        out[tk] = np.sort(g.to_numpy())
    return out


def build_real_events(sc13: pd.DataFrame, cal: pd.DatetimeIndex,
                      close_mat: np.ndarray, tick_idx: dict) -> pd.DataFrame:
    """One row per original 13D with availability, entry, real CAR curve, and
    the priced flag / drop reason."""
    d13 = _orig(sc13, "SC 13D").sort_values(
        ["acceptance_ts", "subject_ticker"]).reset_index(drop=True)
    n = len(cal)
    rows = []
    for r in d13.itertuples():
        ap = availability_pos(r.acceptance_ts, cal)
        rec = {
            "subject_ticker": r.subject_ticker, "subject_cik": r.subject_cik,
            "accession": r.accession, "acceptance_ts": r.acceptance_ts,
            "avail_date": pd.NaT, "entry_date": pd.NaT, "entry_pos": -1,
            "real_priced": False, "drop_reason": "",
        }
        for k in CURVE_OFFSETS:
            rec[f"car{k}"] = np.nan
        if ap is None:
            rec["drop_reason"] = ("pre_calendar"
                                  if pd.Timestamp(r.acceptance_ts).normalize()
                                  < cal[0] else "post_calendar")
            rows.append(rec)
            continue
        ep = ap + 1
        rec["avail_date"] = cal[ap]
        if ep + HOLD_TDAYS > n - 1:
            rec["drop_reason"] = "short_window"
            rows.append(rec)
            continue
        rec["entry_date"] = cal[ep]
        rec["entry_pos"] = ep
        cr = car_curve(close_mat, tick_idx, r.subject_ticker, ep)
        if cr is None:
            rec["drop_reason"] = "unpriced"
            rows.append(rec)
            continue
        rec["real_priced"] = True
        for k, v in cr.items():
            rec[f"car{k}"] = v
        rows.append(rec)
    return pd.DataFrame(rows)


def add_control_a(events: pd.DataFrame, sc13: pd.DataFrame,
                  cal: pd.DatetimeIndex, close_mat: np.ndarray,
                  tick_idx: dict) -> pd.DataFrame:
    """Control (a): same subject, entry shifted -252 td; skip on collision or
    missing price. Adds a_matched, a_skip_reason, a_car{offset}."""
    coll = collision_positions(sc13, cal)
    events = events.copy()
    events["a_matched"] = False
    events["a_skip_reason"] = ""
    for k in CURVE_OFFSETS:
        events[f"a_car{k}"] = np.nan
    for i, r in events.iterrows():
        if not r["real_priced"]:
            events.at[i, "a_skip_reason"] = "real_unpriced"
            continue
        cref = int(r["entry_pos"]) - CONTROL_LOOKBACK_TDAYS
        if cref < 0:
            events.at[i, "a_skip_reason"] = "edge"
            continue
        if collides(cref, coll.get(r["subject_ticker"], np.array([], int))):
            events.at[i, "a_skip_reason"] = "collision"
            continue
        ca = car_curve(close_mat, tick_idx, r["subject_ticker"], cref)
        if ca is None:
            events.at[i, "a_skip_reason"] = "unpriced"
            continue
        events.at[i, "a_matched"] = True
        for k, v in ca.items():
            events.at[i, f"a_car{k}"] = v
    return events


def eligible_13g(sc13: pd.DataFrame, cal: pd.DatetimeIndex,
                 close_mat: np.ndarray, tick_idx: dict) -> pd.DataFrame:
    """Original SC 13Gs whose subject is priced over the control window, with
    entry position and CAR curve resolved."""
    g13 = _orig(sc13, "SC 13G").sort_values(
        ["acceptance_ts", "subject_ticker"]).reset_index(drop=True)
    recs = []
    for r in g13.itertuples():
        ep = entry_pos(r.acceptance_ts, cal)
        cg = car_curve(close_mat, tick_idx, r.subject_ticker, ep)
        if cg is None:
            continue
        rec = {"subject_ticker": r.subject_ticker, "accession": r.accession,
               "acceptance_ts": pd.Timestamp(r.acceptance_ts),
               "entry_pos": int(ep), "entry_date": cal[ep]}
        for k, v in cg.items():
            rec[f"g_car{k}"] = v
        recs.append(rec)
    return pd.DataFrame(recs)


def add_control_b(events: pd.DataFrame, g13: pd.DataFrame) -> pd.DataFrame:
    """Control (b): greedy nearest-`acceptance_ts` original 13G on a different
    subject, no filing reused, no subject reused within +/-63 td. Adds
    b_matched, b_ticker/accession/acceptance_ts/entry_date, b_car{offset}."""
    events = events.copy()
    events["b_matched"] = False
    events["b_ticker"] = pd.NA
    events["b_accession"] = pd.NA
    events["b_acceptance_ts"] = pd.NaT
    events["b_entry_date"] = pd.NaT
    for k in CURVE_OFFSETS:
        events[f"b_car{k}"] = np.nan

    priced = events.index[events["real_priced"]].to_numpy()
    if len(priced) == 0 or g13.empty:
        return events
    d_acc = events.loc[priced, "acceptance_ts"].map(
        lambda t: pd.Timestamp(t).value).to_numpy()
    d_sub = events.loc[priced, "subject_ticker"].to_numpy()
    g_acc = g13["acceptance_ts"].map(lambda t: t.value).to_numpy()
    g_sub = g13["subject_ticker"].to_numpy()
    g_ep = g13["entry_pos"].to_numpy()

    # Candidate pairs pruned to the nearest BSEARCH_K different-subject 13Gs
    # per 13D (>> the at-most-|D| steals + rare reuse blocks a 13D can suffer,
    # so the greedy result is identical to using all pairs).
    pairs = []
    for di in range(len(priced)):
        diff = np.abs(g_acc - d_acc[di])
        elig = np.where(g_sub != d_sub[di])[0]
        order = elig[np.argsort(diff[elig], kind="stable")[:BSEARCH_K]]
        for gi in order:
            pairs.append((int(diff[gi]), di, int(gi)))
    pairs.sort()  # (|dt|, d_idx, g_idx) — deterministic tie-break

    assigned: dict[int, int] = {}
    used_g: set[int] = set()
    subj_eps: dict[str, list[int]] = {}
    for _dt, di, gi in pairs:
        if di in assigned or gi in used_g:
            continue
        gsub, gep = g_sub[gi], int(g_ep[gi])
        if any(abs(gep - e) <= REUSE_TDAYS for e in subj_eps.get(gsub, ())):
            continue
        assigned[di] = gi
        used_g.add(gi)
        subj_eps.setdefault(gsub, []).append(gep)

    for di, gi in assigned.items():
        row = priced[di]
        g = g13.iloc[gi]
        events.at[row, "b_matched"] = True
        events.at[row, "b_ticker"] = g["subject_ticker"]
        events.at[row, "b_accession"] = g["accession"]
        events.at[row, "b_acceptance_ts"] = g["acceptance_ts"]
        events.at[row, "b_entry_date"] = g["entry_date"]
        for k in CURVE_OFFSETS:
            events.at[row, f"b_car{k}"] = g[f"g_car{k}"]
    return events


def build_events(sc13: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Full pipeline: 230 original 13Ds with real + both control CAR curves."""
    cal = pd.DatetimeIndex(np.sort(market["date"].unique()))
    close_mat = (market.pivot(index="date", columns="ticker", values="close")
                 .sort_index())
    tick_idx = {tk: j for j, tk in enumerate(close_mat.columns)}
    C = close_mat.to_numpy()
    events = build_real_events(sc13, cal, C, tick_idx)
    events = add_control_a(events, sc13, cal, C, tick_idx)
    g13 = eligible_13g(sc13, cal, C, tick_idx)
    events = add_control_b(events, g13)
    return events


# --------------------------------------------------------------------------- #
# The 3 pinned looks
# --------------------------------------------------------------------------- #

def paired_look(events: pd.DataFrame, control: str, horizon: int = HOLD_TDAYS,
                lags: int = NW_LAGS) -> dict:
    """Paired real - control mean CAR at `horizon` with HAC t on the per-event
    differences ordered by entry date. control in {'a', 'b'}."""
    matched = events["real_priced"] & events[f"{control}_matched"]
    sub = events.loc[matched].sort_values(
        ["entry_date", "subject_ticker"])
    diff = (sub[f"car{horizon}"] - sub[f"{control}_car{horizon}"]).to_numpy()
    n = len(diff)
    return {
        "control": control, "horizon": horizon, "n_pairs": n,
        "mean_real": float(sub[f"car{horizon}"].mean()) if n else float("nan"),
        "mean_control": float(sub[f"{control}_car{horizon}"].mean())
        if n else float("nan"),
        "mean_diff": float(diff.mean()) if n else float("nan"),
        "hac_lags": lags, "t_hac": hac_tstat(diff, lags) if n else float("nan"),
    }


def curve_look(events: pd.DataFrame, offsets=CURVE_OFFSETS) -> pd.DataFrame:
    """Descriptive CAR curve: mean real, mean control(a), and mean paired
    excess-over-control(a) at each offset (control-a-matched events)."""
    matched = events["real_priced"] & events["a_matched"]
    sub = events.loc[matched]
    rows = []
    for k in offsets:
        d = (sub[f"car{k}"] - sub[f"a_car{k}"])
        rows.append({"offset_td": k, "n": len(sub),
                     "mean_real_car": float(sub[f"car{k}"].mean()),
                     "mean_control_a_car": float(sub[f"a_car{k}"].mean()),
                     "mean_excess": float(d.mean())})
    return pd.DataFrame(rows)


def decide(look_a: dict, look_b: dict, bar: float = PAIRED_T_BAR) -> str:
    """Pinned bar: BOTH paired HAC t >= +2.0 -> candidate; else rejected."""
    passed = look_a["t_hac"] >= bar and look_b["t_hac"] >= bar
    return "candidate" if passed else "rejected"


# --------------------------------------------------------------------------- #
# Build + evaluate on the cached data
# --------------------------------------------------------------------------- #

def main() -> None:
    sc13 = pd.read_parquet(CACHE_DIR / "sc13_filings.parquet")
    market = pd.read_parquet(CACHE_DIR / "market_data.parquet",
                             columns=["date", "ticker", "close"])
    events = build_events(sc13, market)
    events.to_parquet(CACHE_DIR / EVENTS_CACHE, index=False)

    n_total = len(events)
    n_priced = int(events["real_priced"].sum())
    drops = events.loc[~events["real_priced"], "drop_reason"].value_counts()
    logger.info(f"events: {n_total} original 13D, {n_priced} priced "
                f"({events.loc[events['real_priced'], 'subject_ticker'].nunique()}"
                f" subjects); drops: {drops.to_dict()}")

    a_matched = int((events["real_priced"] & events["a_matched"]).sum())
    a_skips = events.loc[events["real_priced"], "a_skip_reason"].value_counts()
    logger.info(f"control (a) same-ticker -252td: matched {a_matched}/{n_priced}"
                f" ({a_matched / n_priced:.1%}); skips {a_skips.to_dict()}")
    b_matched = int((events["real_priced"] & events["b_matched"]).sum())
    logger.info(f"control (b) 13G nearest-acceptance: matched "
                f"{b_matched}/{n_priced} ({b_matched / n_priced:.1%})")

    look_a = paired_look(events, "a")
    look_b = paired_look(events, "b")
    logger.info(
        f"look 1/3 (paired real-control(a), 63td CAR, n {look_a['n_pairs']}): "
        f"real {look_a['mean_real']:+.4%}, control {look_a['mean_control']:+.4%}"
        f", diff {look_a['mean_diff']:+.4%}, HAC t (lag {look_a['hac_lags']}) "
        f"{look_a['t_hac']:+.2f}")
    logger.info(
        f"look 2/3 (paired real-control(b), 63td CAR, n {look_b['n_pairs']}): "
        f"real {look_b['mean_real']:+.4%}, control {look_b['mean_control']:+.4%}"
        f", diff {look_b['mean_diff']:+.4%}, HAC t (lag {look_b['hac_lags']}) "
        f"{look_b['t_hac']:+.2f}")

    curve = curve_look(events)
    logger.info("look 3/3 (descriptive CAR curve, excess over control(a)):\n"
                + curve.to_string(index=False))
    logger.info(f"pinned bar (BOTH paired HAC t >= +{PAIRED_T_BAR}) -> "
                f"{decide(look_a, look_b)}")


if __name__ == "__main__":
    main()
