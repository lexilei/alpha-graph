"""Incremental-IC factor selector: residualize a candidate against the
accepted set per monthly cross-section, test whether the residual still
predicts fwd_return_21d. In-sample tool — pre-register thresholds and count
every variant tried (see reports/factor_preregistration.md).

Usage:
  python scripts/factor_orthogonality.py evaluate --accepted momentum_21d --candidate cosine_similarity
  python scripts/factor_orthogonality.py greedy [--candidates ...] [--t 2.0]
  python scripts/factor_orthogonality.py greedy --panel data/cache/my_factors.parquet --candidates ...
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

FWD_COL = "fwd_return_21d"
MIN_XS = 20          # need at least this many stocks in a month to score it
MIN_MONTHS = 12      # need at least this many scored months to trust the IC


# --------------------------------------------------------------------------- #
# Panel loading + monthly sampling
# --------------------------------------------------------------------------- #

def load_panel(path: str | None) -> pd.DataFrame:
    if path:
        panel = pd.read_parquet(path)
    else:
        from alpha_graph.signals.ml_combiner import build_feature_panel
        panel = build_feature_panel()
    panel["date"] = pd.to_datetime(panel["date"])
    if FWD_COL not in panel.columns:
        raise ValueError(f"panel needs a '{FWD_COL}' column; has {list(panel.columns)}")
    return panel


def to_monthly(panel: pd.DataFrame) -> pd.DataFrame:
    """Keep the last trading-day row per (ticker, month) to limit 21d overlap."""
    panel = panel.copy()
    panel["month"] = panel["date"].dt.to_period("M")
    panel = panel.sort_values("date").groupby(["ticker", "month"], as_index=False).tail(1)
    return panel


# --------------------------------------------------------------------------- #
# Cross-sectional residualization + IC
# --------------------------------------------------------------------------- #

def _xs_rank_z(s: pd.Series) -> pd.Series:
    """Cross-sectional rank -> standardized to mean 0 / std 1."""
    r = s.rank()
    if r.notna().sum() < 2 or r.std(ddof=0) == 0:
        return pd.Series(np.nan, index=s.index)
    return (r - r.mean()) / r.std(ddof=0)


def evaluate(panel_m: pd.DataFrame, accepted: list[str], candidate: str) -> dict:
    """Per-month: residualize candidate on accepted, measure raw vs incremental IC."""
    needed = [candidate, FWD_COL] + accepted
    ic_raw, ic_resid, r2_list, n_xs = [], [], [], []
    # per-accepted correlation accumulators
    corr_acc = {a: [] for a in accepted}

    for _, g in panel_m.groupby("month"):
        sub = g[needed].dropna()
        if len(sub) < MIN_XS:
            continue

        cand = _xs_rank_z(sub[candidate])
        fwd = _xs_rank_z(sub[FWD_COL])
        if cand.isna().all() or fwd.isna().all():
            continue

        # raw IC
        ic_raw.append(np.corrcoef(cand, fwd)[0, 1])

        if accepted:
            X = np.column_stack([_xs_rank_z(sub[a]).values for a in accepted])
            X = np.column_stack([np.ones(len(sub)), X])
            beta, *_ = np.linalg.lstsq(X, cand.values, rcond=None)
            fitted = X @ beta
            resid = cand.values - fitted
            ss_tot = np.sum((cand.values - cand.values.mean()) ** 2)
            r2 = 1 - np.sum(resid ** 2) / ss_tot if ss_tot > 0 else 0.0
            r2_list.append(r2)
            rz = (resid - resid.mean()) / (resid.std(ddof=0) or 1.0)
            ic_resid.append(np.corrcoef(rz, fwd)[0, 1])
            for a in accepted:
                corr_acc[a].append(np.corrcoef(cand, _xs_rank_z(sub[a]))[0, 1])
        else:
            r2_list.append(0.0)
            ic_resid.append(ic_raw[-1])
        n_xs.append(len(sub))

    if len(ic_raw) < MIN_MONTHS:
        return {"candidate": candidate, "n_months": len(ic_raw), "ok": False}

    def agg(x):
        x = np.array(x)
        m, sd = x.mean(), x.std(ddof=1)
        t = m / (sd / np.sqrt(len(x))) if sd > 0 else 0.0
        icir = m / sd if sd > 0 else 0.0
        return m, t, icir

    raw_m, raw_t, raw_icir = agg(ic_raw)
    inc_m, inc_t, inc_icir = agg(ic_resid)
    return {
        "candidate": candidate,
        "accepted": list(accepted),
        "n_months": len(ic_raw),
        "avg_xs": int(np.mean(n_xs)),
        "spanned_r2": float(np.mean(r2_list)),
        "orthogonality": float(1 - np.mean(r2_list)),
        "corr_to_accepted": {a: float(np.mean(v)) for a, v in corr_acc.items() if v},
        "ic_raw": float(raw_m), "ic_raw_t": float(raw_t), "ic_raw_icir": float(raw_icir),
        "ic_incremental": float(inc_m), "ic_incremental_t": float(inc_t),
        "ic_incremental_icir": float(inc_icir),
        "ok": True,
    }


def _print_eval(res: dict) -> None:
    if not res.get("ok"):
        print(f"  {res['candidate']}: too few months ({res['n_months']}) — skip")
        return
    print(f"\n  candidate: {res['candidate']}   accepted: {res['accepted'] or '[] (standalone)'}")
    print(f"    months={res['n_months']}  avg cross-section={res['avg_xs']}")
    if res["accepted"]:
        print(f"    spanned R² by accepted : {res['spanned_r2']:.3f}   "
              f"orthogonality (1-R²): {res['orthogonality']:.3f}")
        cs = "  ".join(f"{a}={c:+.2f}" for a, c in res["corr_to_accepted"].items())
        print(f"    corr to accepted       : {cs}")
    print(f"    IC raw          : {res['ic_raw']:+.4f}  (t={res['ic_raw_t']:+.2f}, ICIR={res['ic_raw_icir']:+.2f})")
    print(f"    IC incremental  : {res['ic_incremental']:+.4f}  (t={res['ic_incremental_t']:+.2f}, "
          f"ICIR={res['ic_incremental_icir']:+.2f})  <-- decides inclusion")


# --------------------------------------------------------------------------- #
# Greedy forward stepwise build
# --------------------------------------------------------------------------- #

def greedy(panel_m: pd.DataFrame, candidates: list[str], t_threshold: float) -> None:
    accepted: list[str] = []
    remaining = list(candidates)
    print("=" * 78)
    print(f"  GREEDY FORWARD BUILD  (accept if incremental-IC t > {t_threshold})")
    print("=" * 78)
    rnd = 0
    while remaining:
        rnd += 1
        scored = [evaluate(panel_m, accepted, c) for c in remaining]
        scored = [s for s in scored if s.get("ok")]
        if not scored:
            print("\n  no scorable candidates left — stop")
            break
        scored.sort(key=lambda s: abs(s["ic_incremental_t"]), reverse=True)
        print(f"\n--- round {rnd}  (accepted so far: {accepted or '[]'}) ---")
        for s in scored:
            star = "  <-- best" if s is scored[0] else ""
            print(f"  {s['candidate']:>18}: incr IC {s['ic_incremental']:+.4f} "
                  f"t={s['ic_incremental_t']:+.2f}  ortho={s['orthogonality']:.2f}{star}")
        best = scored[0]
        if abs(best["ic_incremental_t"]) <= t_threshold:
            print(f"\n  best incremental t={best['ic_incremental_t']:+.2f} <= {t_threshold} "
                  f"— STOP. Nothing left adds signal over {accepted}.")
            break
        accepted.append(best["candidate"])
        remaining.remove(best["candidate"])
        print(f"  ==> ACCEPT {best['candidate']}  (incr IC t={best['ic_incremental_t']:+.2f})")

    print("\n" + "=" * 78)
    print(f"  FINAL ACCEPTED SET (in build order): {accepted}")
    print(f"  REJECTED (redundant / no incremental signal): "
          f"{[c for c in candidates if c not in accepted]}")
    print("=" * 78)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

DEFAULT_NINE = [
    "cosine_similarity", "event_score", "event_count", "regime_state",
    "regime_prob", "momentum_21d", "momentum_5d", "volatility_21d", "volume_zscore",
]


def main():
    p = argparse.ArgumentParser(description="Factor orthogonality + incremental-IC tester")
    p.add_argument("mode", choices=["evaluate", "greedy"])
    p.add_argument("--panel", default=None, help="parquet with date,ticker,fwd_return_21d + factors")
    p.add_argument("--accepted", nargs="*", default=[], help="already-accepted factors (evaluate mode)")
    p.add_argument("--candidate", help="single candidate to score (evaluate mode)")
    p.add_argument("--candidates", nargs="*", default=DEFAULT_NINE, help="candidate pool (greedy mode)")
    p.add_argument("--t", type=float, default=2.0, help="incremental-IC t threshold to accept (greedy)")
    args = p.parse_args()

    panel = load_panel(args.panel)
    panel_m = to_monthly(panel)
    logger.info(f"Monthly panel: {panel_m['month'].nunique()} months, "
                f"{panel_m['ticker'].nunique()} tickers")

    if args.mode == "evaluate":
        if not args.candidate:
            p.error("evaluate mode needs --candidate")
        _print_eval(evaluate(panel_m, args.accepted, args.candidate))
    else:
        greedy(panel_m, args.candidates, args.t)


if __name__ == "__main__":
    main()
