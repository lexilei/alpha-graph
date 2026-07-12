"""Tests for the ic_tools diagnostics library (plan v2, Item 3)."""

import math

import numpy as np
import pandas as pd
import pytest

from alpha_graph.eval.ic_tools import (
    _xs_rank_z,
    default_lags,
    deflated_t,
    effective_n,
    emax_null,
    hac_tstat,
    ic_summary,
    monthly_ic,
    quantile_ls,
    rank_autocorr,
    sector_neutralize,
    split_halves,
)


# --------------------------------------------------------------------------- #
# Synthetic panel with a planted IC
# --------------------------------------------------------------------------- #

def _panel(n_months=120, n_names=200, ic=0.05, seed=7,
           persistence=0.0):
    """Monthly panel where factor rank-correlates with fwd return at ~ic.
    persistence: AR(1) coefficient of the factor across months (0 = fresh)."""
    rng = np.random.RandomState(seed)
    months = pd.period_range("2012-01", periods=n_months, freq="M")
    rows = []
    prev = rng.normal(size=n_names)
    for mo in months:
        f = persistence * prev + math.sqrt(1 - persistence ** 2) * rng.normal(size=n_names)
        prev = f
        ret = ic * f + math.sqrt(1 - ic ** 2) * rng.normal(size=n_names)
        rows.append(pd.DataFrame({
            "month": mo, "ticker": [f"T{i:03d}" for i in range(n_names)],
            "factor": f, "fwd_return_21d": ret,
            "sector": ["A", "B", "C", "D"] * (n_names // 4),
        }))
    return pd.concat(rows, ignore_index=True)


def test_monthly_ic_recovers_planted_ic():
    pm = _panel(ic=0.08)
    ic = monthly_ic(pm, "factor")
    assert len(ic) == 120
    assert ic.mean() == pytest.approx(0.08, abs=0.02)


def test_ic_summary_fields_and_naive_t():
    pm = _panel(ic=0.06)
    s = ic_summary(monthly_ic(pm, "factor"))
    assert s["ok"] and not s["naive_suppressed"]
    assert s["t_naive"] > 3  # 120 months of planted IC
    assert s["t_hac"] == pytest.approx(s["t_naive"], rel=0.35)


# --------------------------------------------------------------------------- #
# HAC
# --------------------------------------------------------------------------- #

def test_hac_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.RandomState(0)
    x = rng.normal(0.02, 0.1, 150)
    for lags in (1, 4, 12):
        ours = hac_tstat(x, lags)
        model = sm.OLS(x, np.ones(len(x))).fit(
            cov_type="HAC", cov_kwds={"maxlags": lags})
        assert ours == pytest.approx(float(model.tvalues[0]), rel=1e-6)


def test_hac_deflates_autocorrelated_series():
    rng = np.random.RandomState(1)
    e = rng.normal(size=300)
    x = np.convolve(e, np.ones(6) / 6, mode="valid") + 0.02  # strong AR structure
    naive = x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))
    assert abs(hac_tstat(x, 12)) < abs(naive) * 0.7


# --------------------------------------------------------------------------- #
# default_lags: sampling- and persistence-aware
# --------------------------------------------------------------------------- #

def test_default_lags_sampling_aware():
    assert default_lags(21, 21) == 1
    assert default_lags(21, 5) == 5      # weekly sampling of a 21d horizon
    assert default_lags(126, 21) == 6


def test_default_lags_persistence_floor():
    rng = np.random.RandomState(2)
    e = rng.normal(size=400)
    slow = np.convolve(e, np.ones(8) / 8, mode="valid")  # persistent IC series
    assert default_lags(21, 21, ic=slow) > 3
    fresh = rng.normal(size=400)
    assert default_lags(21, 21, ic=fresh) == 1


# --------------------------------------------------------------------------- #
# Guardrail: no naive t under overlap
# --------------------------------------------------------------------------- #

def test_naive_t_suppressed_when_interval_below_horizon():
    pm = _panel(ic=0.06)
    s = ic_summary(monthly_ic(pm, "factor"),
                   horizon_days=21, sampling_interval_days=5)
    assert s["naive_suppressed"] and s["t_naive"] is None
    assert s["t_hac"] is not None and s["hac_lags"] >= 5


# --------------------------------------------------------------------------- #
# split_halves boundary semantics (the 2019-drop bug class)
# --------------------------------------------------------------------------- #

def test_split_halves_boundary_inclusive_left():
    idx = pd.period_range("2012-01", "2020-12", freq="M")
    ic = pd.Series(0.0, index=idx)
    ic[idx <= pd.Period("2019-12", "M")] = 1.0   # all of 2012-2019 = 1
    r = split_halves(ic, "2019-12")
    # first half must contain ALL 96 months through Dec-2019; an off-by-one
    # boundary (e.g. < "2019") would change these counts
    assert r["first_n"] == 96 and r["second_n"] == 12
    assert r["first_mean"] == 1.0 and r["second_mean"] == 0.0


# --------------------------------------------------------------------------- #
# quantile_ls: turnover + break-even cost
# --------------------------------------------------------------------------- #

def test_quantile_ls_turnover_on_constructed_rotation():
    # factor = pure noise re-drawn each month -> top-quintile turnover ~ 80%
    pm = _panel(ic=0.0, persistence=0.0)
    r = quantile_ls(pm, "factor")
    assert r["ok"]
    assert r["turnover_top"] == pytest.approx(0.8, abs=0.08)
    # perfectly persistent factor -> ~zero turnover
    pm2 = _panel(ic=0.0, persistence=1.0)
    r2 = quantile_ls(pm2, "factor")
    assert r2["turnover_top"] == pytest.approx(0.0, abs=0.02)


def test_quantile_ls_break_even_scales_with_mean():
    pm = _panel(ic=0.10)
    r = quantile_ls(pm, "factor")
    assert r["mean_monthly"] > 0
    implied_drag = r["break_even_cost_bps"] / 1e4 * 2 * (
        r["turnover_top"] + r["turnover_bottom"])
    assert implied_drag == pytest.approx(r["mean_monthly"], rel=1e-9)


# --------------------------------------------------------------------------- #
# effective_n on block-correlated synthetics
# --------------------------------------------------------------------------- #

def test_effective_n_estimators():
    rng = np.random.RandomState(3)
    months = pd.period_range("2015-01", periods=60, freq="M")
    rows = []
    for mo in months:
        a = rng.normal(size=300)
        b = rng.normal(size=300)
        rows.append(pd.DataFrame({
            "month": mo, "ticker": [f"T{i}" for i in range(300)],
            # 4 factors in 2 near-duplicate blocks
            "f1": a, "f2": a + 0.05 * rng.normal(size=300),
            "f3": b, "f4": b + 0.05 * rng.normal(size=300),
        }))
    pm = pd.concat(rows, ignore_index=True)
    r = effective_n(pm, ["f1", "f2", "f3", "f4"])
    # the capped variant is the redundancy-meaningful one: ~2 blocks
    assert r["m_eff_capped"] == pytest.approx(2.0, abs=0.15)
    # published Li-Ji is lenient below rho=1: between blocks (2) and columns (4)
    assert 2.0 < r["m_eff_li_ji"] <= 4.0
    # Nyholt over ALL M eigenvalues (ddof=1): near-duplicate pairs -> ~3
    # (reviewer-verified value for the perfect-duplicate limit)
    assert r["m_eff_nyholt"] == pytest.approx(3.0, abs=0.2)
    assert r["n_sparse_pairs"] == 0


# --------------------------------------------------------------------------- #
# emax_null / deflated_t
# --------------------------------------------------------------------------- #

def test_emax_null_monotonic():
    assert emax_null(1) == 0.0
    vals = [emax_null(n) for n in (2, 5, 10, 25)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # true E[max of 10 iid N(0,1)] ~= 1.5388; the BLdP two-term approximation
    # should land near it (sqrt(2 ln N) = 2.15 is only the loose asymptote)
    assert emax_null(10) == pytest.approx(1.5388, abs=0.08)


def test_deflated_t_margin():
    d = deflated_t(3.0, 10)
    assert d["margin"] == pytest.approx(3.0 - d["emax_null"])
    assert 0.5 < d["prob_exceeds_null_max"] < 1.0


# --------------------------------------------------------------------------- #
# rank_autocorr: slow-factor warning
# --------------------------------------------------------------------------- #

def test_rank_autocorr_flags_slow_factor():
    slow = rank_autocorr(_panel(persistence=0.98), "factor")
    fresh = rank_autocorr(_panel(persistence=0.0), "factor")
    assert slow["warn_slow"] and slow["mean_rank_autocorr"] > 0.9
    assert not fresh["warn_slow"] and abs(fresh["mean_rank_autocorr"]) < 0.15


# --------------------------------------------------------------------------- #
# sector_neutralize + moved _xs_rank_z identity
# --------------------------------------------------------------------------- #

def test_sector_neutralize_zero_group_means():
    pm = _panel().query("month == month.min()")
    out = sector_neutralize(pm, ["factor"], "sector")
    assert out.groupby("sector")["factor"].mean().abs().max() < 1e-12


def test_xs_rank_z_moved_function_is_identical():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "fo", "scripts/factor_orthogonality.py")
    fo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fo)
    # the script must use the moved function object itself (bit-identity gate)
    assert fo._xs_rank_z is _xs_rank_z


# --------------------------------------------------------------------------- #
# Review hardening (correctness review 2026-07-11)
# --------------------------------------------------------------------------- #

def test_deflated_t_two_sided_ceiling():
    # |t| selection must use the two-sided ceiling E[max |Z|] = emax_null(2N)
    d2 = deflated_t(2.4, 25)                      # default two_sided
    d1 = deflated_t(2.4, 25, two_sided=False)
    assert d2["emax_null"] == pytest.approx(emax_null(50))
    assert d2["emax_null"] > d1["emax_null"]      # stricter than one-sided
    # calibration: at t exactly the ceiling, exceedance prob = 0.5
    e = emax_null(50)
    assert deflated_t(e, 25)["prob_exceeds_null_max"] == pytest.approx(0.5)


def test_ic_summary_override_cannot_defeat_guardrail():
    pm = _panel(ic=0.06)
    ic = monthly_ic(pm, "factor")
    s = ic_summary(ic, horizon_days=21, sampling_interval_days=5, hac_lags=0)
    assert s["naive_suppressed"] and s["t_naive"] is None
    assert s["hac_lags"] >= 5      # floored at the overlap minimum (F6)


def test_quantile_ls_exact_hand_built():
    # 3 months, 100 names, quintiles of 20; top set rotates by exactly 5
    # names per month -> one-way turnover 0.25; constant spread 1% -> exact
    # break-even from the stated convention drag = 2c(f_top+f_bot).
    months = pd.period_range("2020-01", periods=3, freq="M")
    rows = []
    for k, mo in enumerate(months):
        names = [f"T{i:03d}" for i in range(100)]
        fac = np.arange(100, dtype=float)
        fac = np.roll(fac, -5 * k)          # rotate ranks by 5 each month
        ret = np.where(fac >= 80, 0.01, np.where(fac < 20, 0.0, 0.005))
        rows.append(pd.DataFrame({"month": mo, "ticker": names,
                                  "factor": fac, "fwd_return_21d": ret}))
    pm = pd.concat(rows, ignore_index=True)
    r = quantile_ls(pm, "factor", min_names=50)
    assert r["mean_monthly"] == pytest.approx(0.01)
    assert r["turnover_top"] == pytest.approx(0.25)
    assert r["turnover_bottom"] == pytest.approx(0.25)
    assert r["break_even_cost_bps"] == pytest.approx(
        1e4 * 0.01 / (2 * 0.5), rel=1e-9)   # = 100 bps
    assert r["tie_months"] == 0


def test_quantile_ls_tie_warning_fires():
    # 95%-zero factor: tied mass straddles every boundary
    rng = np.random.RandomState(0)
    months = pd.period_range("2020-01", periods=6, freq="M")
    rows = []
    for mo in months:
        fac = np.zeros(200)
        fac[:10] = rng.normal(size=10)
        rows.append(pd.DataFrame({"month": mo,
                                  "ticker": [f"T{i}" for i in range(200)],
                                  "factor": fac,
                                  "fwd_return_21d": rng.normal(0, 0.05, 200)}))
    pm = pd.concat(rows, ignore_index=True)
    r = quantile_ls(pm, "factor", min_names=50)
    assert r["tie_months"] == 6


def test_split_halves_datetimeindex_boundary():
    # month-end-stamped DatetimeIndex: "2019-12" must still include Dec-2019
    idx = pd.date_range("2012-01-31", "2020-12-31", freq="ME")
    ic = pd.Series(0.0, index=idx)
    ic[idx <= "2019-12-31"] = 1.0
    r = split_halves(ic, "2019-12")
    assert r["first_n"] == 96 and r["second_n"] == 12


def test_ic_decay_runs_and_lags_scale():
    from alpha_graph.eval.ic_tools import ic_decay
    pm = _panel(ic=0.06, n_months=40, n_names=80)
    # synthetic prices consistent with the panel dates: random walk
    rng = np.random.RandomState(5)
    px = []
    for tk in [f"T{i:03d}" for i in range(80)]:
        dates = pd.period_range("2012-01", periods=40, freq="M").to_timestamp()
        px.append(pd.DataFrame({"ticker": tk, "date": dates,
                                "close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 40)))}))
    prices = pd.concat(px, ignore_index=True)
    pm2 = pm.copy()
    pm2["date"] = pm2["month"].dt.to_timestamp()
    out = ic_decay(pm2, "factor", prices, horizons=(1, 3), 
                   sampling_interval_days=1, min_names=30)
    assert len(out) == 2
    assert out.loc[out.horizon_days == 3, "hac_lags"].iloc[0] >= 3
