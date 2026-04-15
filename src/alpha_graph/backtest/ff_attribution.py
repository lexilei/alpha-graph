"""Fama-French 5-factor + momentum attribution for Method E.

Extends `attribution.py` (SPY single-factor) to Fama-French 5-factor + Carhart
momentum. The question it answers: after controlling for market, size, value,
profitability, investment, and momentum exposure, how much alpha remains in
the long-only top10 Lazy Prices book?

Factors downloaded directly from Ken French's data library (no API key).

Usage:
    python -m alpha_graph.backtest.ff_attribution
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from alpha_graph.backtest.attribution import reproduce_method_e
from alpha_graph.backtest.extensions import _load_data
from alpha_graph.config import CACHE_DIR, set_global_seeds

FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip"
)


def _fetch_french_zip(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("latin-1")


def _parse_french_monthly(text: str, expected_cols: list[str]) -> pd.DataFrame:
    """Parse a Ken French monthly CSV. Files have a header blurb, then the
    monthly table, then (usually) an annual table. We take only the monthly
    block, which has YYYYMM integer dates.
    """
    lines = text.splitlines()
    # Find first row that looks like "YYYYMM, number, ..."
    start = None
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split(",")]
        if (
            len(parts) >= 2
            and parts[0].isdigit()
            and len(parts[0]) == 6
            and 190000 <= int(parts[0]) <= 210000
        ):
            start = i
            break
    if start is None:
        raise RuntimeError("Could not locate monthly data block in Ken French CSV")
    # Find end: first row whose first field is NOT 6-digit integer (annual block)
    end = len(lines)
    for i in range(start, len(lines)):
        parts = [p.strip() for p in lines[i].split(",")]
        if not (parts and parts[0].isdigit() and len(parts[0]) == 6):
            end = i
            break
    block = "\n".join(lines[start:end])
    df = pd.read_csv(
        io.StringIO(block),
        header=None,
        names=["yyyymm"] + expected_cols,
    )
    df["yyyymm"] = df["yyyymm"].astype(int).astype(str)
    df["month_period"] = pd.to_datetime(df["yyyymm"], format="%Y%m").dt.to_period("M")
    # Values are in percent — convert to decimal
    for c in expected_cols:
        df[c] = df[c].astype(float) / 100.0
    return df[["month_period"] + expected_cols]


def load_ff5_plus_mom() -> pd.DataFrame:
    logger.info("Fetching Fama-French 5-factor monthly from Ken French library")
    ff5_text = _fetch_french_zip(FF5_URL)
    ff5 = _parse_french_monthly(ff5_text, ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "RF"])

    logger.info("Fetching momentum factor monthly from Ken French library")
    mom_text = _fetch_french_zip(MOM_URL)
    mom = _parse_french_monthly(mom_text, ["MOM"])

    factors = ff5.merge(mom, on="month_period", how="inner")
    logger.info(f"Loaded {len(factors)} months of FF5+MOM factors")
    return factors


def _newey_west_lag(n: int) -> int:
    """Andrews (1991) automatic bandwidth, simplified to L = floor(4*(n/100)^(2/9)).
    For n=167 monthly returns this gives L=4, which is the standard choice in
    finance papers using monthly data.
    """
    return int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))


def multivariate_ols(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: list[str],
    hac_lags: int | None = None,
) -> dict:
    """y = X @ beta + eps, where X's first column is ones (intercept = alpha).

    If `hac_lags` is None, uses classical OLS standard errors (i.i.d. residuals).
    If a non-negative integer, uses Newey-West HAC standard errors with that
    Bartlett-window lag length, robust to heteroskedasticity and serial
    correlation (e.g. monthly portfolio returns are mildly autocorrelated;
    classical SE understates uncertainty by ~10-20%).

    Returns coefficients, se, t-stats, p-values, R², residual std.
    """
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    resid = y - y_hat
    sse = float((resid ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r_squared = 1 - sse / sst
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    resid_var = sse / (n - k)
    resid_std = float(np.sqrt(resid_var))

    if hac_lags is None:
        cov_beta = resid_var * XtX_inv
        se_label = "OLS"
    else:
        # Newey-West HAC: cov(β) = (X'X)^-1 · S · (X'X)^-1
        # where S = sum_h w_h (X_t' u_t u_{t-h} X_{t-h} + transpose)
        # and w_h = 1 - h/(L+1) is the Bartlett weight.
        u = resid.reshape(-1, 1)
        Xu = X * u  # n × k
        S = Xu.T @ Xu  # lag 0 contribution
        for h in range(1, hac_lags + 1):
            w = 1.0 - h / (hac_lags + 1.0)
            Gamma_h = Xu[h:].T @ Xu[:-h]
            S = S + w * (Gamma_h + Gamma_h.T)
        # Small-sample correction (Stata-style) to match standard reference impl
        S = S * n / (n - k)
        cov_beta = XtX_inv @ S @ XtX_inv
        se_label = f"NW({hac_lags})"

    se = np.sqrt(np.diag(cov_beta))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    names = ["alpha"] + factor_names
    out = {
        "n": n,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "resid_std": resid_std,
        "se_method": se_label,
        "hac_lags": hac_lags,
        "coef": dict(zip(names, beta)),
        "se": dict(zip(names, se)),
        "t": dict(zip(names, t)),
        "p": dict(zip(names, p)),
    }
    return out


def _print_regression(label: str, fit: dict, factors: list[str]) -> None:
    print()
    print("=" * 80)
    print(f"  {label}")
    print("=" * 80)
    print(f"  N months: {fit['n']}   R²: {fit['r_squared']:.4f}   adj R²: {fit['adj_r_squared']:.4f}")
    print(f"  Residual std (monthly): {fit['resid_std']*100:.3f}%")
    print()
    alpha_m = fit["coef"]["alpha"]
    print(f"  {'name':<10s}  {'coef':>9s}  {'se':>9s}  {'t':>7s}  {'p':>7s}")
    print(f"  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}")
    for name in ["alpha"] + factors:
        c = fit["coef"][name]
        s = fit["se"][name]
        t = fit["t"][name]
        p = fit["p"][name]
        if name == "alpha":
            print(f"  {name:<10s}  {c*100:>+8.4f}%  {s*100:>8.4f}%  {t:>+7.2f}  {p:>7.4f}")
        else:
            print(f"  {name:<10s}  {c:>+9.4f}  {s:>9.4f}  {t:>+7.2f}  {p:>7.4f}")
    alpha_ann = alpha_m * 12
    resid_sharpe = alpha_m / fit["resid_std"] * np.sqrt(12)
    print()
    print(f"  α (annualized, arithmetic): {alpha_ann*100:+.2f}%")
    print(f"  Residual (alpha) Sharpe:    {resid_sharpe:+.3f}")


def _build_method_e_returns(preds: pd.DataFrame) -> pd.DataFrame:
    me_df, _ = reproduce_method_e(preds)
    me_df = me_df.sort_values("month").reset_index(drop=True)
    me_df["month_period"] = me_df["month"].dt.to_period("M")
    return me_df


def _run_all_specs(
    aligned: pd.DataFrame,
    label_prefix: str,
    hac_lags: int | None,
) -> list[tuple[str, dict, list[str]]]:
    y = aligned["ret_excess"].values
    ones = np.ones(len(aligned))
    specs = [
        ("CAPM",     ["Mkt_RF"]),
        ("FF3",      ["Mkt_RF", "SMB", "HML"]),
        ("FF5",      ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]),
        ("FF5+MOM",  ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]),
    ]
    out = []
    for name, factors in specs:
        X = np.column_stack([ones] + [aligned[f].values for f in factors])
        fit = multivariate_ols(y, X, factors, hac_lags=hac_lags)
        _print_regression(f"{label_prefix} — {name} ({fit['se_method']} SE)", fit, factors)
        out.append((name, fit, factors))
    return out


def _print_verdict(rows: list[tuple[str, dict, list[str]]], label: str) -> None:
    print()
    print("=" * 80)
    print(f"  VERDICT — {label}")
    print("=" * 80)
    for name, fit, _ in rows:
        a_m = fit["coef"]["alpha"]
        t_a = fit["t"]["alpha"]
        rs = a_m / fit["resid_std"] * np.sqrt(12)
        status = "OK" if (t_a >= 2.0 and rs >= 0.3) else "WEAK"
        print(
            f"  [{status}] {name:<10s}  α(ann)={a_m*12*100:+6.2f}%  "
            f"t(α)={t_a:+5.2f} ({fit['se_method']})  resid Sharpe={rs:+5.3f}"
        )
    print("=" * 80)


def _rows_for_save(specs: list[tuple[str, dict, list[str]]], universe: str) -> list[dict]:
    rows = []
    for name, fit, _ in specs:
        a_m = fit["coef"]["alpha"]
        row = {
            "universe": universe,
            "model": name,
            "se_method": fit["se_method"],
            "hac_lags": fit["hac_lags"],
            "n_months": fit["n"],
            "r_squared": fit["r_squared"],
            "adj_r_squared": fit["adj_r_squared"],
            "resid_std": fit["resid_std"],
            "alpha_monthly": a_m,
            "alpha_annual": a_m * 12,
            "alpha_t_stat": fit["t"]["alpha"],
            "alpha_p_value": fit["p"]["alpha"],
            "alpha_se": fit["se"]["alpha"],
            "residual_sharpe": a_m / fit["resid_std"] * np.sqrt(12),
        }
        for f in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
            row[f"beta_{f}"] = fit["coef"].get(f, np.nan)
            row[f"t_{f}"] = fit["t"].get(f, np.nan)
        rows.append(row)
    return rows


def main():
    from alpha_graph.backtest.pit_universe import fetch_membership, filter_preds_pit

    set_global_seeds()
    preds, _market, _regimes = _load_data()
    factors = load_ff5_plus_mom()

    # ----- Full universe -----
    me_full = _build_method_e_returns(preds)
    aligned_full = me_full.merge(factors, on="month_period", how="inner")
    aligned_full["ret_excess"] = aligned_full["ret_net"] - aligned_full["RF"]
    n_full = len(aligned_full)
    L = _newey_west_lag(n_full)
    logger.info(f"Full universe: aligned {n_full} months, Newey-West lag L = {L}")

    print("\n\n##### FULL UNIVERSE — classical OLS SE #####")
    full_ols = _run_all_specs(aligned_full, "FULL", hac_lags=None)
    _print_verdict(full_ols, "FULL UNIVERSE — classical OLS SE")

    print("\n\n##### FULL UNIVERSE — Newey-West HAC SE #####")
    full_hac = _run_all_specs(aligned_full, "FULL", hac_lags=L)
    _print_verdict(full_hac, "FULL UNIVERSE — Newey-West HAC SE")

    # ----- PIT universe -----
    membership = fetch_membership()
    pit_preds = filter_preds_pit(preds, membership)
    me_pit = _build_method_e_returns(pit_preds)
    aligned_pit = me_pit.merge(factors, on="month_period", how="inner")
    aligned_pit["ret_excess"] = aligned_pit["ret_net"] - aligned_pit["RF"]
    n_pit = len(aligned_pit)
    L_pit = _newey_west_lag(n_pit)
    logger.info(f"PIT universe: aligned {n_pit} months, Newey-West lag L = {L_pit}")

    print("\n\n##### PIT UNIVERSE — classical OLS SE #####")
    pit_ols = _run_all_specs(aligned_pit, "PIT", hac_lags=None)
    _print_verdict(pit_ols, "PIT UNIVERSE — classical OLS SE")

    print("\n\n##### PIT UNIVERSE — Newey-West HAC SE #####")
    pit_hac = _run_all_specs(aligned_pit, "PIT", hac_lags=L_pit)
    _print_verdict(pit_hac, "PIT UNIVERSE — Newey-West HAC SE")

    # ----- Combined save -----
    all_rows = (
        _rows_for_save(full_ols, "full_OLS")
        + _rows_for_save(full_hac, "full_HAC")
        + _rows_for_save(pit_ols, "pit_OLS")
        + _rows_for_save(pit_hac, "pit_HAC")
    )
    out = pd.DataFrame(all_rows)
    path = CACHE_DIR / "method_e_ff_attribution.parquet"
    out.to_parquet(path, index=False)
    logger.info(f"Saved FF attribution (full+PIT × OLS+HAC) to {path}")


if __name__ == "__main__":
    main()
