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


def multivariate_ols(y: np.ndarray, X: np.ndarray, factor_names: list[str]) -> dict:
    """y = X @ beta + eps, where X's first column is ones (intercept = alpha).
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
    cov_beta = resid_var * XtX_inv
    se = np.sqrt(np.diag(cov_beta))
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    names = ["alpha"] + factor_names
    out = {
        "n": n,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "resid_std": resid_std,
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


def main():
    set_global_seeds()
    preds, _market, _regimes = _load_data()
    me_df, _ = reproduce_method_e(preds)
    me_df = me_df.sort_values("month").reset_index(drop=True)
    me_df["month_period"] = me_df["month"].dt.to_period("M")

    factors = load_ff5_plus_mom()

    aligned = me_df.merge(factors, on="month_period", how="inner")
    logger.info(f"Aligned {len(aligned)} months for FF attribution")
    if len(aligned) < 36:
        logger.error(f"Only {len(aligned)} aligned months — abort")
        return

    # Excess return = Method E net return minus risk-free rate
    aligned["ret_excess"] = aligned["ret_net"] - aligned["RF"]

    y = aligned["ret_excess"].values
    ones = np.ones(len(aligned))

    # Model 1: CAPM (Mkt-RF only)
    X1 = np.column_stack([ones, aligned["Mkt_RF"].values])
    fit1 = multivariate_ols(y, X1, ["Mkt_RF"])
    _print_regression(
        "FF1 (CAPM): r_E - RF = α + β_MKT·(Mkt-RF) + ε",
        fit1, ["Mkt_RF"],
    )

    # Model 2: FF3 (Mkt-RF, SMB, HML)
    X2 = np.column_stack([
        ones, aligned["Mkt_RF"].values, aligned["SMB"].values, aligned["HML"].values
    ])
    fit2 = multivariate_ols(y, X2, ["Mkt_RF", "SMB", "HML"])
    _print_regression(
        "FF3: add size (SMB) and value (HML)",
        fit2, ["Mkt_RF", "SMB", "HML"],
    )

    # Model 3: FF5 (Mkt-RF, SMB, HML, RMW, CMA)
    X3 = np.column_stack([
        ones,
        aligned["Mkt_RF"].values, aligned["SMB"].values, aligned["HML"].values,
        aligned["RMW"].values, aligned["CMA"].values,
    ])
    fit3 = multivariate_ols(y, X3, ["Mkt_RF", "SMB", "HML", "RMW", "CMA"])
    _print_regression(
        "FF5: add profitability (RMW) and investment (CMA)",
        fit3, ["Mkt_RF", "SMB", "HML", "RMW", "CMA"],
    )

    # Model 4: FF5 + Momentum (Carhart-style)
    X4 = np.column_stack([
        ones,
        aligned["Mkt_RF"].values, aligned["SMB"].values, aligned["HML"].values,
        aligned["RMW"].values, aligned["CMA"].values, aligned["MOM"].values,
    ])
    fit4 = multivariate_ols(y, X4, ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"])
    _print_regression(
        "FF5 + MOM: add Carhart momentum factor",
        fit4, ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"],
    )

    # ---- Verdict ----
    print()
    print("=" * 80)
    print("  VERDICT — DOES METHOD E ALPHA SURVIVE FACTOR CONTROLS?")
    print("=" * 80)
    for label, fit in [("CAPM", fit1), ("FF3", fit2), ("FF5", fit3), ("FF5+MOM", fit4)]:
        a_m = fit["coef"]["alpha"]
        t_a = fit["t"]["alpha"]
        rs = a_m / fit["resid_std"] * np.sqrt(12)
        status = "✓" if (t_a >= 2.0 and rs >= 0.3) else "⚠️"
        print(
            f"  {status} {label:<10s}  α(ann)={a_m*12*100:+6.2f}%  "
            f"t(α)={t_a:+5.2f}  resid Sharpe={rs:+5.3f}"
        )
    print("=" * 80)

    # Save summary
    rows = []
    for label, fit in [("CAPM", fit1), ("FF3", fit2), ("FF5", fit3), ("FF5+MOM", fit4)]:
        a_m = fit["coef"]["alpha"]
        row = {
            "model": label,
            "n_months": fit["n"],
            "r_squared": fit["r_squared"],
            "adj_r_squared": fit["adj_r_squared"],
            "resid_std": fit["resid_std"],
            "alpha_monthly": a_m,
            "alpha_annual": a_m * 12,
            "alpha_t_stat": fit["t"]["alpha"],
            "alpha_p_value": fit["p"]["alpha"],
            "residual_sharpe": a_m / fit["resid_std"] * np.sqrt(12),
        }
        for f in ["Mkt_RF", "SMB", "HML", "RMW", "CMA", "MOM"]:
            row[f"beta_{f}"] = fit["coef"].get(f, np.nan)
            row[f"t_{f}"] = fit["t"].get(f, np.nan)
        rows.append(row)
    out = pd.DataFrame(rows)
    path = CACHE_DIR / "method_e_ff_attribution.parquet"
    out.to_parquet(path, index=False)
    logger.info(f"Saved FF attribution to {path}")


if __name__ == "__main__":
    main()
