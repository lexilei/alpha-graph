"""Research Coordinator — combines signals from all analyst agents.

This is the final node in the pipeline. It takes the accumulated signals
from the Filing Analyst, Earnings Call Analyst, and News Synthesizer,
and produces a single combined recommendation with confidence-weighted scoring.

Supports both static weights and data-driven weight optimization via
scipy.optimize.minimize to maximize Information Coefficient (IC).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import optimize, stats

from alpha_graph.agents.state import PipelineState
from alpha_graph.config import CACHE_DIR


# Default weights — can be overridden by optimize_weights()
AGENT_WEIGHTS = {
    "filing_analyst": 0.40,  # highest weight — Lazy Prices is well-documented
    "earnings_analyst": 0.35,  # strong academic backing
    "news_synthesizer": 0.25,  # event-driven, high impact but sporadic
}

# Path to store optimized weights
OPTIMIZED_WEIGHTS_PATH = CACHE_DIR / "optimized_agent_weights.json"


def _load_optimized_weights() -> dict[str, float] | None:
    """Load optimized weights from cache if available."""
    if OPTIMIZED_WEIGHTS_PATH.exists():
        try:
            data = json.loads(OPTIMIZED_WEIGHTS_PATH.read_text(encoding="utf-8"))
            logger.debug(f"Using optimized weights: {data['weights']}")
            return data["weights"]
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def get_active_weights() -> dict[str, float]:
    """Return the best available weights (optimized if available, else defaults)."""
    optimized = _load_optimized_weights()
    return optimized if optimized is not None else AGENT_WEIGHTS


def research_coordinator(state: PipelineState) -> dict:
    """Combine agent signals into a final recommendation.

    Uses confidence-weighted averaging:
        combined = sum(signal_i * confidence_i * weight_i) / sum(confidence_i * weight_i)

    Automatically uses optimized weights if available, falling back to defaults.
    """
    ticker = state["ticker"]
    signals = state.get("signals", [])

    if not signals:
        logger.warning(f"[{ticker}] Coordinator: no signals received")
        return {
            "recommendation": "HOLD",
            "combined_score": 0.0,
            "combined_confidence": 0.0,
            "combined_rationale": "No analyst signals available.",
        }

    active_weights = get_active_weights()

    # Confidence-weighted combination
    weighted_sum = 0.0
    weight_total = 0.0
    rationale_parts = []

    for sig in signals:
        source = sig["source"]
        w = active_weights.get(source, 0.2)
        weighted_sum += sig["signal"] * sig["confidence"] * w
        weight_total += sig["confidence"] * w
        rationale_parts.append(
            f"[{source}] signal={sig['signal']:+.2f} conf={sig['confidence']:.2f}: "
            f"{sig['rationale']}"
        )

    combined_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    combined_score = max(-1.0, min(1.0, combined_score))

    # Combined confidence: weighted average of individual confidences,
    # boosted if agents agree, penalized if they disagree
    avg_confidence = sum(s["confidence"] for s in signals) / len(signals)
    signal_values = [s["signal"] for s in signals]
    agreement = 1.0 - (max(signal_values) - min(signal_values)) / 2.0 if len(signal_values) > 1 else 0.5
    combined_confidence = avg_confidence * (0.5 + 0.5 * agreement)
    combined_confidence = max(0.0, min(1.0, combined_confidence))

    # Map score to recommendation
    if combined_score > 0.25 and combined_confidence > 0.3:
        recommendation = "BUY"
    elif combined_score < -0.25 and combined_confidence > 0.3:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"

    combined_rationale = (
        f"{recommendation} (score={combined_score:+.3f}, confidence={combined_confidence:.3f})\n"
        + "\n".join(rationale_parts)
    )

    logger.info(
        f"[{ticker}] Coordinator: {recommendation} "
        f"score={combined_score:+.3f} confidence={combined_confidence:.3f} "
        f"({len(signals)} signals)"
    )

    return {
        "recommendation": recommendation,
        "combined_score": combined_score,
        "combined_confidence": combined_confidence,
        "combined_rationale": combined_rationale,
    }


# ---------------------------------------------------------------------------
# Weight Optimization
# ---------------------------------------------------------------------------

def optimize_weights(
    signals_path: str | Path | None = None,
    market_path: str | Path | None = None,
) -> dict[str, float]:
    """Optimize agent weights to maximize Information Coefficient (IC).

    Uses historical pipeline signals correlated with forward returns.
    Optimization: scipy.optimize.minimize with negative IC as objective
    (we minimize negative IC = maximize IC).

    Constraints:
    - All weights in [0.05, 0.80] (no agent completely ignored or dominant)
    - Weights sum to 1.0

    Args:
        signals_path: path to pipeline signals with per-agent signals
        market_path: path to market data with forward returns

    Returns:
        Dict of optimized agent weights.
    """
    # Load data
    if signals_path is None:
        signals_path = CACHE_DIR / "pipeline_signals.parquet"
    if market_path is None:
        market_path = CACHE_DIR / "market_data.parquet"

    signals_path = Path(signals_path)
    market_path = Path(market_path)

    if not signals_path.exists():
        logger.error(f"No pipeline signals at {signals_path}. Run pipeline first.")
        return AGENT_WEIGHTS
    if not market_path.exists():
        logger.error(f"No market data at {market_path}. Download market data first.")
        return AGENT_WEIGHTS

    signals_df = pd.read_parquet(signals_path)
    market_df = pd.read_parquet(market_path)

    # Get forward returns
    market_df["date"] = pd.to_datetime(market_df["date"])
    latest_returns = (
        market_df.dropna(subset=["ret_21d"])
        .sort_values("date")
        .groupby("ticker")
        .tail(1)[["ticker", "ret_21d"]]
        .rename(columns={"ret_21d": "fwd_return"})
    )

    # Extract per-agent signals from rationale text (parse the stored format)
    agent_names = ["filing_analyst", "earnings_analyst", "news_synthesizer"]
    agent_signals = _extract_agent_signals(signals_df, agent_names)

    if agent_signals is None or agent_signals.empty:
        logger.warning("Could not extract per-agent signals. Using default weights.")
        return AGENT_WEIGHTS

    # Merge with returns
    merged = agent_signals.merge(latest_returns, on="ticker", how="inner")
    merged = merged.dropna(subset=["fwd_return"])

    if len(merged) < 20:
        logger.warning(f"Only {len(merged)} observations — too few for optimization.")
        return AGENT_WEIGHTS

    # Get active agent columns (those with non-zero variance)
    active_agents = []
    for agent in agent_names:
        col = f"signal_{agent}"
        if col in merged.columns and merged[col].std() > 0.01:
            active_agents.append(agent)

    if not active_agents:
        logger.warning("No active agents with signal variance. Using defaults.")
        return AGENT_WEIGHTS

    n = len(active_agents)
    signal_cols = [f"signal_{a}" for a in active_agents]

    def _neg_ic(weights: np.ndarray) -> float:
        """Negative IC (Spearman rank correlation) — objective to minimize."""
        combined = np.zeros(len(merged))
        for i, col in enumerate(signal_cols):
            combined += weights[i] * merged[col].values
        ic = stats.spearmanr(combined, merged["fwd_return"].values).statistic
        return -ic  # minimize negative IC = maximize IC

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    # Bounds: each weight in [0.05, 0.80]
    bounds = [(0.05, 0.80)] * n
    # Initial guess: equal weights
    w0 = np.ones(n) / n

    result = optimize.minimize(
        _neg_ic, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success:
        logger.warning(f"Optimization did not converge: {result.message}")

    optimized = {agent: float(round(w, 4)) for agent, w in zip(active_agents, result.x)}

    # Add zero weight for inactive agents
    for agent in agent_names:
        if agent not in optimized:
            optimized[agent] = 0.0

    # Compute IC with optimized weights
    combined_opt = np.zeros(len(merged))
    for agent, w in optimized.items():
        col = f"signal_{agent}"
        if col in merged.columns:
            combined_opt += w * merged[col].values
    opt_ic = stats.spearmanr(combined_opt, merged["fwd_return"].values).statistic

    # Compute IC with default weights
    combined_def = np.zeros(len(merged))
    for agent, w in AGENT_WEIGHTS.items():
        col = f"signal_{agent}"
        if col in merged.columns:
            combined_def += w * merged[col].values
    def_ic = stats.spearmanr(combined_def, merged["fwd_return"].values).statistic

    logger.info(
        f"Weight optimization complete:\n"
        f"  Default IC:    {def_ic:.4f}\n"
        f"  Optimized IC:  {opt_ic:.4f}\n"
        f"  Improvement:   {opt_ic - def_ic:+.4f}\n"
        f"  Weights: {optimized}"
    )

    # Save optimized weights
    OPTIMIZED_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        "weights": optimized,
        "default_ic": float(def_ic),
        "optimized_ic": float(opt_ic),
        "n_observations": len(merged),
        "active_agents": active_agents,
    }
    OPTIMIZED_WEIGHTS_PATH.write_text(json.dumps(save_data, indent=2), encoding="utf-8")
    logger.info(f"Saved optimized weights to {OPTIMIZED_WEIGHTS_PATH}")

    return optimized


def _extract_agent_signals(
    signals_df: pd.DataFrame,
    agent_names: list[str],
) -> pd.DataFrame | None:
    """Extract per-agent signals from the rationale column.

    The rationale format is:
    [source] signal=+0.35 conf=0.70: explanation text

    Falls back to using combined_score split evenly if parsing fails.
    """
    import re

    if "rationale" not in signals_df.columns:
        return None

    rows = []
    for _, row in signals_df.iterrows():
        rationale = str(row.get("rationale", ""))
        parsed = {"ticker": row["ticker"]}

        for agent in agent_names:
            pattern = rf"\[{agent}\]\s+signal=([+-]?\d+\.?\d*)\s+conf=(\d+\.?\d*)"
            match = re.search(pattern, rationale)
            if match:
                parsed[f"signal_{agent}"] = float(match.group(1))
                parsed[f"conf_{agent}"] = float(match.group(2))
            else:
                parsed[f"signal_{agent}"] = np.nan
                parsed[f"conf_{agent}"] = np.nan

        rows.append(parsed)

    df = pd.DataFrame(rows)
    # If no per-agent signals were parsed, fall back to combined score
    total_parsed = sum(df[f"signal_{a}"].notna().sum() for a in agent_names)
    if total_parsed == 0:
        logger.debug("No per-agent signals found in rationale — using combined_score only")
        return None

    return df
