# TODO

## Tomorrow (2026-04-09): deeper signal analysis via graphs

The docs are now honest about what the 14-year backtest says. The next step is **not** to chase a better Sharpe — it is to understand the signal we have at a deeper level by visualizing its decay and performance characteristics. Today's pass updated text only; tomorrow's pass should produce plots.

### Decay analysis (how stale does the signal get?)

- [ ] **Lazy Prices signal half-life.** For each 10-K filing, compute the IC of `cosine_similarity` against forward returns at horizons of 1, 5, 10, 21, 42, 63, 126, 252 trading days. Plot IC vs horizon. Hypothesis: the long-side IC decays slowly because "stable company" is a persistent state, but it should still have a measurable half-life. This will tell us whether monthly rebalancing is even the right cadence, or whether we should hold longer.
- [ ] **8-K event signal half-life.** Same exercise for `event_score`. Hypothesis: this should decay much faster (days to weeks) because 8-K events are point-in-time. If the half-life is shorter than 21 days, monthly rebalancing is leaving alpha on the table — we should be running the signal weekly or event-triggered.
- [ ] **Decay by item type.** Are 4.01 (auditor change) signals more or less persistent than 1.01 (material agreement) signals? Plot per-item-type IC decay curves. This is the diagnostic that tells us whether the hand-coded scores in `event_signal.py` are still calibrated correctly, or whether some item types have stopped working.
- [ ] **Signal autocorrelation.** For the ML combiner predicted return, plot the cross-sectional rank autocorrelation between consecutive months. If the signal is highly autocorrelated, turnover is wasted; if it's not, the signal is too noisy.

### Performance decomposition

- [ ] **Per-decile forward return curve over time.** The U-shape diagnostic in METHODOLOGY is currently a single static table. Plot it as a heatmap: x-axis = year, y-axis = decile, color = avg forward return. This will tell us whether the U-shape is stable across the full 14 years or whether it's a recent regime artifact (e.g. only post-2020).
- [ ] **Method E rolling Sharpe (12-month window).** Method E's headline Sharpe of 1.08 is averaged over 168 months. Plot the rolling 12-month Sharpe. Does it have any prolonged underwater periods? If yes, the strategy has hidden tail risk that the headline number conceals. If no, the alpha is structurally consistent.
- [ ] **Method E vs SPY rolling alpha.** Plot the rolling 24-month regression alpha (with t-stat band) over the full 14 years. The static t-stat is 4.35 — does the alpha look stable, or is it concentrated in a few years?
- [ ] **L/S baseline drawdown decomposition.** The L/S baseline has -58% max drawdown. Decompose: how much came from the long leg, how much from the short leg, how much from costs? Stacked bar chart per year.

### Holdings and concentration

- [ ] **Method E ticker turnover.** What fraction of the top10 changes month-to-month? The README mentions 392 unique tickers across 168 months, which implies turnover, but we don't have a turnover series. Plot it.
- [ ] **Method E sector composition over time.** Stacked area chart of GICS sector weights in the top10 across years. The Mag-7 share of 4.3% is encouraging but doesn't tell us whether the strategy quietly tilts into one sector at a time.

### Permutation null robustness

- [ ] **Re-run the permutation test on the 14-year panel.** The current `permutation_test_null.parquet` was generated against the Anti-Momentum extension on the 23-month subset, which is exactly the wrong baseline. Re-run it (a) on the full 14-year panel and (b) for each of the 10 methods in `improvements.py`. Plot real Sharpe vs null distribution per method. This will give us a one-shot graphical answer to "which methods are luck and which are signal."
- [ ] **Increase permutation count from 50 to 500.** 50 permutations gives a noisy null. If the answer to "is the L/S signal real?" is going to be definitive, we want at least 500 perms.

### Stretch (only if everything above is done)

- [ ] **Survivorship-bias check.** This is the biggest unmeasured caveat for Method E. As a first-pass cheap experiment, drop all tickers that joined the S&P 500 after 2015 and re-run Method E. If the alpha collapses, survivorship is the explanation. If it doesn't, the alpha is more robust than we feared. This isn't a true survivorship-free experiment but it's a quick directional test.

### What's deliberately NOT on this list

- ❌ Trying to get a better Sharpe by tweaking hyperparameters / adding features / picking a different cost assumption.
- ❌ Re-running the Anti-Momentum extension on a different subset hoping for a better number.
- ❌ "Fixing" the L/S strategy. The U-shape is a property of the data, not a bug we can engineer around without overfitting.
- ❌ Re-adding components that were removed in the 2026-04-15 cleanup (multi-agent LLM pipeline, paper-trading layer, knowledge graph spillover, LLM filing-change detector, fundamentals feature, transcripts pipeline). They produced no validated alpha.

## Done today (2026-04-08)

- [x] Deleted the old TODO.md (which was full of items premised on the strategy working).
- [x] Updated `README.md`, `METHODOLOGY.md`, `report/alpha_graph_report.tex`, `docs/alpha_graph.tex`, and root `CLAUDE.md` to be on the same page about: the 14-year L/S Sharpe of 0.32, the U-shape diagnostic, the Method E long-only result, and the formal retraction of the previous "Sharpe 1.77" and "Sharpe 1.91" headlines.
- [x] Did not touch core code. The retraction is documentation-only.
