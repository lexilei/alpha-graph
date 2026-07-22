# poly-maker source review

2026-07-21. warproxxx/poly-maker, 1409 stars, ~5.4k lines, MIT. The one
open-source item in the Polymarket tool catalog. Read: strategy core
(quoting/estimators/regime), risk, TIPS.md (author's live-session notes).

## What it is

Maker-only **reward/rebate farmer for political markets** — not a crypto
bot, not an arb bot. Profit model = fixed daily liquidity-reward pools
(split by quoted share) + 20–25% maker rebates. Its own docs concede
spread capture is noise on fine-tick books.

Engineering is **not crude**: typed, tested, deterministic I/O-free quoting
core (built for replay), time-decayed EWMAs, markout-based toxicity that
widens spread after adverse fills, Avellaneda-style reservation price with
inventory skew, regime machine (QUIET/TRENDING/EVENT/HALTED/REDUCE_ONLY),
per-order reward-floor sizing (scoring is per order), heartbeat dead-man,
daily-loss kill switch. Better built than most retail quant code.

And still: **the author's supervised live session lost money** (−$15.51;
one adverse fill on a thin Romanian market cost $10), every parameter was
"fit by intuition on live money", and their #1 recommended next step is
record-the-WS-feed-and-replay — the infrastructure we already run.

## Structural weaknesses (= the exploitable surface)

1. **Fair value is book-internal**: microprice + 0.5·flow_z·tick. The bot
   knows nothing outside its own order book. On political markets no fast
   external reference exists, so it can only *react* (EVENT regime = vol
   spike detector) — by construction late to every news gap. Their Romania
   loss is this exact failure.
2. **Toxicity learns slowly by design**: EWMA of 5-min markouts,
   30-min half-life. It widens spread only after being run over repeatedly.
3. **Never improves the touch**: entries join best_bid or sit behind.
   Anyone quoting inside takes its queue position and reward share.
4. **Static per-profile parameters**, no per-market fitting, no backtest —
   admitted in TIPS.
5. **Economics cap**: reward pools are fixed and shared (diminishing
   returns), rebates only pay on fills (coupled to adverse selection).
   Thin-market seats are min-size farms earning dollars/day.

## What transfers to us

- The **reward mechanics** (rewardsMinSize, rewardsMaxSpread band, per-order
  scoring) — required knowledge for any maker strategy on this venue.
- The markout instrumentation pattern; the fee-rate ambiguity flag (their
  TIPS: "4% vs 0.4%, verify against the UI" — same open question we hit).
- Their wishlist (record L2, replay, fit fill-probability + adverse
  selection offline) is what our recorders are already collecting.

## The honest strategic read

Beating poly-maker is not the bar — its own author loses money at political
reward farming, which mostly tells us that seat is thin. The class weakness
of every bot in this catalog is the **information model**, not engineering.
Where an *external* fair value exists — crypto up/down windows, where
Binance spot leads the Chainlink resolution feed, and where our vol/flow
models already live — a maker quoting from a real FV has the informed seat
that this bot class structurally cannot occupy. The open question is
requote latency vs the snipers on the other side; that is precisely what
the recorder data will answer. Political reward farming: not our trade.
