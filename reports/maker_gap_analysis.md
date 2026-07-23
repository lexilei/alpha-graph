# Maker gap analysis: our kalshi_demo_maker vs poly-maker's engineering

2026-07-23. poly-maker's strategy was rejected (book-internal FV, political
reward farming); its engineering was flagged as the transferable asset. Now
that our maker is heading toward a $50 live test, this is the line-by-line
comparison. Their side read from source (risk/manager.py, execution/
{gateway,reconciler,ratelimit}.py, userstream/, 16 test files); ours is
kalshi_demo_maker.py after the 2026-07-23 review fixes.

## Where we are AHEAD (keep)

| item | ours | theirs |
|---|---|---|
| Fair value | external (Coinbase spot = BRTI constituent, settlement-aligned) | book-internal microprice — their structural weakness |
| Width | closed-form safe width from vol × cancel latency | reward-band clamp + heuristic constants |
| Dead-man | per-order self-expiry (120s) — survives even total process death; renewed at 90s | exchange heartbeat — needs the venue to support it (Kalshi has no such API), and a hung-but-alive process keeps heartbeating |
| Surface area | ~400 lines, one file — right size for the stage | 5.4k lines — right size for their maturity |

## Where we are BEHIND (ranked by adoption priority)

1. **Fill-level telemetry + markout tracker — their smartest machinery,
   our biggest hole.** They ingest fills via a user WebSocket, log each
   with FV-at-fill, and score 5-min markouts into a toxicity EWMA that
   widens spread/cuts size. We poll positions every 10s, never record
   individual fills, and therefore measure NOTHING about adverse
   selection — the quantity that decides whether a maker lives. Adopt
   before live: per-fill event log (price, side, FV, sigma, book state)
   + markout evaluation job. Without this, the $50 test produces a P&L
   number but no diagnosis.
2. **Tests on the pure math.** They have 16 test files (quoting, regime,
   risk, hardening). We have zero — and our own history is the argument:
   the bracket-FV mispricing, the DST close-time bug, and the empty
   `my_orders` cancel-all would each have been caught by the tests they
   routinely write. Adopt: unit tests for fv (T + B markets), width,
   book clamping, side normalization, and the reconcile decision table.
3. **Error-rate circuit breaker.** They kill the engine when order-error
   rate ≥ threshold over ≥20 attempts — the "venue is rejecting me /
   I am malfunctioning" detector. Our loss stop only sees money, not
   malfunction. Cheap to add; catches failure modes money can't.
4. **Rate limiting with pressure signal.** Token bucket, self-limited
   below documented ceilings, with a pressure metric so the engine sheds
   low-value reprices first. We fire ~17 unthrottled requests per 10s
   cycle. Fine on demo; rude and risky in production.
5. **Regime machine (EVENT pull).** On sweeps/FV jumps they pull quotes
   entirely and cool off; on stale data they halt. Our width grows with
   sigma but we never step out of the pool, and a stale spot feed would
   not stop us. Partial adoption: pull-quotes when |Δspot| over one cycle
   exceeds k·sigma, halt when the spot fetch fails twice.
6. **Minimal-diff reconciler with size tolerance.** Theirs computes the
   smallest cancel/place set with reprice_ticks + resize_frac tolerances
   (pure function, tested). Ours cancels on a 2c price drift only —
   no size tolerance, I/O-entangled, untestable as written.
7. Config profiles, structured event logging, SQLite state — maturity
   items, not blockers.

## Verdict and sequencing

Before the $50 live test (gate unchanged): adopt #1 (fill log + markout)
and #2 (math tests). During the live-test week: #3, #4, #5. #6/#7 when the
maker graduates from one file. The strategic asymmetry stands — they built
excellent plumbing around a blind pricer; we built a sighted pricer with
adolescent plumbing. Plumbing is copyable; sight is not.
