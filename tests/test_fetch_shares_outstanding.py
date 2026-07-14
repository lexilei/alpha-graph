"""Synthetic test for the real-split signature detector in
scripts/fetch_shares_outstanding.py (loaded by path — scripts/ is not a
package; no network is touched at import).

The detector guards the >100x-off-median drop rule: it flags would-drop rows
on tickers whose raw history contains a persistent >=50x level break with
>=4 rows on both sides (a real split, e.g. EXE's 1-for-200 reverse split)
as opposed to lone scale mistags (PKG's x1000 rows).
"""

import importlib.util
from pathlib import Path

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
           / "fetch_shares_outstanding.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("fetch_shares_outstanding",
                                                  _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_split_signature_detector():
    has_split_signature = _load_script().has_split_signature
    # persistent 200x break with >=4 rows on both sides: reverse-split signature
    assert has_split_signature([2e9] * 8 + [1e7] * 8)
    # EXE's actual shape: long pre-split history, exactly 4 rows at the
    # post-split count, then a bankruptcy-reorg re-issue only 10x up — the
    # local windows must catch the 200x break that whole-side medians dilute
    exe_like = [6.5e8] * 41 + [1.95e9] * 4 + [9.78e6] * 4 + [9.79e7] * 23
    assert has_split_signature(exe_like)
    # lone x1000 mistag spike (PKG-style): a 4-row median never reaches it
    assert not has_split_signature([9e7] * 10 + [9e10] + [9e7] * 10)
    # ordinary split territory (7x, AAPL-scale) is far below the 50x bar
    assert not has_split_signature([7e8] * 8 + [1e8] * 8)
    # fewer than 4 rows on a side can never qualify
    assert not has_split_signature([2e9] * 3 + [1e7] * 3)
    # non-positive rows are ignored, not counted toward a side
    assert not has_split_signature([0.0] * 6 + [9e7] * 8)
