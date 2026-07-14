"""Tests for the item-code extraction in scripts/build_filing_items.py
(loaded by path — scripts/ is not a package; synthetic text only, no corpus
or cache touched).

Pinned boundary (build 2026-07-13): an item code is credited ONLY when its
"Item<ws>#.##" header STARTS a line (`^\\s*item\\s+([1-9]\\.\\d{2})(?!\\d)`,
IGNORECASE | MULTILINE — C17 sue_pead's ITEM_202_RE token with the code
generalized and a line-start anchor added). Both directions are pinned here:
line-start headers match; the identical string mid-line (body-text
cross-references) does not.
"""

import importlib.util
from pathlib import Path

_SCRIPT = (Path(__file__).resolve().parents[1] / "scripts"
           / "build_filing_items.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("build_filing_items", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_script()
extract_items = _MOD.extract_items
diagnose_zero = _MOD.diagnose_zero


# --------------------------------------------------------------------------- #
# Multi-item extraction: sorted, deduped
# --------------------------------------------------------------------------- #

def test_multi_item_sorted_and_deduped():
    text = ("Item 9.01 Financial Statements and Exhibits\n"
            "Item 2.02 Results of Operations and Financial Condition\n"
            "some body text\n"
            "Item 2.02 Results of Operations and Financial Condition\n")
    assert extract_items(text) == ["2.02", "9.01"]     # sorted, repeat collapsed


def test_header_at_string_start():
    assert extract_items("Item 2.02 Results of Operations") == ["2.02"]


def test_empty_text():
    assert extract_items("") == []


# --------------------------------------------------------------------------- #
# Header variants that MUST match
# --------------------------------------------------------------------------- #

def test_header_variants():
    assert extract_items("ITEM 2.02. RESULTS OF OPERATIONS\n") == ["2.02"]
    assert extract_items("Item 5.02(b) Departure of Directors\n") == ["5.02"]
    assert extract_items("Item\xa07.01 Regulation FD Disclosure\n") == ["7.01"]
    assert extract_items("\t Item 8.01 Other Events\n") == ["8.01"]
    assert extract_items("body line\r\nItem 1.01 Entry into a Material "
                         "Definitive Agreement\r\n") == ["1.01"]
    # code glued to the following title text is still a valid code boundary
    assert extract_items("Item 1.03Bankruptcy or Receivership\n") == ["1.03"]


# --------------------------------------------------------------------------- #
# The line-start pin, both ways
# --------------------------------------------------------------------------- #

def test_line_start_pin_both_ways():
    header = "Item 2.02 Results of Operations and Financial Condition"
    assert extract_items(header + "\n") == ["2.02"]                # at line start
    assert extract_items("as described in " + header + "\n") == []  # mid-line
    assert extract_items("pursuant to Item 2.02 above, the Company...\n") == []
    assert extract_items("the disclosures required by Item 1.01 relating to "
                         "the Transaction.\n") == []


def test_combined_single_line_header_takes_leading_code_only():
    # Accepted cost of the positional rule: a second code on the SAME line
    # ("Item 2.02 ... and Item 7.01 ...") is indistinguishable from a body
    # reference and is NOT credited.
    text = ("Item 2.02 Results of Operations and Financial Condition and "
            "Item 7.01 Regulation FD Disclosure\n")
    assert extract_items(text) == ["2.02"]


# --------------------------------------------------------------------------- #
# Non-8-K item shapes that must NOT match
# --------------------------------------------------------------------------- #

def test_no_hit_on_regulation_sk_item_numbers():
    assert extract_items("Item 601 of Regulation S-K requires exhibits.\n") == []
    assert extract_items("Item 601(b)(32) certification attached.\n") == []


def test_no_hit_on_pre_2004_numbering():
    assert extract_items("Item 5. Other Events\n") == []
    assert extract_items("Item 7. Financial Statements and Exhibits\n") == []


def test_numeric_boundaries():
    assert extract_items("Item 2.021 malformed code\n") == []      # 3-digit tail
    assert extract_items("Item 10.01 no such 8-K section\n") == []  # 2-digit lead
    assert extract_items("Item 0.01 no section zero\n") == []


def test_glued_header_out_of_family():
    # "Item8.01" (no whitespace) is outside the C17 token family `item\s+` —
    # pinned unmatched; the builder counts it in the zero-item diagnosis.
    assert extract_items("Item8.01 Other Events\n") == []
    assert diagnose_zero("Item8.01 Other Events\n") == "glued_header"


def test_zero_item_diagnosis_categories():
    assert diagnose_zero("see Item 2.02 above\n") == "midline_mention_only"
    assert diagnose_zero("Item       Financial\n\nExhibits\n") == "item_word_no_code"
    assert diagnose_zero("cover page only\n") == "no_item_text"
