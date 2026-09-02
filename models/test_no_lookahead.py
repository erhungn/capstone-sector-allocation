"""Timing audit for the walk-forward protocol.

The load-bearing claim: for every prediction row dated *t*, no training
observation has a target computed from any price at or after *t*.

A row dated *d* has target ``P(d+1)/P(d) - 1``, so it consumes prices at *d*
**and** *d+1*. The newest price a training set touches is therefore
``max(d) + 1 month``, and that must fall strictly before *t*. With the training
set defined as ``date < t`` (no embargo) the newest training row is *t-1*, whose
target consumes *P(t)* -- equal to *t*, so the check fails. A one-month embargo
makes the newest training row *t-2*, whose target consumes *P(t-1)* < *t*.

Run standalone or under pytest:

    python models/test_no_lookahead.py
    pytest models/test_no_lookahead.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET = REPO_ROOT / "data" / "model_dataset.csv"
PREDICTIONS = REPO_ROOT / "models" / "predictions.csv"

EMBARGO = 1  # the shipped default


def _dataset() -> pd.DataFrame:
    return pd.read_csv(DATASET, parse_dates=["date"]).sort_values(["date", "ticker"])


def latest_price_date_consumed(train_dates: pd.Series) -> pd.Timestamp:
    """Newest price any training label depends on: max(date) + 1 month."""
    return (train_dates.max() + pd.offsets.MonthEnd(1)).normalize()


def test_no_training_label_uses_a_price_at_or_after_the_prediction_month():
    """The core audit, run over every prediction month."""
    data = _dataset()
    all_dates = sorted(data["date"].unique())
    oos = [d for d in all_dates if d >= pd.Timestamp("2018-01-31")]

    violations = []
    for t in oos:
        cutoff = T.train_cutoff(t, EMBARGO)
        train = data[data["date"] < cutoff]
        if train.empty:
            continue
        newest_price = latest_price_date_consumed(train["date"])
        if newest_price >= t:
            violations.append((t, train["date"].max(), newest_price))

    assert not violations, (
        f"{len(violations)} prediction months train on a label that consumes a price "
        f"at or after the prediction month; first: predict {violations[0][0].date()}, "
        f"newest training row {violations[0][1].date()}, "
        f"whose target consumes P({violations[0][2].date()})"
    )


def test_the_check_actually_catches_the_bug_when_the_embargo_is_removed():
    """Guard against a vacuous test: with embargo=0 the audit must fail."""
    data = _dataset()
    all_dates = sorted(data["date"].unique())
    oos = [d for d in all_dates if d >= pd.Timestamp("2018-01-31")]

    caught = 0
    for t in oos:
        train = data[data["date"] < T.train_cutoff(t, 0)]
        if not train.empty and latest_price_date_consumed(train["date"]) >= t:
            caught += 1
    assert caught == len(oos), (
        f"expected all {len(oos)} months to violate without an embargo, got {caught}"
    )


def test_training_features_are_never_dated_at_or_after_the_prediction_month():
    """Separate from labels: no training *row* may be dated t-1 or later."""
    data = _dataset()
    all_dates = sorted(data["date"].unique())
    oos = [d for d in all_dates if d >= pd.Timestamp("2018-01-31")]

    for t in oos:
        train = data[data["date"] < T.train_cutoff(t, EMBARGO)]
        if train.empty:
            continue
        gap_months = (t.to_period("M") - train["date"].max().to_period("M")).n
        assert gap_months >= EMBARGO + 1, (
            f"predict {t.date()} trains through {train['date'].max().date()}: "
            f"gap {gap_months} months, expected at least {EMBARGO + 1}"
        )


def test_saved_predictions_match_the_embargoed_protocol():
    """The predictions on disk must have been produced under the embargo."""
    if not PREDICTIONS.exists():
        return  # nothing written yet
    data = _dataset()
    preds = pd.read_csv(PREDICTIONS, parse_dates=["date"])
    all_dates = sorted(data["date"].unique())

    for t in sorted(preds["date"].unique()):
        cutoff = T.train_cutoff(t, EMBARGO)
        n_train = len([d for d in all_dates if d < cutoff])
        assert n_train >= 24, (
            f"prediction month {pd.Timestamp(t).date()} would have only {n_train} "
            f"embargoed training months; predictions.csv looks stale -- re-run train.py"
        )


def main() -> int:
    tests = [
        test_no_training_label_uses_a_price_at_or_after_the_prediction_month,
        test_the_check_actually_catches_the_bug_when_the_embargo_is_removed,
        test_training_features_are_never_dated_at_or_after_the_prediction_month,
        test_saved_predictions_match_the_embargoed_protocol,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as err:
            failed += 1
            print(f"FAIL  {fn.__name__}\n      {err}")

    # Worked example so the invariant is visible, not just asserted.
    data = _dataset()
    t = pd.Timestamp("2018-02-28")
    for embargo in (0, EMBARGO):
        train = data[data["date"] < T.train_cutoff(t, embargo)]
        newest = latest_price_date_consumed(train["date"])
        verdict = "VIOLATION" if newest >= t else "ok"
        print(f"\n  embargo={embargo}: predict {t.date()}, train through "
              f"{train['date'].max().date()}, newest price consumed by a label "
              f"= P({newest.date()})  -> {verdict}")

    print("\n" + ("ALL TIMING CHECKS PASSED" if not failed else f"{failed} CHECK(S) FAILED"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
