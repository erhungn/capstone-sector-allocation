"""The asset universe, and the output paths that depend on which one is in use.

Two universes
-------------
``11asset``  the eleven SPDR equity sector ETFs. **The main specification.**
             Every published result in the report is this universe, and its
             output files keep their original unsuffixed names.

``14asset``  the same eleven plus TLT, IEF and GLD. An extension, run because
             FOMC communication bears on duration and on gold more directly
             than on equity sectors: if a monetary-policy signal exists at all,
             it should be easiest to find there. Outputs carry a ``_14asset``
             suffix so the main specification is never overwritten.

The universe used to be declared three times -- in ``data/collect.py``,
``data/build_dataset.py`` and ``llm/sentiment_pipeline.py`` -- which was already
a hazard when all three said eleven. Adding a second universe would have made it
three places to get wrong. It is declared once, here.

A note on what the extension is not
-----------------------------------
Adding bonds and gold changes the meaning of the cross-sectional target. The
target asks whether an asset beats the equal-weight average of the universe that
month, and TLT and IEF carry roughly a third of the monthly volatility of an
equity sector. A ranking on raw returns therefore partly ranks on volatility:
in an up month the low-volatility assets are nearly guaranteed to be in the
bottom half regardless of any forecast. That is a real property of the extended
universe, not a bug to be normalised away, and the comparison reports it rather
than hiding it -- see ``models/universe_comparison.py``.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "EQUITY_SECTORS", "EXTENDED_ASSETS", "UNIVERSES", "DEFAULT_UNIVERSE",
    "ALL_TICKERS", "ASSET_NAMES", "REFERENCE_TICKER",
    "tickers", "dummy_tickers", "interactions", "suffix", "path_for",
]

EQUITY_SECTORS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLV",   # Health Care
    "XLI",   # Industrials
    "XLY",   # Consumer Discretionary
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLB",   # Materials
    "XLRE",  # Real Estate         (inception 2015-10)
    "XLC",   # Communication Svcs  (inception 2018-06)
]

# All three long predate the 2016-01 sample start, so unlike XLC they introduce
# no leading gap. collect.py verifies this rather than trusting it.
EXTENDED_ASSETS = [
    "TLT",   # 20+ year Treasuries (inception 2002-07)
    "IEF",   # 7-10 year Treasuries (inception 2002-07)
    "GLD",   # Gold bullion         (inception 2004-11)
]

UNIVERSES = {
    "11asset": EQUITY_SECTORS,
    "14asset": EQUITY_SECTORS + EXTENDED_ASSETS,
}
DEFAULT_UNIVERSE = "11asset"

# Every ticker any universe can contain. collect.py downloads this superset once,
# so the price files serve both universes and the 11-asset results stay
# reproducible from the same raw data.
ALL_TICKERS = EQUITY_SECTORS + EXTENDED_ASSETS

# Descriptors given to the LLM. The three new ones name the exposure rather than
# the fund, because the model is being asked about duration and inflation
# sensitivity, not about an issuer.
ASSET_NAMES = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Health Care",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "TLT": "long-duration US Treasuries",
    "IEF": "intermediate-duration US Treasuries",
    "GLD": "gold",
}

# XLC stays the reference level in both universes. It is the youngest equity
# sector and absent from the panel until 2019-07, so making it the reference
# keeps every remaining dummy column fully populated -- and keeping the same
# reference across universes means the dummy coefficients stay comparable.
REFERENCE_TICKER = "XLC"

# Economically motivated macro x asset terms. A linear model cannot otherwise
# learn that energy and utilities respond to the same VIX reading in opposite
# directions: every asset sees an identical macro row each month, so one
# coefficient would have to serve all of them.
INTERACTIONS_EQUITY = [
    ("VIX_lag1", "XLE"),        # energy is high-beta to volatility
    ("VIX_lag1", "XLU"),        # utilities are defensive
    ("VIX_lag1", "XLF"),        # financials sensitive to volatility
    ("T10Y2Y_lag1", "XLF"),     # banks earn the curve
    ("T10Y2Y_lag1", "XLRE"),    # real estate is rate sensitive
    ("T10Y2Y_lag1", "XLU"),     # utilities trade as bond proxies
    ("WTI_lag1", "XLE"),        # energy tracks crude
    ("CPI_YOY_lag2", "XLP"),    # staples as an inflation hedge
]

# Added only for the extended universe, chosen on the same basis: each pairs an
# asset with the macro variable most directly driving it.
INTERACTIONS_EXTENDED = [
    ("T10Y2Y_lag1", "TLT"),     # long bonds are most sensitive to curve slope
    ("FEDFUNDS_lag1", "IEF"),   # intermediate bonds track the policy rate
    ("T10YIE_lag1", "GLD"),     # gold responds to inflation expectations
]


def _check(universe: str) -> str:
    if universe not in UNIVERSES:
        raise ValueError(f"unknown universe {universe!r}; expected one of "
                         f"{', '.join(UNIVERSES)}")
    return universe


def tickers(universe: str = DEFAULT_UNIVERSE) -> list[str]:
    """The assets in this universe, in a fixed order."""
    return list(UNIVERSES[_check(universe)])


def dummy_tickers(universe: str = DEFAULT_UNIVERSE) -> list[str]:
    """Assets that get a one-hot column: everything but the reference."""
    return [t for t in tickers(universe) if t != REFERENCE_TICKER]


def interactions(universe: str = DEFAULT_UNIVERSE) -> list[tuple[str, str]]:
    """Macro x asset terms, restricted to assets this universe contains."""
    present = set(tickers(universe))
    pairs = INTERACTIONS_EQUITY + INTERACTIONS_EXTENDED
    return [(macro, t) for macro, t in pairs if t in present]


def suffix(universe: str = DEFAULT_UNIVERSE) -> str:
    """Filename suffix for this universe's outputs.

    The main specification deliberately gets an empty suffix: its filenames are
    the ones the report, the README and every committed analysis already cite,
    and renaming them to make the scheme look symmetric would break all of that
    for no gain.
    """
    return "" if _check(universe) == DEFAULT_UNIVERSE else f"_{universe}"


def path_for(base: Path, universe: str = DEFAULT_UNIVERSE) -> Path:
    """``models/predictions.csv`` -> ``models/predictions_14asset.csv``."""
    s = suffix(universe)
    return base if not s else base.with_name(f"{base.stem}{s}{base.suffix}")
