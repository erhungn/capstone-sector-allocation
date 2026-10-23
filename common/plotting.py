"""Figure constants shared across every plotting script.

Why this module exists
----------------------
Figures were written one analysis at a time, and two inconsistencies crept in
that only show up when the figures sit side by side in a report:

1. **Resolution.** Every script saved at 150 DPI except the one that had been
   explicitly asked for at 300. A report mixing the two looks like a mistake.
2. **Colour.** The four models were consistently coloured, but ``LLM sentiment``
   was drawn in Model A's blue in one figure and purple in another. A reader
   who has learned that blue means Model A is then misled.

Both are fixed by importing from here rather than restating a palette per file.

The palette is Tableau-10 derived: distinguishable in colour, and still
distinguishable in greyscale by luminance ordering, since capstone reports get
printed.
"""

from __future__ import annotations

__all__ = ["DPI", "MODEL_COLORS", "SERIES_COLORS", "color_for"]

# Every figure in the project. 300 DPI is the floor for print submission.
DPI = 300

# The 2x2 design. These four never change meaning anywhere in the project.
MODEL_COLORS = {
    "A": "#4C78A8",   # blue    -- Logistic, macro
    "B": "#72B7B2",   # teal    -- Logistic, + LLM
    "C": "#E45756",   # red     -- XGBoost, macro
    "D": "#F58518",   # orange  -- XGBoost, + LLM
}

# Benchmarks and standalone signals, keyed by the label used in the CSVs.
SERIES_COLORS = {
    "Model A": MODEL_COLORS["A"],
    "Model B": MODEL_COLORS["B"],
    "Model C": MODEL_COLORS["C"],
    "Model D": MODEL_COLORS["D"],
    "Equal weight": "#444444",
    "SPY buy & hold": "#2E7D32",
    "Momentum 12-1": "#8E44AD",
    # Purple, deliberately not Model A's blue: this is a raw signal, not a model.
    "LLM sentiment": "#9467BD",
}

# Neutral fills for histograms, null distributions and de-emphasised bars.
NEUTRAL = "#B0BEC5"
HIGHLIGHT = MODEL_COLORS["C"]


def color_for(label: str, default: str = "#777777") -> str:
    """Colour for a series label, tolerant of the suffixes the scripts add.

    ``ranking_metrics.py`` writes "Model C (xgb, macro)" and ``long_short.py``
    writes "LLM sentiment (L/S)"; both should get the colour their base name
    owns rather than falling through to grey.
    """
    if label in SERIES_COLORS:
        return SERIES_COLORS[label]
    for name, value in SERIES_COLORS.items():
        if label.startswith(name):
            return value
    if label in MODEL_COLORS:
        return MODEL_COLORS[label]
    return default
