"""Sector-level sentiment scores from FOMC communications, via the Claude API.

For each rebalance month, finds the most recent FOMC statement and minutes that
were *published* on or before that month-end, pairs them with the month's lagged
macro features, and asks Claude to score each of the 11 SPDR sector ETFs from
-1.0 (very bearish) to +1.0 (very bullish).

Outputs
-------
    data/llm_sentiment.csv   120 x 11 sentiment scores, one row per month-end
    logs/llm_calls.jsonl     one record per API call, for reproducibility

Usage
-----
    export ANTHROPIC_API_KEY=...
    python llm/sentiment_pipeline.py --dry-run     # 3 months, ~$0.05
    python llm/sentiment_pipeline.py               # full run
    python llm/sentiment_pipeline.py --start 2020-01 --end 2020-12

Prompt standardization
----------------------
SYSTEM_PROMPT and USER_TEMPLATE are frozen. Only three things vary between
calls: the statement text, the minutes text, and the macro block. The sector
list, the scale, the output schema, and the constraints are identical in every
period, so scores are comparable across the full sample. The template is also
held fixed across *retries* -- a repair instruction appended after a parse
failure would make that period's prompt differ from every other period's.

Look-ahead bias
---------------
Document selection uses ``statement_date`` and ``minutes_date`` from
fomc_calendar.csv -- publication dates, never meeting dates. A document is
eligible for month *t* only if it was public on or before the last day of *t*,
so the score for month *t* rests only on text an investor could have read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))

import universe as U  # noqa: E402

# --------------------------------------------------------------------------
# Configuration -- treat everything in this block as frozen
# --------------------------------------------------------------------------

MODEL = "claude-haiku-4-5-20251001"
TEMPERATURE = 0
# 4096 rather than 2000: headroom only. Observed responses run 700-900 output
# tokens, so this was never the binding constraint -- max_tokens is a cap, not a
# charge, so the extra ceiling costs nothing.
MAX_TOKENS = 4096
MAX_ATTEMPTS = 3

# Claude Haiku 4.5 list price, USD per million tokens.
PRICE_IN_PER_MTOK = 1.00
PRICE_OUT_PER_MTOK = 5.00

# Set by configure() before any call is made. They start at the main
# specification so importing this module and reading the prompt gives the
# eleven-asset version that produced the published scores.
SECTORS: list[str] = list(U.EQUITY_SECTORS)
SECTOR_NAMES = dict(U.ASSET_NAMES)

SYSTEM_PROMPT = (
    "You are a senior macro strategist analyzing Federal Reserve communications "
    "for sector-level investment implications."
)

# Frozen user-message template. The only substitutions are the three {} fields.
USER_TEMPLATE = """\
## FOMC statement (published {statement_date})

{statement_text}

## FOMC minutes (published {minutes_date})

{minutes_text}

## Macroeconomic context as of {month}

All values are lagged so they were observable at the start of the month.

{macro_block}

## Task

Rate each of the {asset_count} {asset_plural} below on how this FOMC communication
affects that {asset_noun}'s outlook over the coming month.

{sector_list}

Use a sentiment scale from -1.0 to +1.0:
  -1.0  very bearish for the sector
   0.0  neutral
  +1.0  very bullish for the sector

Judge each {asset_noun} on the transmission channel from this policy stance to that
{asset_noun}'s earnings, funding costs, and demand. Do not give every {asset_noun} the same
score unless the communication genuinely is neutral across all of them.

## Output format

Respond with strict JSON matching exactly this schema, and nothing else:

{{"sectors": {{"XLK": {{"sentiment": 0.0, "rationale": "..."}}, ...}}}}

Constraints:
- All {asset_count} {asset_noun}s listed above must be present, using those exact ticker keys.
- "sentiment" must be a number between -1.0 and 1.0 inclusive.
- "rationale" must be at most 2 sentences.
- Output raw JSON only. No markdown fences, no commentary before or after.
"""

SECTOR_LIST_BLOCK = "\n".join(f"- {t}: {SECTOR_NAMES[t]}" for t in SECTORS)

# Wording that has to agree with the universe. The 11-asset values reproduce the
# original prompt character for character, so the published scores remain the
# output of a prompt this file can still render.
PROMPT_WORDS = {
    "11asset": {"asset_count": 11, "asset_plural": "SPDR sector ETFs", "asset_noun": "sector"},
    "14asset": {"asset_count": 14, "asset_plural": "assets", "asset_noun": "asset"},
}
UNIVERSE = U.DEFAULT_UNIVERSE

# Enforced server-side via output_config.format. This makes malformed JSON,
# missing sectors, unexpected sectors, and wrong-typed scores impossible at the
# API level rather than something to detect afterwards.
#
# What the schema CANNOT express (the API rejects these keywords): numeric
# bounds and string length. So the [-1.0, 1.0] range check and the 2-sentence
# rationale limit stay in parse_and_validate() and the prompt respectively --
# the schema narrows the failure surface, it does not replace validation.
def build_schema(assets: list[str]) -> dict[str, Any]:
    """The server-side output schema for one universe."""
    return {
        "type": "object",
        "properties": {
            "sectors": {
                "type": "object",
                "properties": {
                    ticker: {
                        "type": "object",
                        "properties": {
                            "sentiment": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["sentiment", "rationale"],
                        "additionalProperties": False,
                    }
                    for ticker in assets
                },
                "required": list(assets),
                "additionalProperties": False,
            }
        },
        "required": ["sectors"],
        "additionalProperties": False,
    }


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "sectors": {
            "type": "object",
            "properties": {
                ticker: {
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "number"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["sentiment", "rationale"],
                    "additionalProperties": False,
                }
                for ticker in SECTORS
            },
            "required": list(SECTORS),
            "additionalProperties": False,
        }
    },
    "required": ["sectors"],
    "additionalProperties": False,
}

# Separator written by fomc_scraper.py between the provenance header and body.
HEADER_SEPARATOR = "-" * 70

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
FOMC_DIR = DATA_DIR / "fomc"
CALENDAR_CSV = FOMC_DIR / "fomc_calendar.csv"
MACRO_CSV = DATA_DIR / "macro_features.csv"
OUT_CSV = DATA_DIR / "llm_sentiment.csv"


def configure(universe: str) -> None:
    """Point the module's prompt, schema and output path at one universe.

    Mutating module globals is not elegant, but the alternative is threading a
    universe argument through every function that mentions a ticker, and this
    script's value is that its prompt is *frozen* -- the fewer moving parts
    between the template and the API call, the easier that is to verify. The
    switch happens once, before any request is built.
    """
    global SECTORS, SECTOR_LIST_BLOCK, OUTPUT_SCHEMA, OUT_CSV, UNIVERSE
    UNIVERSE = universe
    SECTORS = U.tickers(universe)
    SECTOR_LIST_BLOCK = "\n".join(f"- {t}: {SECTOR_NAMES[t]}" for t in SECTORS)
    OUTPUT_SCHEMA = build_schema(SECTORS)
    OUT_CSV = U.path_for(DATA_DIR / "llm_sentiment.csv", universe)
LOG_JSONL = REPO_ROOT / "logs" / "llm_calls.jsonl"

log = logging.getLogger("sentiment")


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def strip_header(raw: str) -> str:
    """Drop the provenance header fomc_scraper.py writes above the separator.

    The header carries meeting/publication dates. Sending it would leak the
    document's own metadata into the prompt and vary the prefix between calls
    for no analytical gain, so only the body is sent to the API.
    """
    idx = raw.find(HEADER_SEPARATOR)
    body = raw[idx + len(HEADER_SEPARATOR):] if idx != -1 else raw
    return body.strip()


def read_document(rel_path: str) -> str:
    path = FOMC_DIR / rel_path
    if not path.exists():
        raise FileNotFoundError(f"FOMC document missing: {path} (re-run llm/fomc_scraper.py)")
    text = strip_header(path.read_text(encoding="utf-8"))
    if not text:
        raise ValueError(f"FOMC document is empty after header strip: {path}")
    return text


def load_calendar() -> pd.DataFrame:
    df = pd.read_csv(CALENDAR_CSV, parse_dates=["meeting_date", "statement_date", "minutes_date"])
    return df.sort_values("meeting_date").reset_index(drop=True)


def load_macro() -> pd.DataFrame:
    df = pd.read_csv(MACRO_CSV, index_col=0, parse_dates=True)
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df


def latest_available(
    calendar: pd.DataFrame, date_col: str, file_col: str, month_end: pd.Timestamp
) -> tuple[str, pd.Timestamp] | tuple[None, None]:
    """Most recently *published* document of one kind as of ``month_end``.

    Selection is on the publication date, never the meeting date -- that is the
    whole point of the calendar's separate columns.
    """
    avail = calendar[
        calendar[date_col].notna()
        & (calendar[date_col] <= month_end)
        & calendar[file_col].astype(str).str.len().gt(0)
    ]
    if avail.empty:
        return None, None
    # Ties broken by meeting_date so the shared-minutes rows (two calendar rows
    # pointing at one document) resolve to the same file deterministically.
    row = avail.sort_values([date_col, "meeting_date"]).iloc[-1]
    return str(row[file_col]), row[date_col]


def format_macro_block(row: pd.Series) -> str:
    """Fixed-order, fixed-format macro block.

    Column order and number formatting are held constant so the only thing that
    changes between periods is the numbers themselves.
    """
    lines = []
    for col in row.index:
        val = row[col]
        lines.append(f"- {col}: {'n/a' if pd.isna(val) else f'{val:.2f}'}")
    return "\n".join(lines)


def build_user_prompt(
    month_end: pd.Timestamp,
    statement_text: str,
    statement_date: pd.Timestamp,
    minutes_text: str | None,
    minutes_date: pd.Timestamp | None,
    macro_row: pd.Series,
) -> str:
    return USER_TEMPLATE.format(
        statement_date=statement_date.date(),
        statement_text=statement_text,
        minutes_date=minutes_date.date() if minutes_date is not None else "not yet published",
        minutes_text=minutes_text if minutes_text else "(No FOMC minutes published as of this date.)",
        month=month_end.date(),
        macro_block=format_macro_block(macro_row),
        sector_list=SECTOR_LIST_BLOCK,
        **PROMPT_WORDS[UNIVERSE],
    )


# --------------------------------------------------------------------------
# Response handling
# --------------------------------------------------------------------------

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)


class ValidationError(ValueError):
    """Response did not satisfy the output contract."""


def close_delimiters(text: str) -> str:
    """Close any unterminated string, object, or array at the end of ``text``.

    Scans character by character tracking string state and backslash escapes, so
    a brace inside a rationale ("cut rates by 25bp {sic}") is not mistaken for
    structure -- a naive ``count('{') - count('}')`` gets that wrong.
    """
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack and ((ch == "}" and stack[-1] == "{") or (ch == "]" and stack[-1] == "[")):
                stack.pop()

    out = text + ('"' if in_string else "")
    out = re.sub(r",\s*$", "", out)  # a dangling comma is never valid
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def repair_json(cleaned: str) -> tuple[Any, bool]:
    """Best-effort parse of a malformed JSON object. Returns (payload, repaired).

    **This should never fire now that output_config enforces OUTPUT_SCHEMA
    server-side.** It is retained as a canary rather than a mechanism: a
    ``"json_repaired": true`` record in the call log means the API returned
    something the schema should have prevented, which is worth knowing about.
    It also still covers the two cases the schema cannot -- a response truncated
    at max_tokens, and a safety refusal -- though both are caught by stop_reason
    before parsing.

    Tries, in order: the text as-is; the text with delimiters closed; then the
    text trimmed back to each earlier complete object and re-closed. The last
    step recovers a genuinely truncated response by discarding the incomplete
    trailing entry -- schema validation downstream then rejects it for the
    missing sector rather than accepting a half-parsed record.

    Raises ValidationError if nothing parses.
    """
    attempts: list[str] = [cleaned]
    closed = close_delimiters(cleaned)
    if closed != cleaned:
        attempts.append(closed)

    idx = len(cleaned)
    for _ in range(30):
        idx = cleaned.rfind("}", 0, idx)
        if idx == -1:
            break
        attempts.append(close_delimiters(cleaned[: idx + 1]))

    first_error: Exception | None = None
    for i, candidate in enumerate(attempts):
        try:
            return json.loads(candidate), i > 0
        except json.JSONDecodeError as err:
            if first_error is None:
                first_error = err
    raise ValidationError(f"invalid JSON, unrepairable: {first_error}")


def parse_and_validate(text: str) -> tuple[dict[str, dict[str, Any]], bool]:
    """Parse the model's JSON and enforce the schema in the prompt.

    Returns (scores, repaired). ``repaired`` is True when the raw response did
    not parse and ``repair_json`` had to close delimiters -- it is recorded in
    the call log so months whose JSON needed fixing are identifiable later.

    Raises ValidationError with a specific reason so the log records *why* a
    retry happened, not just that one did.
    """
    cleaned = FENCE_RE.sub("", text).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1:
        raise ValidationError("no JSON object found in response")
    # rfind may be -1 on a response with no closing brace at all; repair handles it.
    candidate = cleaned[start: end + 1] if end > start else cleaned[start:]
    payload, repaired = repair_json(candidate)

    if not isinstance(payload, dict) or "sectors" not in payload:
        raise ValidationError("missing top-level 'sectors' key")
    sectors = payload["sectors"]
    if not isinstance(sectors, dict):
        raise ValidationError("'sectors' is not an object")

    missing = [t for t in SECTORS if t not in sectors]
    if missing:
        raise ValidationError(f"missing sectors: {missing}")
    extra = [t for t in sectors if t not in SECTORS]
    if extra:
        raise ValidationError(f"unexpected sectors: {extra}")

    out: dict[str, dict[str, Any]] = {}
    for ticker in SECTORS:
        entry = sectors[ticker]
        if not isinstance(entry, dict) or "sentiment" not in entry:
            raise ValidationError(f"{ticker}: missing 'sentiment'")
        raw = entry["sentiment"]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValidationError(f"{ticker}: sentiment is not a number ({raw!r})")
        score = float(raw)
        if not -1.0 <= score <= 1.0:
            raise ValidationError(f"{ticker}: sentiment {score} outside [-1.0, 1.0]")
        out[ticker] = {"sentiment": score, "rationale": str(entry.get("rationale", "")).strip()}
    return out, repaired


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------

def call_claude(client, user_prompt: str) -> tuple[Any, float]:
    """One Messages API call, with the output schema enforced server-side.

    ``output_config.format`` constrains generation to OUTPUT_SCHEMA, so the
    response is guaranteed to be a JSON object of the right shape. This is a
    request parameter, not prompt text -- USER_TEMPLATE is unchanged, so prompt
    standardization across periods is unaffected.

    The first call with a new schema pays a one-time compilation cost; after
    that the schema is cached for 24 hours.
    """
    started = time.monotonic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    return response, time.monotonic() - started


def response_text(response) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def call_cost(usage) -> float:
    return (
        usage.input_tokens / 1e6 * PRICE_IN_PER_MTOK
        + usage.output_tokens / 1e6 * PRICE_OUT_PER_MTOK
    )


def append_log(record: dict[str, Any]) -> None:
    LOG_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with LOG_JSONL.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def score_month(
    client,
    month_end: pd.Timestamp,
    user_prompt: str,
    statement_file: str,
    minutes_file: str | None,
) -> tuple[dict[str, dict[str, Any]], float]:
    """Call the API and validate, retrying up to MAX_ATTEMPTS.

    The prompt is byte-identical on every attempt -- see the module docstring on
    prompt standardization.
    """
    import anthropic

    month = month_end.strftime("%Y-%m")
    spend = 0.0
    last_reason = "unknown"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "month": month,
            "attempt": attempt,
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "statement_file": statement_file,
            "minutes_file": minutes_file,
            "prompt_chars": len(user_prompt),
            "system_prompt_chars": len(SYSTEM_PROMPT),
            "structured_output": True,
        }
        try:
            response, latency = call_claude(client, user_prompt)
        except anthropic.APIStatusError as err:
            record.update({"ok": False, "error": f"{type(err).__name__}: {err}"})
            append_log(record)
            log.warning("  %s attempt %d: API error %s", month, attempt, type(err).__name__)
            last_reason = str(err)
            time.sleep(2 ** attempt)
            continue
        except anthropic.APIConnectionError as err:
            record.update({"ok": False, "error": f"connection: {err}"})
            append_log(record)
            log.warning("  %s attempt %d: connection error", month, attempt)
            last_reason = str(err)
            time.sleep(2 ** attempt)
            continue

        raw = response_text(response)
        cost = call_cost(response.usage)
        spend += cost
        record.update({
            "latency_s": round(latency, 3),
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "cost_usd": round(cost, 6),
            "stop_reason": response.stop_reason,
            "request_id": getattr(response, "_request_id", None),
            "raw_response": raw,
        })

        # The schema holds only for a normally-completed turn. A truncated or
        # refused response can still violate it, so check before parsing.
        if response.stop_reason == "max_tokens":
            record.update({"ok": False, "error": "hit max_tokens; response truncated"})
            append_log(record)
            log.warning("  %s attempt %d: truncated at max_tokens", month, attempt)
            last_reason = "truncated at max_tokens"
            continue

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            reason = f"model refused (category={getattr(details, 'category', None)})"
            record.update({"ok": False, "error": reason})
            append_log(record)
            log.warning("  %s attempt %d: %s", month, attempt, reason)
            last_reason = reason
            continue

        try:
            scores, repaired = parse_and_validate(raw)
        except ValidationError as err:
            record.update({"ok": False, "error": f"validation: {err}"})
            append_log(record)
            log.warning("  %s attempt %d: %s", month, attempt, err)
            last_reason = str(err)
            continue

        record.update({
            "ok": True,
            "json_repaired": repaired,
            "parsed_scores": {t: scores[t]["sentiment"] for t in SECTORS},
            "rationales": {t: scores[t]["rationale"] for t in SECTORS},
        })
        append_log(record)
        if repaired:
            log.warning("  %s: response JSON was malformed and repaired before parsing", month)
        log.info(
            "  %s ok: %d in / %d out tokens, %.2fs, $%.4f",
            month, response.usage.input_tokens, response.usage.output_tokens, latency, cost,
        )
        return scores, spend

    raise RuntimeError(
        f"{month}: failed after {MAX_ATTEMPTS} attempts (last: {last_reason}). "
        f"Partial results were saved; re-run to resume."
    )


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def month_end(period: str) -> pd.Timestamp:
    return pd.Period(period, freq="M").to_timestamp(how="end").normalize()


# Rough English prose ratio. Only used for the pre-run estimate; the logged
# cost_usd figures come from the API's own token counts.
CHARS_PER_TOKEN = 3.6
EST_OUTPUT_TOKENS = 900


def estimate_cost(calendar: pd.DataFrame, macro: pd.DataFrame, months: list[pd.Timestamp]) -> float:
    """Ballpark spend before committing to a run, so the bill is never a surprise."""
    chars = 0
    for m in months:
        sf, _ = latest_available(calendar, "statement_date", "statement_file", m)
        mf, _ = latest_available(calendar, "minutes_date", "minutes_file", m)
        if sf is None:
            continue
        try:
            chars += len(read_document(sf)) + (len(read_document(mf)) if mf else 0)
        except (FileNotFoundError, ValueError):
            return float("nan")
    return (
        chars / CHARS_PER_TOKEN / 1e6 * PRICE_IN_PER_MTOK
        + len(months) * EST_OUTPUT_TOKENS / 1e6 * PRICE_OUT_PER_MTOK
    )


def run(start_str: str, end_str: str, dry_run: bool, force: bool,
        universe: str = U.DEFAULT_UNIVERSE) -> int:
    configure(universe)
    log.info("universe %r: %d assets -- %s", universe, len(SECTORS), ", ".join(SECTORS))
    log.info("output: %s", OUT_CSV)

    for path in (CALENDAR_CSV, MACRO_CSV):
        if not path.exists():
            log.error("%s not found -- run the upstream script first", path)
            return 1

    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        log.error(
            "ANTHROPIC_API_KEY is not set. Get a key at https://console.anthropic.com "
            "and run: export ANTHROPIC_API_KEY=..."
        )
        return 1

    import anthropic

    calendar = load_calendar()
    macro = load_macro()

    start, end = month_end(start_str), month_end(end_str)
    months = [d for d in macro.index if start <= d <= end]
    if not months:
        log.error("No months in macro_features.csv between %s and %s", start.date(), end.date())
        return 1
    if dry_run:
        months = months[:3]
        log.info("DRY RUN: first %d months only", len(months))

    # Resume: keep months already scored unless --force. An interrupted run
    # should not have to re-pay for work it already did.
    existing = pd.DataFrame(columns=SECTORS, dtype=float)
    if OUT_CSV.exists() and not force:
        existing = pd.read_csv(OUT_CSV, index_col=0, parse_dates=True)
        existing.index = pd.DatetimeIndex(existing.index).normalize()
        done = [m for m in months if m in existing.index]
        if done:
            log.info("Resuming: %d of %d months already scored (--force to redo)",
                     len(done), len(months))

    client = anthropic.Anthropic()
    rows: dict[pd.Timestamp, dict[str, float]] = {
        d: existing.loc[d, SECTORS].to_dict() for d in existing.index if d in months
    }

    prev_docs: tuple[str, str | None] | None = None
    prev_sentiments: dict[str, float] | None = None
    n_calls = n_carried = 0
    total_cost = 0.0
    started = time.monotonic()

    todo = [m for m in months if m not in rows]
    log.info(
        "%d months to score (%d already done). Rough cost at ~%.0f chars/token: $%.2f",
        len(todo), len(months) - len(todo), 3.6, estimate_cost(calendar, macro, todo),
    )

    try:
        for i, m in enumerate(months, 1):
            stmt_file, stmt_date = latest_available(calendar, "statement_date", "statement_file", m)
            min_file, min_date = latest_available(calendar, "minutes_date", "minutes_file", m)

            if stmt_file is None:
                log.error("%s: no FOMC statement published on or before this month-end", m.date())
                return 1

            docs = (stmt_file, min_file)
            tag = f"[{i:3d}/{len(months)}] {m.date()}"

            if m in rows:
                # Already scored on an earlier run. Seed the carry-forward state
                # from it so a resumed run doesn't re-pay for the next month.
                log.info("%s cached", tag)
                prev_docs, prev_sentiments = docs, dict(rows[m])
                continue

            # Carry forward when neither document has changed since last month.
            if prev_docs == docs and prev_sentiments is not None:
                rows[m] = dict(prev_sentiments)
                n_carried += 1
                append_log({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "month": m.strftime("%Y-%m"),
                    "carried_forward": True,
                    "statement_file": stmt_file,
                    "minutes_file": min_file,
                    "parsed_scores": rows[m],
                })
                log.info("%s carried forward (no new FOMC document)", tag)
                continue

            statement_text = read_document(stmt_file)
            minutes_text = read_document(min_file) if min_file else None
            prompt = build_user_prompt(
                m, statement_text, stmt_date, minutes_text, min_date, macro.loc[m]
            )
            log.info("%s calling (%d chars: %s + %s)", tag, len(prompt),
                     stmt_file, min_file or "no minutes")

            scores, spend = score_month(client, m, prompt, stmt_file, min_file)
            rows[m] = {t: scores[t]["sentiment"] for t in SECTORS}
            total_cost += spend
            n_calls += 1
            prev_docs, prev_sentiments = docs, dict(rows[m])
    finally:
        # Always persist whatever was completed, so a failure is resumable.
        if rows:
            out = pd.DataFrame.from_dict(rows, orient="index")[SECTORS].sort_index()
            out.index.name = "date"
            if not existing.empty and not force:
                keep = existing[~existing.index.isin(out.index)]
                out = pd.concat([out, keep[SECTORS]]).sort_index()
            OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(OUT_CSV, float_format="%.4f")
            log.info("Wrote %s (%d rows x %d columns)", OUT_CSV, *out.shape)

    log.info(
        "Done in %.1fs: %d API calls, %d carried forward, $%.4f total",
        time.monotonic() - started, n_calls, n_carried, total_cost,
    )
    log.info("Call log: %s", LOG_JSONL)

    result = pd.read_csv(OUT_CSV, index_col=0, parse_dates=True)
    if not dry_run:
        target = [d for d in macro.index if start <= d <= end]
        missing = [d for d in target if d not in pd.DatetimeIndex(result.index).normalize()]
        if missing:
            log.warning("%d months still unscored: %s", len(missing), [str(d.date()) for d in missing[:5]])
        elif not pd.DatetimeIndex(result.index).normalize().equals(macro.index):
            log.warning("Index does not match macro_features.csv exactly")
        else:
            log.info("Index matches macro_features.csv (%d rows)", len(result))

    n_bad = int(((result < -1.0) | (result > 1.0)).sum().sum())
    if n_bad:
        log.error("%d scores outside [-1, 1] made it to disk", n_bad)
        return 1
    log.info("Range check passed: all scores within [-1.0, 1.0]")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2016-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2025-12", help="last month, YYYY-MM")
    parser.add_argument("--dry-run", action="store_true", help="process only the first 3 months")
    parser.add_argument("--force", action="store_true", help="rescore months already in the CSV")
    parser.add_argument("--universe", default=U.DEFAULT_UNIVERSE, choices=sorted(U.UNIVERSES),
                        help="asset universe to score (default %(default)s)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    return run(args.start, args.end, args.dry_run, args.force, args.universe)


if __name__ == "__main__":
    sys.exit(main())
