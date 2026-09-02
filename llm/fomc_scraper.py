"""Scrape FOMC statements and minutes from federalreserve.gov.

Collects every FOMC policy statement and set of meeting minutes in a date
window and writes them as plain text, alongside a calendar CSV that records
when each document actually became public.

Outputs
-------
    data/fomc/statements/<statement_date>.txt   e.g. 2024-01-31.txt
    data/fomc/minutes/<meeting_date>.txt        e.g. 2024-01-31.txt
    data/fomc/fomc_calendar.csv

Usage
-----
    python llm/fomc_scraper.py
    python llm/fomc_scraper.py --start 2016-01 --end 2025-12 --delay 1.5
    python llm/fomc_scraper.py --force        # re-download existing files

Why publication dates matter
----------------------------
A statement is public the moment it is released (2:00 p.m. on the last day of
the meeting). Minutes are public roughly three weeks later. A backtest may only
condition on a document from the rebalance month *after* its publication date,
so ``statement_date`` and ``minutes_date`` -- not ``meeting_date`` -- are the
fields that gate usage. They are scraped from the source rather than assumed:
minutes release dates come from the "(Released ...)" text on the calendar page,
and statement dates from the ``article__time`` stamp on the release itself.

Page structure
--------------
The Fed uses two different layouts, both handled here:

* ``fomchistorical{YYYY}.htm`` -- older years, one ``div.panel`` per meeting
  whose heading carries the meeting date and type ("March 2 (unscheduled)").
* ``fomccalendars.htm`` -- recent years, one panel per year containing
  ``div.row.fomc-meeting`` rows with separate month/date cells.

Edge cases handled
------------------
* **Unscheduled meetings.** March 2020 held two (March 2 and March 15) on top
  of the regular calendar.
* **Meeting date != statement date.** The March 2, 2020 meeting released its
  statement on March 3. The scraper keeps them distinct.
* **Notation votes.** March 19/23/31 and August 27, 2020 produced statements
  with no meeting and no minutes.
* **Cancelled meetings.** The scheduled March 17-18, 2020 meeting never
  happened; it is skipped, having neither document.
* **Shared minutes.** The March 2, 2020 minutes are appended to the end of the
  March 15 document rather than published separately. That row points at the
  containing file and is flagged ``minutes_shared``.
* **Look-alike links.** ``monetary{date}a1.htm`` is the Implementation Note and
  ``monetary{date}b.htm`` is the Statement on Longer-Run Goals; neither is the
  policy statement, and both are excluded.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE = "https://www.federalreserve.gov"
CALENDAR_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"
HISTORICAL_URL = f"{BASE}/monetarypolicy/fomchistorical{{year}}.htm"

# Identify the scraper honestly; federalreserve.gov publishes no robots.txt.
USER_AGENT = (
    "capstone-research/1.0 (masters thesis, sector allocation; "
    "contact via repository owner) python-requests"
)

DEFAULT_DELAY = 1.5      # seconds between requests
DELAY_JITTER = 0.4       # +/- randomisation so requests are not metronomic
REQUEST_TIMEOUT = 45
MAX_RETRIES = 3

# The policy statement is always the bare 'a' suffix. 'a1' is the
# Implementation Note and 'b' is the Statement on Longer-Run Goals.
STATEMENT_RE = re.compile(r"/newsevents/pressreleases/monetary(\d{8})a\.htm$")
MINUTES_RE = re.compile(r"/monetarypolicy/fomcminutes(\d{8})\.htm$")

RELEASED_RE = re.compile(r"Released\s+([A-Za-z]+\.?\s+\d{1,2},\s+\d{4})", re.I)
# Panel headings, including the slash form used when a meeting straddles a
# month boundary:
#   "March 2 (unscheduled) Meeting - 2020"
#   "January 26-27 Meeting - 2021"
#   "Jan/Feb 31-1 Meeting - 2017"     <- Jan 31 to Feb 1
HEADING_RE = re.compile(
    r"^(?P<month>[A-Za-z]+(?:\.?\s*/\s*[A-Za-z]+)?)\.?\s+"
    r"(?P<days>\d{1,2}(?:\s*[-–]\s*\d{1,2})?)\s*"
    r"(?:\((?P<note>[^)]+)\))?",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "fomc"
STATEMENTS_DIR = OUT_DIR / "statements"
MINUTES_DIR = OUT_DIR / "minutes"
CALENDAR_CSV = OUT_DIR / "fomc_calendar.csv"

log = logging.getLogger("fomc_scraper")


@dataclass
class Meeting:
    """One row of the FOMC calendar."""

    meeting_date: date
    meeting_type: str = "scheduled"
    statement_url: str | None = None
    statement_date: date | None = None
    minutes_url: str | None = None
    minutes_date: date | None = None
    # Set when this meeting's minutes live inside another meeting's document.
    minutes_shared_with: date | None = None
    statement_file: str = ""
    minutes_file: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> date:
        return self.meeting_date


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------

class Fetcher:
    """Rate-limited GET with retries.

    The delay is applied *before* every request rather than after the previous
    one, so a retry cannot burst.
    """

    def __init__(self, delay: float = DEFAULT_DELAY):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.count = 0

    def _sleep(self) -> None:
        if self.count:  # no delay before the very first request
            time.sleep(max(0.0, self.delay + random.uniform(-DELAY_JITTER, DELAY_JITTER)))

    def get(self, url: str, allow_404: bool = False) -> str | None:
        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            self._sleep()
            self.count += 1
            try:
                resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
                if resp.status_code == 404 and allow_404:
                    log.debug("404 (expected possible): %s", url)
                    return None
                resp.raise_for_status()
                # federalreserve.gov serves UTF-8 but omits charset from the
                # Content-Type header, so requests falls back to ISO-8859-1 and
                # turns every en-dash into mojibake ("January 30<E2><80><93>31").
                # The charset is declared in the document's own <meta> tag.
                if "charset" not in resp.headers.get("Content-Type", "").lower():
                    resp.encoding = resp.apparent_encoding or "utf-8"
                return resp.text
            except Exception as err:  # noqa: BLE001 - network layer
                last_err = err
                log.warning("  fetch failed (%d/%d) %s: %s", attempt, MAX_RETRIES, url, err)
                if attempt < MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise RuntimeError(f"failed to fetch {url}") from last_err


# --------------------------------------------------------------------------
# Date helpers
# --------------------------------------------------------------------------

def parse_us_date(text: str) -> date | None:
    """'February 21, 2024' / 'Feb. 21, 2024' -> date."""
    cleaned = re.sub(r"\s+", " ", text).replace(".", "").strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


def month_number(token: str) -> int | None:
    token = token.replace(".", "").strip()
    for fmt in ("%B", "%b"):
        try:
            return datetime.strptime(token, fmt).month
        except ValueError:
            continue
    return None


def date_from_url(url: str, pattern: re.Pattern[str]) -> date | None:
    m = pattern.search(url)
    return datetime.strptime(m.group(1), "%Y%m%d").date() if m else None


def last_day_of_meeting(month_text: str, day_text: str, year: int) -> date | None:
    """Resolve a calendar row's month/day cells to the meeting's final day.

    Handles the cross-month form the Fed uses for meetings that straddle a
    month boundary: month 'Apr/May' with days '30-1' is 1 May. The trailing '*'
    that marks a meeting with projections is stripped.
    """
    months = [t.strip() for t in re.split(r"/", month_text.replace("\xa0", " ")) if t.strip()]
    days = [d for d in re.split(r"[-–]", re.sub(r"[^0-9\-–]", "", day_text)) if d]
    if not months or not days:
        return None
    mnum = month_number(months[-1])
    if mnum is None:
        return None

    # A Dec/Jan meeting ends in the following calendar year.
    first = month_number(months[0])
    if first is not None and first > mnum:
        year += 1

    try:
        return date(year, mnum, int(days[-1]))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Discovery: historical year pages (one panel per meeting)
# --------------------------------------------------------------------------

def _statement_anchor(scope) -> str | None:
    """The policy statement link, identified by its label rather than its URL.

    The URL suffix alone is not sufficient. On most days the policy statement
    takes the bare 'a' slot, but when it is the only release of the day another
    document can occupy it -- the 2025-08-22 "Statement on Longer-Run Goals and
    Monetary Policy Strategy" is published at ``monetary20250822a.htm``. Keying
    on the anchor text avoids mistaking the framework document, the
    Implementation Note, or a plain press release for a policy statement.
    """
    for a in scope.find_all("a", href=True):
        if not STATEMENT_RE.search(a["href"]):
            continue
        label = re.sub(r"\s+", " ", a.get_text(" ", strip=True))

        # Historical layout: the anchor is literally labelled "Statement".
        if label == "Statement":
            return BASE + a["href"]

        # Calendar layout: anchor labelled "HTML" inside the "Statement:" cell.
        if label.upper() == "HTML":
            node = a
            for _ in range(3):
                node = node.find_parent("div")
                if node is None:
                    break
                text = re.sub(r"\s+", " ", node.get_text(" ", strip=True))
                if text.startswith("Statement:"):
                    return BASE + a["href"]
    return None


def _minutes_anchor(scope) -> tuple[str | None, date | None]:
    """Minutes URL plus the release date printed alongside it."""
    for p in scope.find_all(["p", "div"]):
        a = p.find("a", href=MINUTES_RE)
        if not a:
            continue
        released = RELEASED_RE.search(re.sub(r"\s+", " ", p.get_text(" ", strip=True)))
        return BASE + a["href"], parse_us_date(released.group(1)) if released else None
    return None, None


def parse_historical_year(html: str, year: int) -> list[Meeting]:
    soup = BeautifulSoup(html, "html.parser")
    meetings: list[Meeting] = []

    for panel in soup.select("div.panel.panel-default"):
        head_el = panel.select_one(".panel-heading")
        if not head_el:
            continue
        heading = re.sub(r"\s+", " ", head_el.get_text(" ", strip=True))
        m = HEADING_RE.match(heading)
        if not m:
            log.debug("    unparsed heading: %r", heading)
            continue
        meeting_date = last_day_of_meeting(m.group("month"), m.group("days"), year)
        if meeting_date is None:
            log.warning("    could not resolve date from heading: %r", heading)
            continue

        note = (m.group("note") or "").lower()
        if "cancel" in note:
            log.info("    %s: cancelled meeting, no documents - skipped", heading)
            continue
        if "notation" in note:
            mtype = "notation_vote"
        elif "unscheduled" in note:
            mtype = "unscheduled"
        else:
            mtype = "scheduled"

        mt = Meeting(meeting_date=meeting_date, meeting_type=mtype)
        mt.statement_url = _statement_anchor(panel)
        mt.minutes_url, mt.minutes_date = _minutes_anchor(panel)

        # "Minutes: See end of minutes of March 15 meeting"
        panel_text = re.sub(r"\s+", " ", panel.get_text(" ", strip=True))
        if mt.minutes_url is None and "See end of minutes" in panel_text:
            # "See end of minutes of March 15 meeting"
            # "See end of minutes of October 29-30 meeting"  <- host keyed on the 30th
            ref = re.search(
                r"See end of minutes of ([A-Za-z]+)\s+(\d{1,2}(?:\s*[-–]\s*\d{1,2})?)",
                panel_text,
            )
            if ref:
                host = last_day_of_meeting(ref.group(1), ref.group(2), year)
                if host:
                    mt.minutes_shared_with = host
                    mt.notes.append(f"minutes appended to {host.isoformat()} document")
                else:
                    log.warning("    could not parse shared-minutes reference: %r", ref.group(0))

        if mt.statement_url is None and mt.minutes_url is None and mt.minutes_shared_with is None:
            log.debug("    %s: no documents, skipped", heading)
            continue
        meetings.append(mt)

    return meetings


# --------------------------------------------------------------------------
# Discovery: current calendar page (one panel per year)
# --------------------------------------------------------------------------

def parse_calendars(html: str) -> list[Meeting]:
    soup = BeautifulSoup(html, "html.parser")
    meetings: list[Meeting] = []

    for panel in soup.select("div.panel.panel-default"):
        head_el = panel.select_one(".panel-heading")
        if not head_el:
            continue
        ym = re.search(r"(20\d{2})", head_el.get_text(" ", strip=True))
        if not ym:
            continue
        year = int(ym.group(1))

        for row in panel.select("div.row.fomc-meeting"):
            month_el = row.select_one(".fomc-meeting__month")
            day_el = row.select_one(".fomc-meeting__date")
            if not month_el or not day_el:
                continue
            meeting_date = last_day_of_meeting(
                month_el.get_text(" ", strip=True), day_el.get_text(" ", strip=True), year
            )
            if meeting_date is None:
                continue

            raw = re.sub(r"\s+", " ", row.get_text(" ", strip=True)).lower()
            if "cancelled" in raw:
                continue
            if "notation vote" in raw:
                mtype = "notation_vote"
            elif "unscheduled" in raw:
                mtype = "unscheduled"
            else:
                mtype = "scheduled"

            mt = Meeting(meeting_date=meeting_date, meeting_type=mtype)
            mt.statement_url = _statement_anchor(row)
            mt.minutes_url, mt.minutes_date = _minutes_anchor(row)
            if mt.statement_url is None and mt.minutes_url is None:
                continue  # future meeting, nothing published yet
            meetings.append(mt)

    return meetings


def discover(fetch: Fetcher, start: date, end: date) -> list[Meeting]:
    """Merge both layouts, preferring whichever supplied a statement URL."""
    found: dict[date, Meeting] = {}

    log.info("Discovering meetings from %s", CALENDAR_URL)
    for mt in parse_calendars(fetch.get(CALENDAR_URL)):
        found[mt.key] = mt
    log.info("  calendar page yielded %d meetings", len(found))

    years_covered = {m.meeting_date.year for m in found.values()}
    for year in range(start.year, end.year + 1):
        if year in years_covered:
            continue
        url = HISTORICAL_URL.format(year=year)
        log.info("Discovering %d from %s", year, url)
        html = fetch.get(url, allow_404=True)
        if html is None:
            log.warning("  no historical page for %d", year)
            continue
        got = parse_historical_year(html, year)
        log.info("  %d yielded %d meetings", year, len(got))
        for mt in got:
            found.setdefault(mt.key, mt)

    selected = sorted(
        (m for m in found.values() if start <= m.meeting_date <= end),
        key=lambda m: m.meeting_date,
    )
    return selected


# --------------------------------------------------------------------------
# Text extraction
# --------------------------------------------------------------------------

BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote"]
NOISE_SELECTORS = "script, style, .share, .sharethis-wrapper, #share, .lastUpdate, noscript"


def extract_text(html: str) -> str:
    """Plain text of a Fed article, with paragraph breaks and no markup.

    Block elements are pulled individually and joined with blank lines. Using
    ``get_text()`` on the whole container would either run paragraphs together
    or, with a newline separator, split sentences at every inline <a> and <em>.
    """
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("#article") or soup.select_one("#content") or soup.body
    if article is None:
        return ""

    for el in article.select(NOISE_SELECTORS):
        el.decompose()

    # Soft hyphens are invisible but sit inside words, breaking string matching
    # and tokenisation downstream; non-breaking spaces should read as spaces.
    for node in article.find_all(string=True):
        cleaned = node.replace("\xad", "").replace("\xa0", " ")
        if cleaned != node:
            node.replace_with(cleaned)

    blocks: list[str] = []
    for el in article.find_all(BLOCK_TAGS):
        # Skip nested blocks; the ancestor already captured this text.
        if el.find_parent(BLOCK_TAGS) is not None:
            continue
        text = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        if text and text.lower() not in {"share", "pdf", "html"}:
            blocks.append(text)

    out: list[str] = []
    for b in blocks:  # drop consecutive duplicates (repeated nav labels)
        if not out or out[-1] != b:
            out.append(b)
    return "\n\n".join(out).strip()


def write_document(path: Path, header: dict[str, str], body: str) -> None:
    """Write a text file with a short, clearly delimited provenance header."""
    lines = [f"{k}: {v}" for k, v in header.items() if v]
    content = "\n".join(lines) + "\n" + "-" * 70 + "\n\n" + body + "\n"
    path.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def scrape(start: date, end: date, delay: float, force: bool) -> int:
    STATEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)

    fetch = Fetcher(delay=delay)
    meetings = discover(fetch, start, end)
    if not meetings:
        log.error("No meetings discovered - the page layout may have changed")
        return 1

    log.info("")
    log.info("%d meetings in %s..%s", len(meetings), start, end)
    by_year: dict[int, int] = {}
    for mt in meetings:
        by_year[mt.meeting_date.year] = by_year.get(mt.meeting_date.year, 0) + 1
    log.info("  per year: %s", ", ".join(f"{y}={n}" for y, n in sorted(by_year.items())))
    log.info("")

    n_stmt = n_min = 0
    for i, mt in enumerate(meetings, 1):
        tag = "" if mt.meeting_type == "scheduled" else f" [{mt.meeting_type}]"
        log.info("[%2d/%d] %s%s", i, len(meetings), mt.meeting_date, tag)

        # --- statement ---
        if mt.statement_url:
            url_date = date_from_url(mt.statement_url, STATEMENT_RE)
            html = fetch.get(mt.statement_url)
            soup = BeautifulSoup(html, "html.parser")
            stamp = soup.select_one("p.article__time") or soup.select_one(".article__time")
            # The stamp on the release itself is authoritative; the URL date is
            # the fallback. They differ when a meeting ran past its statement,
            # e.g. the 2020-03-02 meeting released on 2020-03-03.
            mt.statement_date = (parse_us_date(stamp.get_text(strip=True)) if stamp else None) or url_date
            body = extract_text(html)
            if not body:
                log.error("        empty statement body: %s", mt.statement_url)
                return 1
            fname = f"{mt.statement_date.isoformat()}.txt"
            path = STATEMENTS_DIR / fname
            if path.exists() and not force:
                log.info("        statement %s (cached)", fname)
            else:
                write_document(
                    path,
                    {
                        "document": "FOMC statement",
                        "meeting_date": mt.meeting_date.isoformat(),
                        "statement_date": mt.statement_date.isoformat(),
                        "meeting_type": mt.meeting_type,
                        "source": mt.statement_url,
                    },
                    body,
                )
                log.info("        statement %s (%d chars)", fname, len(body))
            mt.statement_file = f"statements/{fname}"
            n_stmt += 1
            if mt.statement_date != mt.meeting_date:
                mt.notes.append(f"statement released {mt.statement_date} (day after meeting)")
        else:
            log.info("        no statement")

        # --- minutes ---
        if mt.minutes_url:
            html = fetch.get(mt.minutes_url)
            body = extract_text(html)
            if not body:
                log.error("        empty minutes body: %s", mt.minutes_url)
                return 1
            fname = f"{mt.meeting_date.isoformat()}.txt"
            path = MINUTES_DIR / fname
            if path.exists() and not force:
                log.info("        minutes   %s (cached)", fname)
            else:
                write_document(
                    path,
                    {
                        "document": "FOMC minutes",
                        "meeting_date": mt.meeting_date.isoformat(),
                        "minutes_date": mt.minutes_date.isoformat() if mt.minutes_date else "",
                        "meeting_type": mt.meeting_type,
                        "source": mt.minutes_url,
                    },
                    body,
                )
                log.info("        minutes   %s (%d chars, released %s)",
                         fname, len(body), mt.minutes_date)
            mt.minutes_file = f"minutes/{fname}"
            n_min += 1
            if mt.minutes_date is None:
                log.warning("        minutes release date missing for %s", mt.meeting_date)
        elif mt.minutes_shared_with:
            log.info("        minutes   shared with %s document", mt.minutes_shared_with)
        else:
            log.info("        no minutes")

    # Resolve shared-minutes rows once every document is on disk.
    by_date = {m.meeting_date: m for m in meetings}
    for mt in meetings:
        if mt.minutes_shared_with and not mt.minutes_file:
            host = by_date.get(mt.minutes_shared_with)
            if host and host.minutes_file:
                mt.minutes_file = host.minutes_file
                mt.minutes_date = host.minutes_date
                log.info("Linked %s minutes -> %s (appended to that document)",
                         mt.meeting_date, host.minutes_file)
            else:
                log.warning("Could not resolve shared minutes for %s", mt.meeting_date)

    # --- calendar CSV ---
    with CALENDAR_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "meeting_date", "statement_date", "minutes_date",
            "statement_file", "minutes_file", "meeting_type", "notes",
        ])
        for mt in meetings:
            w.writerow([
                mt.meeting_date.isoformat(),
                mt.statement_date.isoformat() if mt.statement_date else "",
                mt.minutes_date.isoformat() if mt.minutes_date else "",
                mt.statement_file,
                mt.minutes_file,
                mt.meeting_type,
                "; ".join(mt.notes),
            ])

    log.info("")
    log.info("Wrote %s (%d rows)", CALENDAR_CSV, len(meetings))
    log.info("  %d statements, %d minutes documents, %d HTTP requests",
             n_stmt, n_min, fetch.count)

    # --- sanity checks ---
    scheduled = [m for m in meetings if m.meeting_type == "scheduled"]
    for year, n in sorted(by_year.items()):
        n_sched = sum(1 for m in scheduled if m.meeting_date.year == year)
        if n_sched != 8:
            log.warning(
                "  %d has %d scheduled meetings, not the usual 8 "
                "(cancelled meetings are excluded - 2020's March 17-18 was "
                "cancelled and replaced by the March 15 unscheduled meeting)",
                year, n_sched,
            )

    missing_minutes = [m for m in scheduled if not m.minutes_file]
    if missing_minutes:
        log.warning("  %d scheduled meetings without minutes: %s",
                    len(missing_minutes), [str(m.meeting_date) for m in missing_minutes])

    bad_order = [
        m for m in meetings
        if m.statement_date and m.minutes_date and m.minutes_date <= m.statement_date
    ]
    if bad_order:
        log.error("  minutes dated on/before statement for: %s",
                  [str(m.meeting_date) for m in bad_order])
        return 1
    log.info("  publication-order check passed (minutes always after statement)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2016-01", help="first month, YYYY-MM")
    parser.add_argument("--end", default="2025-12", help="last month, YYYY-MM")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                        help=f"seconds between requests (default {DEFAULT_DELAY})")
    parser.add_argument("--force", action="store_true", help="re-download cached files")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    start = datetime.strptime(args.start + "-01", "%Y-%m-%d").date()
    first_of_end = datetime.strptime(args.end + "-01", "%Y-%m-%d").date()
    # Last day of the end month: first of the next month, minus a day.
    next_month = date(first_of_end.year + first_of_end.month // 12,
                      first_of_end.month % 12 + 1, 1)
    end = next_month - timedelta(days=1)
    if start >= end:
        log.error("--start must be before --end")
        return 2
    if args.delay < 1.0:
        log.warning("Delay below 1s is impolite for a public site; using 1.0s")
        args.delay = 1.0

    return scrape(start, end, args.delay, args.force)


if __name__ == "__main__":
    sys.exit(main())
