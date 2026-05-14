"""
fetch_references.py
-------------------
Auto-downloader for Grateful Dead reference recordings of a given song.

Pulls candidate live performances from (in order of preference):
  1. Relisten.net  -- cleanest metadata, good for query matching
  2. Internet Archive (GratefulDead collection) -- canonical source
  3. YouTube (via yt-dlp) -- last-resort fallback

The downloaded MP3s land in tooling/references/<song_slug>/ and are
intermediate inputs for the template-building pipeline. They are
intentionally not checked in (see .gitignore).

Usage
-----
    # Pin to a specific performance, pull multiple mixes of it:
    python tooling/fetch_references.py \\
        --song peggy-o --count 3 --show "5/8/77 Cornell" \\
        [--sources SBD,MATRIX]

    # Or pull different shows in an era / style:
    python tooling/fetch_references.py \\
        --song peggy-o --count 3 --era "1977-1981" \\
        [--sources SBD,MATRIX]

--show and --era both accept free-form text. Examples:
    "1977"            -> any 1977 show (±1 yr window)
    "1977-1981"       -> any show in the range
    "May 1977"        -> May of 1977
    "5/8/77"          -> the specific date
    "Cornell"         -> 5/8/77 Barton Hall
    "Dick's Picks 25" -> 5/10/78
    "Brent era"       -> 1979-1990

--sources is a comma-separated allowlist of recording mix types
(default: "SBD,MATRIX"). Use "SBD,AUD,MATRIX" to include audience tapes.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import date
from pathlib import Path
from typing import Iterable, Optional

try:
    from dateutil import parser as dateparser
except ImportError:  # pragma: no cover
    dateparser = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


log = logging.getLogger("fetch_references")

REFERENCES_ROOT = Path(__file__).resolve().parent / "references"
RELISTEN_BASE = "https://api.relisten.net/api/v3"
ARTIST_SLUG = "grateful-dead"

# Relisten (and Archive's CDN) reject default Python requests UAs with 403.
USER_AGENT = "countme.in/1.0 (https://github.com/TheKeeks/countme.in)"
HTTP_HEADERS = {"User-Agent": USER_AGENT}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A single candidate live performance being considered for download."""
    source: str                      # "relisten" | "archive" | "youtube"
    show_date: Optional[str]         # ISO YYYY-MM-DD if known
    venue: Optional[str]
    quality: Optional[str]           # "SBD" | "AUD" | "MATRIX" | etc.
    url: str                         # direct media URL or page URL
    title: Optional[str] = None
    extra: dict = field(default_factory=dict)
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    # True when at least one query-derived signal (date / year / venue /
    # keyword / era) matched. Quality bias alone does not count.
    query_matched: bool = False

    @property
    def show_year(self) -> Optional[int]:
        if self.show_date and len(self.show_date) >= 4:
            try:
                return int(self.show_date[:4])
            except ValueError:
                return None
        return None


# ---------------------------------------------------------------------------
# Slugging / normalization
# ---------------------------------------------------------------------------

def slugify(text: str, max_len: int = 60) -> str:
    """ASCII slug: lowercase, underscore-joined, punctuation stripped."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len]


def song_match_forms(song: str) -> set[str]:
    """Return normalized comparison forms for a song name.

    "peggy-o", "peggy o", "PeggyO" -> all collapse to the same canonical form.
    """
    base = re.sub(r"[^a-z0-9]+", "", song.lower())
    return {base, base.replace("o", ""), song.lower()}


def song_matches(title: str, song: str) -> bool:
    """Loose match: ignore hyphens, spaces, case."""
    if not title:
        return False
    norm_title = re.sub(r"[^a-z0-9]+", "", title.lower())
    norm_song = re.sub(r"[^a-z0-9]+", "", song.lower())
    return norm_song in norm_title


# ---------------------------------------------------------------------------
# Query parsing & scoring
# ---------------------------------------------------------------------------

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}


# A small directory of common-knowledge GD references, used to expand queries
# like "Cornell" -> 1977-05-08 or "Dick's Picks 25" -> 1978-05-10.
KEYWORD_DATES = {
    "cornell": ("1977-05-08", "Barton Hall"),
    "barton hall": ("1977-05-08", "Barton Hall"),
    "dick's picks 25": ("1978-05-10", "Veterans Memorial Coliseum"),
    "dicks picks 25": ("1978-05-10", "Veterans Memorial Coliseum"),
    "veneta": ("1972-08-27", "Field Trip"),
    "europe 72": ("1972-04-01", "Europe '72 tour"),
    "wall of sound": ("1974-06-23", "Wall of Sound era"),
}


ERA_KEYWORDS = {
    "primal": (1965, 1971),
    "europe 72": (1972, 1972),
    "wall of sound": (1973, 1974),
    "keith era": (1971, 1979),
    "brent era": (1979, 1990),
    "vince era": (1990, 1995),
    "jgb era": (1976, 1995),
}


def parse_date_tokens(query: str) -> list[date]:
    """Extract any dates referenced in the query string.

    Handles M/D/YY, M-D-YYYY, "May 10 1978", etc.
    """
    if not query:
        return []
    found: list[date] = []
    q = query.strip()

    # Numeric date: M/D/YY or M/D/YYYY (also -, .)
    for m in re.finditer(r"\b(\d{1,2})[/\-.](\d{1,2})(?:[/\-.](\d{2,4}))?\b", q):
        month, day, year = m.group(1), m.group(2), m.group(3)
        try:
            mo, dy = int(month), int(day)
            if not (1 <= mo <= 12 and 1 <= dy <= 31):
                continue
            if year is None:
                # No year provided; skip -- caller can still match month/day if needed
                continue
            yr = int(year)
            if yr < 100:
                yr += 1900 if yr >= 60 else 2000
            found.append(date(yr, mo, dy))
        except ValueError:
            continue

    # "May 10 1978" / "May 10, 1978" / "10 May 1978"
    for m in re.finditer(
        r"\b(" + "|".join(MONTHS.keys()) + r")\s+(\d{1,2})(?:[,\s]+(\d{2,4}))?\b",
        q.lower(),
    ):
        mo = MONTHS[m.group(1)]
        try:
            dy = int(m.group(2))
            if m.group(3):
                yr = int(m.group(3))
                if yr < 100:
                    yr += 1900 if yr >= 60 else 2000
                found.append(date(yr, mo, dy))
        except ValueError:
            continue

    # Last resort: dateutil
    if not found and dateparser is not None:
        try:
            dt = dateparser.parse(q, fuzzy=True, default=None)
            if dt:
                found.append(dt.date())
        except (ValueError, TypeError, OverflowError):
            pass

    return found


def parse_year_range(query: Optional[str]) -> Optional[tuple[int, int]]:
    """Detect `YYYY-YYYY` (e.g. `1977-1981`). Returns (lo, hi) or None."""
    if not query:
        return None
    m = re.search(r"\b(19[6-9]\d|20\d{2})\s*-\s*(19[6-9]\d|20\d{2})\b", query)
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    return (min(a, b), max(a, b))


def parse_month_year(query: Optional[str]) -> Optional[tuple[int, int]]:
    """Detect bare month/year refs like `May 1977` or `1977-05` (no day).

    Skips full dates (`May 8 1977`, `1977-05-08`) which the date parser handles.
    """
    if not query:
        return None
    q = query.lower()
    for m in re.finditer(
        r"\b(" + "|".join(MONTHS.keys()) + r")[,\s]+(19[6-9]\d|20\d{2})\b", q
    ):
        return (MONTHS[m.group(1)], int(m.group(2)))
    for m in re.finditer(
        r"\b(19[6-9]\d|20\d{2})-(\d{1,2})\b(?!-\d)(?!\s*-\s*\d{4})", q
    ):
        month = int(m.group(2))
        if 1 <= month <= 12:
            return (month, int(m.group(1)))
    return None


def parse_year_only(query: str) -> Optional[int]:
    """A bare 4-digit year like '1977' (not part of a YYYY-YYYY range)."""
    if not query:
        return None
    if parse_year_range(query) is not None:
        return None
    for m in re.finditer(r"\b(19[6-9]\d|20\d{2})\b", query):
        return int(m.group(1))
    return None


def query_target_year(query: Optional[str]) -> Optional[int]:
    """Best guess at what year the query is asking for, for ±1 window filtering.

    Returns None when the query specifies a year range -- the range supersedes
    the single-year ±1 window penalty.
    """
    if not query:
        return None
    if parse_year_range(query) is not None:
        return None
    # Hand-curated keyword shortcuts win first -- otherwise something like
    # "Dick's Picks 25" gets day-parsed by dateutil into the current year.
    q = query.lower()
    for kw, (kdate, _venue) in KEYWORD_DATES.items():
        if kw in q:
            try:
                return int(kdate[:4])
            except ValueError:
                pass
    dates = parse_date_tokens(query)
    if dates:
        return dates[0].year
    my = parse_month_year(query)
    if my:
        return my[1]
    return parse_year_only(query)


YEAR_WINDOW_PENALTY = -100
YEAR_RANGE_OUTSIDE_PENALTY = -50


def score_candidate(cand: Candidate, query: Optional[str]) -> float:
    """Fuzzy score: higher = better match for the query.

    Components:
        - Exact date match: +50
        - Keyword hits ("Cornell", "Dick's Picks"): +30
        - Month+year match: +25 (same month and year)
        - Venue substring match: up to +20 (token-overlap based)
        - Single-year match: +10
        - Month/year partial: +10 same year diff month, +5 same month diff year
        - Year-in-range (YYYY-YYYY): +5
        - Era keyword: +5
        - Quality bias: SBD > MATRIX > AUD (+5/+3/+0)
        - Year-window penalty: -100 for single-year queries when the
          candidate is more than 1 year away (skipped if a range was given).
        - Year-out-of-range penalty: -50 for candidates outside an
          explicit YYYY-YYYY range in the query.
    """
    cand.score = 0.0
    cand.score_reasons = []
    cand.query_matched = False

    def mark(bonus: float, reason: str, *, query_signal: bool = True) -> None:
        cand.score += bonus
        cand.score_reasons.append(reason)
        if query_signal:
            cand.query_matched = True

    # Always-on quality bias so default sorts make sense even without a query.
    q_str = (cand.quality or "").upper()
    if "SBD" in q_str or "SOUNDBOARD" in q_str:
        mark(5, "+5 SBD", query_signal=False)
    elif "MATRIX" in q_str:
        mark(3, "+3 matrix", query_signal=False)
    elif "AUD" in q_str:
        cand.score_reasons.append("+0 AUD")

    if not query:
        return cand.score

    q = query.lower()
    year_range = parse_year_range(query)
    month_year = parse_month_year(query)

    # Year range (e.g. "1977-1981"): in-range +5, out-of-range -50.
    if year_range and cand.show_year is not None:
        lo, hi = year_range
        if lo <= cand.show_year <= hi:
            mark(5, f"+5 year {cand.show_year} in range {lo}-{hi}")
        else:
            cand.score += YEAR_RANGE_OUTSIDE_PENALTY
            cand.score_reasons.append(
                f"{YEAR_RANGE_OUTSIDE_PENALTY} year {cand.show_year} outside {lo}-{hi}"
            )

    # Month/year (e.g. "May 1977" or "1977-05"): graded.
    if month_year and cand.show_date and cand.show_year is not None:
        target_month, target_year = month_year
        cand_month: Optional[int] = None
        try:
            cand_month = int(cand.show_date[5:7])
        except (ValueError, IndexError):
            cand_month = None
        if cand_month is not None:
            if cand_month == target_month and cand.show_year == target_year:
                mark(25, f"+25 month/year {target_year}-{target_month:02d}")
            elif cand.show_year == target_year:
                mark(10, f"+10 same year {target_year}")
            elif cand_month == target_month:
                mark(5, f"+5 same month {target_month:02d}")

    # Keyword shortcuts (Cornell, Dick's Picks N, ...)
    for kw, (kdate, kvenue) in KEYWORD_DATES.items():
        if kw in q:
            if cand.show_date == kdate:
                mark(30, f"+30 keyword '{kw}' -> date match")
            elif cand.venue and kvenue.lower() in cand.venue.lower():
                mark(20, f"+20 keyword '{kw}' -> venue match")

    # Date match
    dates = parse_date_tokens(query)
    if dates and cand.show_date:
        for d in dates:
            if cand.show_date == d.isoformat():
                mark(50, f"+50 exact date {d.isoformat()}")
                break

    # Year-only match (skipped if month/year already scored the year axis).
    if not month_year:
        yr = parse_year_only(query)
        if yr and cand.show_year == yr:
            mark(10, f"+10 year {yr}")

    # Venue substring / token overlap
    if cand.venue:
        venue_l = cand.venue.lower()
        tokens = [
            t for t in re.findall(r"[a-z]+", q)
            if len(t) >= 4 and t not in {"show", "tape", "soundboard", "live", "with", "from"}
        ]
        venue_hits = [t for t in tokens if t in venue_l]
        if venue_hits:
            bonus = min(20, 8 * len(venue_hits))
            mark(bonus, f"+{bonus} venue tokens {venue_hits}")

    # Era keyword
    for kw, (lo, hi) in ERA_KEYWORDS.items():
        if kw in q and cand.show_year and lo <= cand.show_year <= hi:
            mark(5, f"+5 era '{kw}'")

    # Year-window penalty: skipped when a range was provided (range handles
    # its own filtering).
    target_year = query_target_year(query)
    if target_year and cand.show_year and abs(cand.show_year - target_year) > 1:
        cand.score += YEAR_WINDOW_PENALTY
        cand.score_reasons.append(
            f"{YEAR_WINDOW_PENALTY} year {cand.show_year} outside {target_year}±1"
        )

    return cand.score


# ---------------------------------------------------------------------------
# Relisten source
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 30) -> Optional[object]:
    if requests is None:
        log.error("requests is not installed")
        return None
    try:
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            log.debug("GET %s -> %s", url, resp.status_code)
            return None
        return resp.json()
    except requests.RequestException as exc:
        log.debug("GET %s failed: %s", url, exc)
        return None


def relisten_find_song_slug(song: str) -> Optional[str]:
    """Look up the Relisten song slug for a song name (loose match)."""
    data = _http_get_json(f"{RELISTEN_BASE}/artists/{ARTIST_SLUG}/songs")
    if not isinstance(data, list):
        return None
    norm_target = re.sub(r"[^a-z0-9]+", "", song.lower())
    best: Optional[tuple[int, str]] = None  # (preference, slug)
    for entry in data:
        name = entry.get("name") or ""
        slug = entry.get("slug") or ""
        norm_name = re.sub(r"[^a-z0-9]+", "", name.lower())
        norm_slug = re.sub(r"[^a-z0-9]+", "", slug.lower())
        if norm_target == norm_slug:
            return slug
        if norm_target == norm_name:
            return slug
        if norm_target and (norm_target in norm_name or norm_target in norm_slug):
            if best is None:
                best = (len(norm_name), slug)
    return best[1] if best else None


def relisten_candidates(song: str, era: Optional[tuple[int, int]]) -> list[Candidate]:
    slug = relisten_find_song_slug(song)
    if not slug:
        log.info("Relisten: song slug not found for %r", song)
        return []

    song_data = _http_get_json(f"{RELISTEN_BASE}/artists/{ARTIST_SLUG}/songs/{slug}")
    if not isinstance(song_data, dict):
        return []

    shows = song_data.get("shows") or song_data.get("tracks") or []
    candidates: list[Candidate] = []
    for entry in shows:
        # Relisten v3 returns either show stubs or track entries; handle both shapes.
        show_date = entry.get("display_date") or entry.get("date") or entry.get("show_date")
        venue = None
        venue_obj = entry.get("venue")
        if isinstance(venue_obj, dict):
            venue = venue_obj.get("name") or venue_obj.get("location")
        elif isinstance(venue_obj, str):
            venue = venue_obj

        if show_date and "T" in show_date:
            show_date = show_date.split("T", 1)[0]

        if era and show_date:
            try:
                yr = int(show_date[:4])
                if not (era[0] <= yr <= era[1]):
                    continue
            except ValueError:
                pass

        show_uuid = entry.get("show_uuid") or entry.get("uuid") or entry.get("show_id")
        candidates.append(Candidate(
            source="relisten",
            show_date=show_date,
            venue=venue,
            quality=(entry.get("source_type") or "").upper() or None,
            url="",  # resolved later via the show endpoint
            title=entry.get("title") or song,
            extra={"show_uuid": show_uuid, "song_slug": slug},
        ))
    return candidates


def relisten_resolve_track_url(cand: Candidate, song: str) -> Optional[str]:
    """Drill into a Relisten show and pick the track URL for this song."""
    show_uuid = cand.extra.get("show_uuid")
    if not show_uuid:
        return None
    show = _http_get_json(f"{RELISTEN_BASE}/artists/{ARTIST_SLUG}/shows/{show_uuid}")
    if not isinstance(show, dict):
        return None

    # Walk sources -> sets -> tracks looking for the matching song.
    sources = show.get("sources") or []
    # Prefer SBD sources.
    sources.sort(key=lambda s: 0 if "SBD" in (s.get("source_type") or "").upper() else 1)
    for src in sources:
        if not cand.quality:
            cand.quality = (src.get("source_type") or "").upper() or None
        for st in src.get("sets") or []:
            for track in st.get("tracks") or []:
                title = track.get("title") or ""
                if song_matches(title, song):
                    return track.get("mp3_url") or track.get("flac_url")
    return None


# ---------------------------------------------------------------------------
# Internet Archive source
# ---------------------------------------------------------------------------

def _archive_song_phrase(song: str) -> str:
    """Phrase form of a song name for Archive's full-text search.

    Hyphens become spaces; the rest is left alone. Used inside quoted
    field queries, so internal punctuation other than hyphens is fine.
    """
    return re.sub(r"[-_]+", " ", song).strip()


def archive_candidates(song: str, era: Optional[tuple[int, int]]) -> list[Candidate]:
    try:
        import internetarchive as ia
    except ImportError:
        log.warning("internetarchive not installed; skipping Archive source")
        return []

    # Pre-filter to shows that actually mention the song. Without this we'd
    # pull the entire GratefulDead collection (~10k items) and try them
    # one-by-one before noticing the song isn't on the setlist.
    phrase = _archive_song_phrase(song)
    song_clause = (
        f'(subject:"{phrase}" OR description:"{phrase}" OR title:"{phrase}")'
    )
    query_parts = ['collection:GratefulDead', 'mediatype:etree', song_clause]
    if era:
        query_parts.append(f"year:[{era[0]} TO {era[1]}]")
    query = " AND ".join(query_parts)
    log.debug("Archive query: %s", query)

    candidates: list[Candidate] = []
    try:
        results = ia.search_items(query, fields=["identifier", "date", "venue", "coverage", "source", "title"])
        for n, hit in enumerate(results):
            if n > 400:
                break
            ident = hit.get("identifier")
            if not ident:
                continue
            src_field = (hit.get("source") or "").upper()
            quality = None
            ident_upper = ident.upper()
            if "SBD" in src_field or ".SBD" in ident_upper or "SBD." in ident_upper:
                quality = "SBD"
            elif "MATRIX" in src_field or "MTX" in ident_upper:
                quality = "MATRIX"
            elif "AUD" in src_field:
                quality = "AUD"

            # We only want soundboards for the primary path; keep matrix as
            # a fallback if a query points to one specifically.
            if quality not in {"SBD", "MATRIX"}:
                continue

            show_date = hit.get("date")
            if show_date and "T" in show_date:
                show_date = show_date.split("T", 1)[0]
            venue = hit.get("venue") or hit.get("coverage")
            candidates.append(Candidate(
                source="archive",
                show_date=show_date,
                venue=venue,
                quality=quality,
                url=f"https://archive.org/details/{ident}",
                title=hit.get("title"),
                extra={"identifier": ident},
            ))
    except Exception as exc:
        log.warning("Archive search failed: %s", exc)
    return candidates


def archive_resolve_track_url(cand: Candidate, song: str) -> Optional[str]:
    """Find the MP3 transcode for the given song in this Archive item."""
    try:
        import internetarchive as ia
    except ImportError:
        return None
    ident = cand.extra.get("identifier")
    if not ident:
        return None
    try:
        item = ia.get_item(ident)
    except Exception as exc:
        log.debug("get_item(%s) failed: %s", ident, exc)
        return None

    # Prefer 64Kbps or VBR MP3 transcodes; never shn/flac (login often required).
    mp3_files = [
        f for f in item.files
        if (f.get("format") or "").lower() in {"vbr mp3", "64kbps mp3", "128kbps mp3", "mp3"}
        or (f.get("name") or "").lower().endswith(".mp3")
    ]
    for f in mp3_files:
        title = f.get("title") or f.get("name") or ""
        if song_matches(title, song):
            return f"https://archive.org/download/{ident}/{f['name']}"
    return None


# ---------------------------------------------------------------------------
# YouTube source
# ---------------------------------------------------------------------------

def youtube_candidates(song: str, query: Optional[str], era: Optional[tuple[int, int]],
                       count: int) -> list[Candidate]:
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        log.warning("yt-dlp not installed; skipping YouTube source")
        return []

    search_terms = [f'Grateful Dead "{song.replace("-", " ")}"']
    if query:
        search_terms.append(query)
    elif era:
        search_terms.append(f"{era[0]}-{era[1]}")
    search_terms.append("soundboard")
    search = " ".join(search_terms)

    # ytsearchN: pick the top N
    n = max(count * 3, 5)
    from yt_dlp import YoutubeDL
    candidates: list[Candidate] = []
    try:
        with YoutubeDL({"quiet": True, "skip_download": True, "extract_flat": "in_playlist"}) as ydl:
            info = ydl.extract_info(f"ytsearch{n}:{search}", download=False)
    except Exception as exc:
        log.warning("YouTube search failed: %s", exc)
        return []

    for entry in (info or {}).get("entries", []) or []:
        if not entry:
            continue
        title = entry.get("title") or ""
        url = entry.get("url") or entry.get("webpage_url") or entry.get("id")
        if url and not str(url).startswith("http"):
            url = f"https://www.youtube.com/watch?v={url}"
        # Try to scrape a date from the title
        m = re.search(r"(19\d{2}|20\d{2})[-/.]?(\d{1,2})[-/.]?(\d{1,2})", title)
        show_date = None
        if m:
            try:
                show_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
            except ValueError:
                show_date = None
        candidates.append(Candidate(
            source="youtube",
            show_date=show_date,
            venue=None,
            quality="SBD" if "soundboard" in title.lower() or "sbd" in title.lower() else "AUD",
            url=url or "",
            title=title,
            extra={"channel": entry.get("uploader")},
        ))
    return candidates


def youtube_download(url: str, out_dir: Path, stem: str) -> Optional[Path]:
    try:
        from yt_dlp import YoutubeDL
    except ImportError:
        return None
    target_template = str(out_dir / f"{stem}.%(ext)s")
    final_path = out_dir / f"{stem}.mp3"
    ydl_opts = {
        "quiet": True,
        "outtmpl": target_template,
        "format": "bestaudio/best",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as exc:
        log.warning("yt-dlp download failed for %s: %s", url, exc)
        return None
    return final_path if final_path.exists() else None


# ---------------------------------------------------------------------------
# Download utilities
# ---------------------------------------------------------------------------

def http_download(url: str, dest: Path) -> bool:
    if requests is None:
        return False
    try:
        with requests.get(url, headers=HTTP_HEADERS, stream=True, timeout=60) as resp:
            if resp.status_code != 200:
                log.warning("HTTP %s for %s", resp.status_code, url)
                return False
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
            tmp.replace(dest)
        return True
    except requests.RequestException as exc:
        log.warning("Download failed for %s: %s", url, exc)
        return False


def venue_slug(venue: Optional[str]) -> str:
    if not venue:
        return "unknown"
    s = slugify(venue, max_len=30)
    return s or "unknown"


def date_slug(show_date: Optional[str]) -> str:
    if not show_date:
        return "00000000"
    parts = show_date.split("-")
    if len(parts) == 3:
        return "".join(p.zfill(2) for p in parts)[:8]
    return re.sub(r"[^0-9]", "", show_date)[:8] or "00000000"


def candidate_filename(song_slug: str, cand: Candidate) -> str:
    return f"{song_slug}_{date_slug(cand.show_date)}_{venue_slug(cand.venue)}.mp3"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

SOURCE_ORDER = ("relisten", "archive", "youtube")


def collect_candidates(source: str, song: str, query: Optional[str],
                       era: Optional[tuple[int, int]], count: int) -> dict[str, list[Candidate]]:
    """Gather candidates from the requested source(s)."""
    sources = SOURCE_ORDER if source == "auto" else (source,)
    out: dict[str, list[Candidate]] = {}
    for s in sources:
        if s == "relisten":
            out[s] = relisten_candidates(song, era)
        elif s == "archive":
            out[s] = archive_candidates(song, era)
        elif s == "youtube":
            out[s] = youtube_candidates(song, query, era, count)
        else:
            log.warning("Unknown source %r", s)
            out[s] = []
        log.debug("Source %s returned %d candidates", s, len(out[s]))
    return out


def rank(cands: list[Candidate], query: Optional[str]) -> list[Candidate]:
    for c in cands:
        score_candidate(c, query)
    cands.sort(key=lambda c: (-c.score, c.show_date or "9999"))
    return cands


def log_candidate(c: Candidate, verbose: bool, kept: bool, note: str = "") -> None:
    if not verbose:
        return
    flag = "PICK " if kept else "skip "
    log.info("  %s [%s] %s @ %s (%s) score=%.1f %s %s",
             flag, c.source, c.show_date or "????", c.venue or "?",
             c.quality or "?", c.score,
             ",".join(c.score_reasons), note)


def download_candidate(cand: Candidate, song: str, out_dir: Path) -> Optional[Path]:
    """Resolve and download a candidate; return local path on success."""
    song_slug = slugify(song)
    target = out_dir / candidate_filename(song_slug, cand)
    if target.exists():
        log.info("  already have %s (skipping)", target.name)
        return target

    if cand.source == "relisten":
        url = relisten_resolve_track_url(cand, song)
        if not url:
            log.warning("  no Relisten track url for %s", cand.show_date)
            return None
        cand.url = url
        ok = http_download(url, target)
        return target if ok else None

    if cand.source == "archive":
        url = archive_resolve_track_url(cand, song)
        if not url:
            log.warning("  no Archive mp3 transcode for %s/%s",
                        cand.extra.get("identifier"), song)
            return None
        cand.url = url
        ok = http_download(url, target)
        return target if ok else None

    if cand.source == "youtube":
        path = youtube_download(cand.url, out_dir, target.stem)
        return path

    return None


def write_manifest(out_dir: Path, entries: list[dict]) -> None:
    manifest_path = out_dir / "manifest.json"
    existing: list[dict] = []
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text())
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    # Merge by filename
    by_name = {e.get("filename"): e for e in existing}
    for e in entries:
        by_name[e["filename"]] = e
    manifest_path.write_text(json.dumps(list(by_name.values()), indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_SOURCES = "SBD,MATRIX"


def parse_sources(s: Optional[str]) -> set[str]:
    """Comma-separated mix-type allowlist -> {'SBD', 'MATRIX', ...} (uppercase)."""
    if not s:
        return set()
    return {item.strip().upper() for item in s.split(",") if item.strip()}


def candidate_in_sources(cand: Candidate, allowed: set[str]) -> bool:
    if not allowed:
        return True
    q = (cand.quality or "").upper()
    if not q:
        # Unknown source type: only let it through if the user explicitly
        # opened the allowlist with "ANY" or "*".
        return "ANY" in allowed or "*" in allowed
    # SBD substring match handles the "SOUNDBOARD"/"SBD"/"SBD>FOB" varieties.
    return any(a in q or q in a for a in allowed)


def derive_year_range(query: Optional[str]) -> Optional[tuple[int, int]]:
    """Pull a (lo, hi) year window out of a free-form query, for pre-filtering.

    Used to narrow Archive/Relisten searches before the per-candidate
    scorer takes over. A bare year becomes that year ±1. A range becomes
    itself. A specific date becomes a 1-year window around it. Era
    keywords (Brent era etc.) expand to their canonical bounds.
    """
    if not query:
        return None
    rng = parse_year_range(query)
    if rng:
        return rng
    # Keyword shortcuts beat dateutil fuzzy parsing: "Brent era" must map to
    # (1979, 1990), not to today's year via fuzzy-parse.
    q = query.lower()
    for kw, (kdate, _venue) in KEYWORD_DATES.items():
        if kw in q:
            try:
                y = int(kdate[:4])
                return (y, y)
            except ValueError:
                pass
    for kw, span in ERA_KEYWORDS.items():
        if kw in q:
            return span
    my = parse_month_year(query)
    if my:
        return (my[1], my[1])
    yr = parse_year_only(query)
    if yr:
        # Bare year: widen by ±1 so Archive search includes adjacent shows
        # the scorer's ±1 tolerance is willing to consider.
        return (yr - 1, yr + 1)
    dates = parse_date_tokens(query)
    if dates:
        y = dates[0].year
        return (y, y)
    return None


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Auto-download Grateful Dead reference recordings for a song."
    )
    p.add_argument("--song", required=True, help="Song slug or name, e.g. 'peggy-o'")
    p.add_argument("--count", type=int, default=3,
                   help="How many references to fetch (default 3)")
    p.add_argument(
        "--show",
        help=(
            "Specific show (e.g. '5/8/77 Cornell', \"Dick's Picks 25\"). "
            "When set, fetches multiple distinct mixes of THIS one show."
        ),
    )
    p.add_argument(
        "--era",
        help=(
            "Era / style (e.g. '1977', '1977-1981', 'May 1977', 'Brent era'). "
            "When set (and --show is empty), fetches different shows that match "
            "the era. Accepts year, year range, month+year, specific date, "
            "venue/keyword, or era keyword."
        ),
    )
    # Backwards-compat: --query is a deprecated alias for --era.
    p.add_argument("--query", help=argparse.SUPPRESS)
    p.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help=(
            "Comma-separated allowlist of recording sources (default 'SBD,MATRIX'). "
            "Examples: 'SBD' for pure soundboards only, 'SBD,AUD,MATRIX' to "
            "include audience tapes. Pass 'ANY' to accept unlabelled candidates."
        ),
    )
    p.add_argument("--source", choices=["auto", "relisten", "archive", "youtube"], default="auto",
                   help="Which upstream service to query (default auto cascades).")
    p.add_argument("-v", "--verbose", action="store_true", help="Show each candidate and score")
    p.add_argument("--out", help="Override output directory (defaults to tooling/references/<song_slug>/)")
    p.add_argument(
        "--max-attempts-per-source",
        type=int,
        default=25,
        help="Give up on a source after this many consecutive download failures (default 25)",
    )
    args = p.parse_args(argv)

    if args.query and not args.era:
        log.warning("--query is deprecated; use --era for the same behaviour.")
        args.era = args.query
    if args.show and args.era:
        log.debug("--show is set; --era %r will be ignored.", args.era)
        args.era = None

    return args


def _quality_bucket(cand: Candidate) -> str:
    """Coarse mix-type bucket for diversification: SBD / MATRIX / AUD / OTHER."""
    q = (cand.quality or "").upper()
    if "SBD" in q or "SOUNDBOARD" in q:
        return "SBD"
    if "MATRIX" in q:
        return "MATRIX"
    if "AUD" in q:
        return "AUD"
    return "OTHER"


def _show_key(cand: Candidate) -> tuple:
    """Identity for "the same performance" -- date is the canonical anchor."""
    return (cand.show_date or "?", _venue_key(cand.venue))


def _venue_key(venue: Optional[str]) -> str:
    if not venue:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", venue.lower()).strip("_")


def select_canonical_show(cands: list[Candidate]) -> Optional[tuple[str, Optional[str]]]:
    """Return the (date, venue) of the highest-scored candidate with a date."""
    for c in cands:
        if c.show_date:
            return (c.show_date, c.venue)
    return None


def diversify_by_quality(cands: list[Candidate], count: int) -> list[Candidate]:
    """Round-robin pick across SBD / MATRIX / AUD buckets, prefer top-scored within each."""
    buckets: dict[str, list[Candidate]] = {}
    for c in cands:
        buckets.setdefault(_quality_bucket(c), []).append(c)
    # Bias bucket iteration order: SBD first, then MATRIX, then AUD, then anything else.
    order = ["SBD", "MATRIX", "AUD"] + [b for b in buckets if b not in {"SBD", "MATRIX", "AUD"}]
    picked: list[Candidate] = []
    while len(picked) < count and any(buckets.get(b) for b in order):
        for b in order:
            if buckets.get(b):
                picked.append(buckets[b].pop(0))
                if len(picked) >= count:
                    break
    return picked


def dedupe_by_year_month(cands: list[Candidate]) -> list[Candidate]:
    """Keep at most one candidate per (year, month) for stylistic spread."""
    seen: set[tuple[int, int]] = set()
    out: list[Candidate] = []
    for c in cands:
        if not c.show_date or len(c.show_date) < 7:
            out.append(c)
            continue
        try:
            ym = (int(c.show_date[:4]), int(c.show_date[5:7]))
        except ValueError:
            out.append(c)
            continue
        if ym in seen:
            continue
        seen.add(ym)
        out.append(c)
    return out


def _try_download(cand: Candidate, song: str, song_slug: str,
                  out_dir: Path, seen_files: set[str], verbose: bool) -> Optional[Path]:
    fname = candidate_filename(song_slug, cand)
    if fname in seen_files:
        log_candidate(cand, verbose, kept=False, note="(dup filename)")
        return None
    local_path = download_candidate(cand, song, out_dir)
    if local_path is None:
        log_candidate(cand, verbose, kept=False, note="(download failed)")
        return None
    seen_files.add(fname)
    log_candidate(cand, verbose, kept=True)
    return local_path


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    song_slug = slugify(args.song)
    out_dir = Path(args.out) if args.out else REFERENCES_ROOT / song_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    allowed_sources = parse_sources(args.sources)

    if args.show:
        mode = "show"
        match_query = args.show
    elif args.era:
        mode = "era"
        match_query = args.era
    else:
        mode = "default"
        match_query = None
        log.warning(
            "Neither --show nor --era given; falling back to 'most popular SBDs' for %s.",
            song_slug,
        )

    log.info("Fetching %d reference(s) for %s -> %s [mode=%s]",
             args.count, song_slug, out_dir, mode)
    if match_query:
        log.info("Match query: %r", match_query)
    log.info("Sources allowlist: %s", sorted(allowed_sources) or "(any)")

    year_range = derive_year_range(match_query)
    source_buckets = collect_candidates(
        args.source, args.song, match_query, year_range, args.count
    )

    # Gather, filter by sources, score, and tag with the source they came from.
    all_cands: list[Candidate] = []
    for s, cands in source_buckets.items():
        kept = [c for c in cands if candidate_in_sources(c, allowed_sources)]
        if args.verbose:
            log.info("[%s] %d candidates (%d after --sources filter)",
                     s, len(cands), len(kept))
        all_cands.extend(kept)

    ranked = rank(all_cands, match_query)

    if match_query and ranked and not any(c.query_matched for c in ranked):
        log.error("No good matches for query %r in any source. Aborting.", match_query)
        return 2

    downloaded: list[tuple[Candidate, Path]] = []
    seen_files: set[str] = set()
    manifest_entries: list[dict] = []

    if mode == "show":
        canonical = select_canonical_show(ranked)
        if not canonical:
            log.error("Could not identify a canonical show for query %r.", match_query)
            return 2
        target_date, target_venue = canonical
        log.info("Canonical show: %s @ %s -- pulling up to %d mixes",
                 target_date, target_venue or "?", args.count)
        same_show = [c for c in ranked if c.show_date == target_date]
        picks = diversify_by_quality(same_show, args.count)
    elif mode == "era":
        picks = dedupe_by_year_month(ranked)
    else:
        picks = ranked

    # Download loop with per-source failure cap.
    failures_by_source: dict[str, int] = {}
    for cand in picks:
        if len(downloaded) >= args.count:
            break
        n_fail = failures_by_source.get(cand.source, 0)
        if n_fail >= args.max_attempts_per_source:
            log_candidate(cand, args.verbose, kept=False,
                          note=f"(source {cand.source} hit failure cap)")
            continue
        local_path = _try_download(cand, args.song, song_slug, out_dir,
                                   seen_files, args.verbose)
        if local_path is None:
            failures_by_source[cand.source] = n_fail + 1
            continue
        failures_by_source[cand.source] = 0
        downloaded.append((cand, local_path))
        manifest_entries.append({
            "source": cand.source,
            "show_date": cand.show_date,
            "venue": cand.venue,
            "quality": cand.quality,
            "original_url": cand.url,
            "filename": local_path.name,
            "mode": mode,
            "show": args.show,
            "era": args.era,
            "score": round(cand.score, 2),
        })

    if manifest_entries:
        write_manifest(out_dir, manifest_entries)

    any_query_matches_seen = bool(match_query and any(c.query_matched for c in ranked))

    # Summary
    print(f"\nDownloaded {len(downloaded)}/{args.count} references for {song_slug}:")
    for cand, path in downloaded:
        print(f"  - [{cand.source}] {cand.show_date or '????'} "
              f"{cand.venue or '?'} ({cand.quality or '?'}) -> {path.name}")
    if not downloaded:
        if args.query and not any_query_matches_seen:
            print(f"  no good matches for query {args.query!r} in any source")
            return 2
        print("  (nothing downloaded)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
