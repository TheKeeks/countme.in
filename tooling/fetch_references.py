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
    python tooling/fetch_references.py \\
        --song peggy-o --count 2 [--era 1977-1981] \\
        [--query "New Haven 1977"] [--source auto|relisten|archive|youtube]

The --query flag is the primary natural-language entry point. Examples:
    "New Haven 1977"  -> 5/5/77 New Haven Coliseum
    "5/8/77"          -> Cornell, Barton Hall
    "May 10 1978"     -> Veterans Memorial Coliseum
    "Cornell"         -> 5/8/77 Barton Hall
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


def parse_year_only(query: str) -> Optional[int]:
    """A bare 4-digit year like '1977' (not part of a date)."""
    if not query:
        return None
    for m in re.finditer(r"\b(19[6-9]\d|20\d{2})\b", query):
        return int(m.group(1))
    return None


def score_candidate(cand: Candidate, query: Optional[str]) -> float:
    """Fuzzy score: higher = better match for the query.

    Components:
        - Exact date match: +50
        - Year match: +10
        - Venue substring match: +20 (with partial credit for token overlap)
        - Keyword hits ("Cornell", "Dick's Picks"): +30
        - Era keyword: +5
        - Quality bias: SBD > MATRIX > AUD (+5/+3/+0)
    """
    cand.score = 0.0
    cand.score_reasons = []

    # Always-on quality bias so default sorts make sense even without a query.
    q_str = (cand.quality or "").upper()
    if "SBD" in q_str or "SOUNDBOARD" in q_str:
        cand.score += 5
        cand.score_reasons.append("+5 SBD")
    elif "MATRIX" in q_str:
        cand.score += 3
        cand.score_reasons.append("+3 matrix")
    elif "AUD" in q_str:
        cand.score_reasons.append("+0 AUD")

    if not query:
        return cand.score

    q = query.lower()

    # Keyword shortcuts (Cornell, Dick's Picks N, ...)
    for kw, (kdate, kvenue) in KEYWORD_DATES.items():
        if kw in q:
            if cand.show_date == kdate:
                cand.score += 30
                cand.score_reasons.append(f"+30 keyword '{kw}' -> date match")
            elif cand.venue and kvenue.lower() in cand.venue.lower():
                cand.score += 20
                cand.score_reasons.append(f"+20 keyword '{kw}' -> venue match")

    # Date match
    dates = parse_date_tokens(query)
    if dates and cand.show_date:
        for d in dates:
            if cand.show_date == d.isoformat():
                cand.score += 50
                cand.score_reasons.append(f"+50 exact date {d.isoformat()}")
                break

    # Year-only match
    yr = parse_year_only(query)
    if yr and cand.show_year == yr:
        cand.score += 10
        cand.score_reasons.append(f"+10 year {yr}")

    # Venue substring / token overlap
    if cand.venue:
        venue_l = cand.venue.lower()
        # Direct substring of significant tokens from query
        # Tokens are words >= 4 chars not consisting only of digits
        tokens = [
            t for t in re.findall(r"[a-z]+", q)
            if len(t) >= 4 and t not in {"show", "tape", "soundboard", "live", "with", "from"}
        ]
        venue_hits = [t for t in tokens if t in venue_l]
        if venue_hits:
            bonus = min(20, 8 * len(venue_hits))
            cand.score += bonus
            cand.score_reasons.append(f"+{bonus} venue tokens {venue_hits}")

    # Era keyword
    for kw, (lo, hi) in ERA_KEYWORDS.items():
        if kw in q and cand.show_year and lo <= cand.show_year <= hi:
            cand.score += 5
            cand.score_reasons.append(f"+5 era '{kw}'")

    return cand.score


# ---------------------------------------------------------------------------
# Relisten source
# ---------------------------------------------------------------------------

def _http_get_json(url: str, timeout: int = 30) -> Optional[object]:
    if requests is None:
        log.error("requests is not installed")
        return None
    try:
        resp = requests.get(url, timeout=timeout)
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

def archive_candidates(song: str, era: Optional[tuple[int, int]]) -> list[Candidate]:
    try:
        import internetarchive as ia
    except ImportError:
        log.warning("internetarchive not installed; skipping Archive source")
        return []

    query_parts = ['collection:GratefulDead', 'mediatype:etree']
    if era:
        query_parts.append(f"year:[{era[0]} TO {era[1]}]")
    query = " AND ".join(query_parts)

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
        with requests.get(url, stream=True, timeout=60) as resp:
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

def parse_era(s: Optional[str]) -> Optional[tuple[int, int]]:
    if not s:
        return None
    m = re.match(r"^\s*(\d{4})\s*-\s*(\d{4})\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--era must look like 1977-1981, got {s!r}")
    lo, hi = int(m.group(1)), int(m.group(2))
    if lo > hi:
        lo, hi = hi, lo
    return (lo, hi)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto-download Grateful Dead reference recordings for a song.")
    p.add_argument("--song", required=True, help="Song slug or name, e.g. 'peggy-o'")
    p.add_argument("--count", type=int, default=2, help="How many references to fetch (default 2)")
    p.add_argument("--era", type=parse_era, help="Year range filter, e.g. 1977-1981")
    p.add_argument("--query", help='Free-form match text, e.g. "New Haven 1977" or "5/8/77"')
    p.add_argument("--source", choices=["auto", "relisten", "archive", "youtube"], default="auto")
    p.add_argument("-v", "--verbose", action="store_true", help="Show each candidate and score")
    p.add_argument("--out", help="Override output directory (defaults to tooling/references/<song_slug>/)")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    song_slug = slugify(args.song)
    out_dir = Path(args.out) if args.out else REFERENCES_ROOT / song_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Fetching %d reference(s) for %s -> %s",
             args.count, song_slug, out_dir)
    if args.query:
        log.info("Query: %r", args.query)

    source_buckets = collect_candidates(args.source, args.song, args.query, args.era, args.count)

    downloaded: list[tuple[Candidate, Path]] = []
    seen_files: set[str] = set()
    manifest_entries: list[dict] = []

    sources_to_try = SOURCE_ORDER if args.source == "auto" else (args.source,)
    for s in sources_to_try:
        if len(downloaded) >= args.count:
            break
        cands = rank(source_buckets.get(s, []), args.query)
        if args.verbose:
            log.info("[%s] %d candidates", s, len(cands))
        for cand in cands:
            if len(downloaded) >= args.count:
                break
            fname = candidate_filename(song_slug, cand)
            if fname in seen_files:
                log_candidate(cand, args.verbose, kept=False, note="(dup filename)")
                continue

            local_path = download_candidate(cand, args.song, out_dir)
            if local_path is None:
                log_candidate(cand, args.verbose, kept=False, note="(download failed)")
                continue

            seen_files.add(fname)
            downloaded.append((cand, local_path))
            log_candidate(cand, args.verbose, kept=True)

            manifest_entries.append({
                "source": cand.source,
                "show_date": cand.show_date,
                "venue": cand.venue,
                "quality": cand.quality,
                "original_url": cand.url,
                "filename": local_path.name,
                "query": args.query,
                "score": round(cand.score, 2),
            })

    if manifest_entries:
        write_manifest(out_dir, manifest_entries)

    # Summary
    print(f"\nDownloaded {len(downloaded)}/{args.count} references for {song_slug}:")
    for cand, path in downloaded:
        print(f"  - [{cand.source}] {cand.show_date or '????'} "
              f"{cand.venue or '?'} ({cand.quality or '?'}) -> {path.name}")
    if not downloaded:
        print("  (nothing downloaded)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
