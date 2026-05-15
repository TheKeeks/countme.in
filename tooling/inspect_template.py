"""
inspect_template.py
-------------------
Diagnostic CLI for built song templates. Prints the same report
build_song.py logs at the end of a build, plus a per-section breakdown
table, so any template (past or present) can be QA'd without re-running
the build pipeline.

    python tooling/inspect_template.py web/templates/peggy_o_aligned.json

The report logic is also imported by build_song.py to emit the report
during the build itself; keep the two in sync by editing the helpers
here rather than copy-pasting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Sanity floors: a real song with fewer distinct timings or embeddings
# than this is almost certainly a broken build.
MIN_UNIQUE_TIMINGS = 4
MIN_UNIQUE_EMBEDDINGS = 4

# Section type heuristic. Anything not in this set is treated as vocal.
INSTRUMENTAL_TYPES = {
    "intro", "outro", "jam", "instrumental", "interlude", "solo", "break",
}


def _all_lines(template: dict) -> list[dict]:
    return [l for s in template.get("structure", []) for l in s.get("lines", [])]


def _is_instrumental(section: dict) -> bool:
    return (section.get("section_type") or "").lower() in INSTRUMENTAL_TYPES


def _embedding_key(emb) -> Optional[tuple]:
    if not emb:
        return None
    return tuple(emb)


def _line_interpolated_in_blend(line: dict) -> bool:
    """True iff every per-reference timing for this line was interpolated."""
    per_ref = line.get("timing_per_reference") or {}
    if not per_ref:
        return True
    return all(bool(sp.get("interpolated", True)) for sp in per_ref.values())


def _per_reference_match_rates(template: dict) -> dict[str, tuple[int, int]]:
    """Per-reference: how many lines did this reference actually pin via whisper?"""
    refs = template.get("references") or []
    total = len(_all_lines(template))
    rates: dict[str, tuple[int, int]] = {}
    for ref in refs:
        matched = 0
        for line in _all_lines(template):
            sp = (line.get("timing_per_reference") or {}).get(ref)
            if sp and not sp.get("interpolated", True) and sp.get("matched_word_count", 0) > 0:
                matched += 1
        rates[ref] = (matched, total)
    return rates


def compute_stats(template: dict) -> dict:
    """Compute the diagnostic numbers as a flat dict; used by report() and tests."""
    sections = template.get("structure", [])
    lines = _all_lines(template)
    n_lines = len(lines)
    n_sections = len(sections)
    n_instrumental = sum(1 for s in sections if _is_instrumental(s))
    n_vocal = n_sections - n_instrumental

    n_sections_with_seq = sum(1 for s in sections if s.get("audio_embedding_sequence"))
    empty_seq_sections = [
        s.get("section_id", "?") for s in sections
        if not s.get("audio_embedding_sequence")
    ]

    interpolated = sum(1 for l in lines if _line_interpolated_in_blend(l))

    unique_timings = {
        (l.get("start_sec"), l.get("end_sec"))
        for l in lines
        if l.get("start_sec") is not None or l.get("end_sec") is not None
    }
    unique_embeddings = {
        _embedding_key(l.get("audio_embedding_blended"))
        for l in lines
        if l.get("audio_embedding_blended")
    }
    unique_embeddings.discard(None)

    return {
        "references": template.get("references") or [],
        "embedding_model": template.get("embedding_model"),
        "n_sections": n_sections,
        "n_instrumental_sections": n_instrumental,
        "n_vocal_sections": n_vocal,
        "n_sections_with_seq": n_sections_with_seq,
        "empty_seq_sections": empty_seq_sections,
        "n_lines": n_lines,
        "n_interpolated_lines": interpolated,
        "n_unique_timings": len(unique_timings),
        "n_unique_embeddings": len(unique_embeddings),
        "per_reference_match_rates": _per_reference_match_rates(template),
    }


def warnings_for(stats: dict) -> list[str]:
    warns: list[str] = []
    if stats["n_lines"] >= MIN_UNIQUE_TIMINGS and stats["n_unique_timings"] < MIN_UNIQUE_TIMINGS:
        warns.append(
            f"unique line-level blended timings ({stats['n_unique_timings']}) "
            f"< {MIN_UNIQUE_TIMINGS}"
        )
    if stats["n_lines"] >= MIN_UNIQUE_EMBEDDINGS and stats["n_unique_embeddings"] < MIN_UNIQUE_EMBEDDINGS:
        warns.append(
            f"unique line-level blended embeddings ({stats['n_unique_embeddings']}) "
            f"< {MIN_UNIQUE_EMBEDDINGS}"
        )
    if stats["empty_seq_sections"]:
        warns.append(
            f"sections with empty audio_embedding_sequence: "
            f"{stats['empty_seq_sections']}"
        )
    if stats["n_lines"] and stats["n_interpolated_lines"] >= 0.5 * stats["n_lines"]:
        warns.append(
            f"interpolated lines ({stats['n_interpolated_lines']}/"
            f"{stats['n_lines']}) make up >=50% of the song"
        )
    return warns


def report(template: dict, n_attempted_refs: Optional[int] = None) -> str:
    """Multi-line summary string. Used by both the inspector and the builder."""
    stats = compute_stats(template)
    attempted = n_attempted_refs if n_attempted_refs is not None else len(stats["references"])
    n_gt = sum(
        1 for s in template.get("structure", [])
        if s.get("bounds_source") == "ground_truth"
    )
    out: list[str] = [
        "Template build report:",
        f"  references included: {len(stats['references'])} (of {attempted} attempted)",
        f"  embedding model: {stats['embedding_model']}",
        f"  sections: {stats['n_sections']}  "
        f"(instrumental: {stats['n_instrumental_sections']}, "
        f"vocal: {stats['n_vocal_sections']})",
        f"    sections with non-empty audio_embedding_sequence: {stats['n_sections_with_seq']}",
        f"    bounds source: ground_truth={n_gt}, "
        f"whisper={stats['n_sections'] - n_gt}",
        f"  lines: {stats['n_lines']}",
    ]
    rates = stats["per_reference_match_rates"]
    if rates:
        bits = []
        for ref, (m, t) in rates.items():
            pct = (100 * m / t) if t else 0
            bits.append(f"{ref}: {m}/{t} ({pct:.0f}%)")
        out.append(f"    per-reference match rates: {', '.join(bits)}")
    out.append(
        f"    interpolated lines (blended): {stats['n_interpolated_lines']} / {stats['n_lines']}"
    )
    out.append(f"    unique line-level blended timings: {stats['n_unique_timings']}")
    out.append(f"    unique line-level blended embeddings: {stats['n_unique_embeddings']}")
    warns = warnings_for(stats)
    if warns:
        out.append("  ⚠ warnings:")
        for w in warns:
            out.append(f"    - {w}")
    return "\n".join(out)


def section_table(template: dict) -> str:
    """Per-section breakdown table."""
    headers = ["section_id", "type", "start_time", "end_time", "duration",
               "bounds_source", "n_lines", "n_frames", "n_unique_timings"]
    rows: list[list[str]] = [headers]
    for section in template.get("structure", []):
        lines = section.get("lines", [])
        seq = section.get("audio_embedding_sequence") or []
        unique = {
            (l.get("start_sec"), l.get("end_sec"))
            for l in lines
            if l.get("start_sec") is not None or l.get("end_sec") is not None
        }
        st = section.get("start_time")
        et = section.get("end_time")
        dur = None
        if st is not None and et is not None:
            try:
                dur = float(et) - float(st)
            except (TypeError, ValueError):
                dur = None
        rows.append([
            str(section.get("section_id", "")),
            str(section.get("section_type", "")),
            _fmt_sec(st),
            _fmt_sec(et),
            _fmt_sec(dur),
            str(section.get("bounds_source") or "-"),
            str(len(lines)),
            str(len(seq)),
            str(len(unique)),
        ])
    widths = [max(len(r[i]) for r in rows) for i in range(len(headers))]
    out_lines = []
    for n, row in enumerate(rows):
        out_lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if n == 0:
            out_lines.append("  ".join("-" * widths[i] for i in range(len(headers))))
    return "\n".join(out_lines)


def _fmt_sec(v) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return str(v)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("template", type=Path, help="Path to an *_aligned.json template")
    args = p.parse_args(argv)

    if not args.template.exists():
        print(f"No such file: {args.template}", file=sys.stderr)
        return 2

    template = json.loads(args.template.read_text())
    print(report(template))
    print()
    print("Per-section breakdown:")
    print(section_table(template))
    return 0


if __name__ == "__main__":
    sys.exit(main())
