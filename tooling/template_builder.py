"""
template_builder.py
-------------------
Builds a song template (JSON) for the live lyric teleprompter.

A template is the offline-computed song knowledge base. The live runtime
loads it and uses it as the prior for online position tracking against
your stage mic signal.

Inputs
~~~~~~
1. A structured lyrics file (.lyrics) describing the song's section structure
   and lyric text. See peggy_o.lyrics for the format.

2. A reference audio file (board tape, studio cut, or a clean live recording).
   Optional, but strongly recommended -- without audio you only get structure,
   no acoustic fingerprints or timing.

Output
~~~~~~
A JSON template with:
    - song_id, title, version_notes
    - audio_features: tempo, key estimate, beat times, global chroma
    - structure: list of sections, each with lines, elasticity flags, and
      placeholders for line-level timestamps + chroma signatures populated
      by the alignment step (see Phase 2: alignment.py).

Usage
~~~~~
    python template_builder.py \
        --lyrics peggy_o.lyrics \
        --audio peggy_o_reference.wav \
        --out peggy_o.json
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import librosa
import numpy as np


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class WordTiming:
    text: str
    start_sec: float
    end_sec: float
    confidence: float = 0.0


@dataclass
class Line:
    line_index: int
    text: str
    start_sec: Optional[float] = None       # filled by aligner
    end_sec: Optional[float] = None         # filled by aligner
    words: list = field(default_factory=list)
    chroma_signature: Optional[list] = None  # filled by aligner: 12-d vector


@dataclass
class Section:
    section_id: str
    section_type: str            # intro, verse, chorus, bridge, jam, outro, ...
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    elasticity: str = "low"      # low | medium | high
    expected_duration_range_sec: Optional[list] = None  # [min, max]
    lines: list = field(default_factory=list)
    notes: str = ""


@dataclass
class AudioFeatures:
    sample_rate: int
    duration_sec: float
    tempo_bpm: float
    key_estimate: str
    beat_times_sec: list = field(default_factory=list)
    chroma_global: Optional[list] = None    # mean chroma across full song


@dataclass
class SongTemplate:
    song_id: str
    title: str
    version_notes: str
    audio_features: Optional[AudioFeatures]
    structure: list             # list of Section


# ---------------------------------------------------------------------------
# Lyrics file parser
# ---------------------------------------------------------------------------

# Section header looks like:
#   [verse 1]
#   [chorus]
#   [jam] elasticity=high duration_range=60,300
#   [outro]
SECTION_RE = re.compile(
    r"^\[(?P<type>[a-zA-Z_]+)(?:\s+(?P<id>[^\]]+))?\](?P<attrs>.*)$"
)
ATTR_RE = re.compile(r"(\w+)=([\w\.\-,]+)")

VALID_ELASTICITY = {"low", "medium", "high"}


def _parse_attrs(s: str) -> dict:
    return dict(ATTR_RE.findall(s or ""))


def parse_lyrics(path: Path) -> list:
    """Parse a structured .lyrics file into a list of Section objects."""
    sections: list = []
    current: Optional[Section] = None
    line_counter = 0

    for raw in path.read_text().splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue

        m = SECTION_RE.match(stripped)
        if m:
            # Close the previous section
            if current is not None:
                sections.append(current)
            stype = m.group("type").lower()
            # If user provided an explicit id ("verse 1"), use it; else auto-number
            raw_id = m.group("id")
            if raw_id:
                sid = f"{stype}_{raw_id.strip().replace(' ', '_')}"
            else:
                existing_of_type = sum(
                    1 for s in sections if s.section_type == stype
                )
                sid = f"{stype}_{existing_of_type + 1}"

            attrs = _parse_attrs(m.group("attrs"))
            elasticity = attrs.get("elasticity", "low").lower()
            if elasticity not in VALID_ELASTICITY:
                elasticity = "low"

            dur_range = None
            if "duration_range" in attrs:
                parts = attrs["duration_range"].split(",")
                if len(parts) == 2:
                    dur_range = [float(parts[0]), float(parts[1])]

            current = Section(
                section_id=sid,
                section_type=stype,
                elasticity=elasticity,
                expected_duration_range_sec=dur_range,
            )
            line_counter = 0
            continue

        # Parenthetical = a director's note (instrumental, dynamics, etc.),
        # not a sung lyric line.
        if stripped.startswith("(") and stripped.endswith(")"):
            if current is not None:
                current.notes = (current.notes + " " + stripped[1:-1]).strip()
            continue

        # Otherwise: a lyric line
        if current is None:
            # Allow lyrics without an explicit opening section
            current = Section(section_id="unmarked_1", section_type="unmarked")
            line_counter = 0
        current.lines.append(Line(line_index=line_counter, text=stripped))
        line_counter += 1

    if current is not None:
        sections.append(current)

    return sections


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


def extract_audio_features(audio_path: Path) -> AudioFeatures:
    """Pull the global features the live aligner will reference."""
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = len(y) / sr

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = chroma.mean(axis=1)
    key_idx = int(np.argmax(mean_chroma))

    return AudioFeatures(
        sample_rate=sr,
        duration_sec=float(duration),
        tempo_bpm=float(np.atleast_1d(tempo)[0]),
        key_estimate=PITCH_NAMES[key_idx],
        beat_times_sec=[float(t) for t in beat_times],
        chroma_global=[float(c) for c in mean_chroma],
    )


# ---------------------------------------------------------------------------
# Build pipeline
# ---------------------------------------------------------------------------

def build_template(
    lyrics_path: Path,
    audio_path: Optional[Path],
    song_id: str,
    title: str,
    version_notes: str = "",
) -> SongTemplate:
    sections = parse_lyrics(lyrics_path)

    audio_features = None
    if audio_path is not None:
        audio_features = extract_audio_features(audio_path)
        # Phase 2 (alignment.py) will populate per-line start_sec/end_sec
        # and word timings using forced alignment on the audio.

    return SongTemplate(
        song_id=song_id,
        title=title,
        version_notes=version_notes,
        audio_features=audio_features,
        structure=sections,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build a song template for the lyric teleprompter.")
    p.add_argument("--lyrics", type=Path, required=True,
                   help="Path to the structured .lyrics file")
    p.add_argument("--audio", type=Path, default=None,
                   help="Path to reference audio (board tape, studio, etc.)")
    p.add_argument("--out", type=Path, required=True,
                   help="Path for the output JSON template")
    p.add_argument("--song-id", default=None,
                   help="Slug ID (default: lyrics filename stem)")
    p.add_argument("--title", default=None,
                   help="Display title (default: derived from song-id)")
    p.add_argument("--version-notes", default="",
                   help="Free-form notes about which arrangement/recording this represents")
    args = p.parse_args()

    song_id = args.song_id or args.lyrics.stem
    title = args.title or song_id.replace("_", " ").replace("-", " ").title()

    tmpl = build_template(
        lyrics_path=args.lyrics,
        audio_path=args.audio,
        song_id=song_id,
        title=title,
        version_notes=args.version_notes,
    )

    args.out.write_text(json.dumps(asdict(tmpl), indent=2))

    print(f"Wrote {args.out}")
    print(f"  sections: {len(tmpl.structure)}")
    print(f"  lines:    {sum(len(s.lines) for s in tmpl.structure)}")
    if tmpl.audio_features:
        af = tmpl.audio_features
        print(f"  audio:    {af.duration_sec:.1f}s @ {af.sample_rate}Hz, "
              f"tempo ~{af.tempo_bpm:.1f} bpm, dominant pitch class: {af.key_estimate}")
    else:
        print("  audio:    (no reference audio supplied)")


if __name__ == "__main__":
    main()
