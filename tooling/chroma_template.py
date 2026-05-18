"""
chroma_template.py
------------------
Compute chroma_cqt features for a song's reference audio and stamp them
on the template's JSON. Used by tooling/dtw_align.py as the
mix-invariant signal for online position tracking.

The brief specifies online DTW on chroma as the primary signal, with
MERT embeddings as a secondary timbral check. MERT-only ran at 21% raw
/ 55% smoothed on band audio because timbre doesn't generalise across
mixes; chroma (per-pitch-class energy) is mix-invariant. This script
adds the chroma side to existing templates non-destructively -- the
MERT fields stay put during the transition.

CLI:
    python tooling/chroma_template.py \\
        --template tooling/songs/peggy_o_aligned.json \\
        [--audio path/to/reference.mp3]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import librosa
import numpy as np


log = logging.getLogger("chroma_template")

DEFAULT_FRAMES_PER_SEC = 10.0
DEFAULT_SAMPLE_RATE = 22050
CHROMA_DECIMALS = 4


def _resolve_reference_audio(template: dict, template_path: Path) -> Optional[Path]:
    """Locate the reference audio file recorded in template["references"][0].

    Build_song.py writes references to ``tooling/references/<song_id>/<basename>``.
    We try that path, plus a couple of slug variants, plus treating the
    reference field as an absolute or cwd-relative path.
    """
    refs = template.get("references") or []
    if not refs:
        return None
    name = refs[0]
    candidates: list[Path] = []
    raw = Path(name)
    candidates.append(raw)
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)

    tooling_dir = Path(__file__).resolve().parent
    song_id = (template.get("song_id") or "").strip()
    for variant in {song_id, song_id.replace("-", "_"), song_id.replace("_", "-")}:
        if variant:
            candidates.append(tooling_dir / "references" / variant / name)

    for c in candidates:
        if c.exists():
            return c
    return None


def compute_chroma(audio_path: Path,
                   sample_rate: int = DEFAULT_SAMPLE_RATE,
                   frames_per_sec: float = DEFAULT_FRAMES_PER_SEC
                   ) -> tuple[np.ndarray, int, int, float]:
    """Return (chroma_matrix, sample_rate, hop_length, fps).

    chroma_matrix has shape (n_frames, 12) so each row is one frame --
    convenient for both JSON serialisation and downstream consumers.
    """
    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    duration = len(y) / sr
    log.info("Loading audio from %s (%.1fs)", audio_path, duration)
    hop_length = max(1, int(round(sr / frames_per_sec)))
    fps = sr / hop_length
    log.info("Computing chroma_cqt at %.1f frames/sec...", fps)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    # librosa returns (n_chroma=12, n_frames). Transpose so each row is a frame.
    return chroma.T.astype(np.float32), sr, hop_length, fps


def stamp_section_frames(template: dict, sample_rate: int,
                         hop_length: int) -> int:
    """Attach chroma_start_frame / chroma_end_frame to each section based
    on its start_time / end_time. Returns the number of sections stamped.
    """
    fps = sample_rate / hop_length
    n = 0
    for section in template.get("structure", []):
        st = section.get("start_time")
        et = section.get("end_time")
        if st is None or et is None:
            section.pop("chroma_start_frame", None)
            section.pop("chroma_end_frame", None)
            continue
        section["chroma_start_frame"] = int(round(float(st) * fps))
        section["chroma_end_frame"] = int(round(float(et) * fps))
        n += 1
    return n


def write_chroma_into_template(template: dict, chroma_matrix: np.ndarray,
                               sample_rate: int, hop_length: int) -> None:
    """Idempotent: replaces any prior chroma_reference field."""
    fps = sample_rate / hop_length
    template["chroma_reference"] = {
        "feature_type": "chroma_cqt",
        "sample_rate": int(sample_rate),
        "hop_length": int(hop_length),
        "frames_per_sec": float(fps),
        "n_frames": int(chroma_matrix.shape[0]),
        "data": [
            [round(float(v), CHROMA_DECIMALS) for v in row]
            for row in chroma_matrix
        ],
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--template", type=Path, required=True,
                   help="Path to an *_aligned.json template (will be updated "
                        "in place).")
    p.add_argument("--audio", type=Path, default=None,
                   help="Reference audio path. Defaults to looking up the "
                        "template's references[0] under tooling/references/<song_id>/.")
    p.add_argument("--sample-rate", type=int, default=None,
                   help="Override the chroma sample rate. Defaults to the "
                        "template's audio_features.sample_rate or 22050.")
    p.add_argument("--frames-per-sec", type=float, default=DEFAULT_FRAMES_PER_SEC,
                   help="Target chroma frame rate (default 10).")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    template = json.loads(args.template.read_text())

    audio_path = args.audio
    if audio_path is None:
        audio_path = _resolve_reference_audio(template, args.template)
    if audio_path is None or not audio_path.exists():
        log.error(
            "Could not find a reference audio file. Pass --audio explicitly, "
            "or place the file at tooling/references/<song_id>/<references[0]>.",
        )
        return 1

    sr = args.sample_rate
    if sr is None:
        sr = int((template.get("audio_features") or {}).get("sample_rate")
                 or DEFAULT_SAMPLE_RATE)

    chroma_matrix, sr_used, hop_length, fps = compute_chroma(
        audio_path, sample_rate=sr, frames_per_sec=args.frames_per_sec,
    )
    write_chroma_into_template(template, chroma_matrix, sr_used, hop_length)
    log.info("Stored %d chroma frames in template", chroma_matrix.shape[0])

    n_stamped = stamp_section_frames(template, sr_used, hop_length)
    log.info("Stamped chroma frame indices on %d sections", n_stamped)

    args.template.write_text(json.dumps(template, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
