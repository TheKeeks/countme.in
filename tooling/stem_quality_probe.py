"""
stem_quality_probe.py
---------------------
Probe: is a LOWER-QUALITY vocal separator still good enough for the
energy-onset detector to find verse_5?

The known-good re-anchoring path runs Demucs (htdemucs) and then energy-onset
detection on the vocal stem -- it finds verse_5 at ~280s on the band
recording. But Demucs is offline-only; a live in-browser architecture would
have to use a much lighter separator (real-time / browser-grade quality).
This probe gates the whole separate-then-detect live architecture: it runs
the SAME energy-onset detection on two stems of very different quality and
compares them head to head.

  demucs      htdemucs two-stem vocals -- the high-quality reference.
  open_unmix  Open-Unmix umxhq vocals -- a lighter, lower-quality separator
              standing in for real-time/browser-grade separation.

Both separators get the identical ffmpeg-normalised 44.1 kHz stereo input.
Each stem then goes through the existing pipeline (imported, not copied,
from detect_first_verse / raw_onset_probe): RMS in 100ms windows -> dB ->
1.0s moving average, auto threshold from the post-offset baseline window
(median + offset, clamped), and dip-tolerant sustained rising-edge crossings
over the whole post-offset envelope. The report scores every detected onset
against ground-truth section starts, computes the vocal-vs-jam energy
contrast (mean verse dB minus mean jam dB -- how much headroom the detector
has), and gives the verdict: does the proxy separator still find verse_5
within +-5s, and how much contrast does it lose vs Demucs?

This module is intentionally standalone -- a feasibility experiment, not
integrated into the validator or other probes.

CLI:

    python tooling/stem_quality_probe.py \\
        --audio path/to/band_recording.mp3 \\
        --ground-truth tooling/ground_truth/peggy_o_band.json \\
        --song-offset 20.0 \\
        --output-md stem_quality.md \\
        --output-json stem_quality.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse the existing energy pipeline rather than re-implementing it: the
# whole point is to run the SAME detection on both stems.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import (  # noqa: E402
    ENERGY_FRAME_MS,
    ENERGY_SMOOTHING_SEC,
    _separate_vocals,
    _smoothed_envelope,
    _vocal_stem_rms_db,
)
from raw_onset_probe import (  # noqa: E402
    ANCHOR_SECTIONS,
    VERDICT_TOLERANCE_SEC,
    _detect,
    _envelope_to_json,
    _fmt,
    _fmt_signed,
    _load_sections,
    _method_markdown,
    _score_sections,
    _verdict,
)

# Both separators are trained at 44.1 kHz; feed them the identical input.
SEPARATION_SAMPLE_RATE = 44100

SEPARATORS = ("demucs", "open_unmix")
SEPARATOR_LABELS = {
    "demucs": "Demucs (reference)",
    "open_unmix": "Open-Unmix (proxy)",
}
OPENUNMIX_MODEL = "umxhq"

# Detection knobs -- same values raw_onset_probe uses, so stem results are
# directly comparable to the no-separation probe too.
BASELINE_WINDOW_SEC = 8.0
AUTO_OFFSET_DB = 20.0
SUSTAIN_WINDOW_SEC = 2.0
SUSTAIN_TOLERANCE_DB = 5.0


# ---------------------------------------------------------------------------
# Separation
# ---------------------------------------------------------------------------

def _ensure_separation_wav(audio_path: Path) -> Path:
    """Pre-convert the input to 44.1 kHz stereo WAV via ffmpeg so both
    separators see exactly the same signal (and extensionless rehearsal
    uploads decode cleanly). Caller unlinks the result.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
         "-ar", str(SEPARATION_SAMPLE_RATE), "-ac", "2", "-f", "wav",
         str(wav_path)],
        check=True,
    )
    return wav_path


def _separate_vocals_openunmix(wav_path: Path) -> tuple[Path, Path, float]:
    """Open-Unmix vocal stem. Returns (vocals_wav, cleanup_dir, rms_db),
    mirroring detect_first_verse._separate_vocals so the two separators are
    interchangeable downstream.

    Audio I/O goes through soundfile rather than Open-Unmix's torchaudio CLI
    path -- the input is already a clean 44.1 kHz WAV, and this avoids
    torchaudio backend quirks on CI. residual=True because Open-Unmix's
    Wiener EM step needs at least two sources; the residual is discarded.
    """
    import soundfile as sf
    import torch
    from openunmix import predict

    out_dir = Path(tempfile.mkdtemp(prefix="umx_"))
    print(f"  running Open-Unmix ({OPENUNMIX_MODEL}, vocals) ...", flush=True)
    y, sr = sf.read(str(wav_path), always_2d=True)  # (samples, channels)
    audio = torch.from_numpy(np.ascontiguousarray(y.T).astype(np.float32))
    estimates = predict.separate(
        audio,
        rate=sr,
        model_str_or_path=OPENUNMIX_MODEL,
        targets=["vocals"],
        residual=True,
        device="cpu",
    )
    vocals = estimates["vocals"][0].cpu().numpy().T  # (samples, channels)
    vocals_path = out_dir / "vocals.wav"
    sf.write(str(vocals_path), vocals, SEPARATION_SAMPLE_RATE)
    rms_db = _vocal_stem_rms_db(vocals_path)
    return vocals_path, out_dir, rms_db


# ---------------------------------------------------------------------------
# Envelope + contrast
# ---------------------------------------------------------------------------

def _trim_envelope_edges(times: np.ndarray, db: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Drop the frames where the 1.0s moving average zero-pads past the
    signal edge (same fix raw_onset_probe applies): dB values are negative,
    so the padding drags edge frames toward 0 and fakes a loud blip.
    """
    window = max(1, int(round(ENERGY_SMOOTHING_SEC / (ENERGY_FRAME_MS / 1000.0))))
    half = window // 2
    if half and len(times) > 2 * half:
        return times[half:-half], db[half:-half]
    return times, db


def _vocal_jam_contrast(times: np.ndarray, db: np.ndarray,
                        sections: list[dict]) -> dict:
    """Mean stem energy over the verse sections minus mean over the jam
    sections. This is the headroom the onset detector lives on: a clean
    stem is loud during verses and near-silent during the jam; a leaky
    stem lets the jam bleed through and the contrast collapses.
    """
    verse_mask = np.zeros(len(times), dtype=bool)
    jam_mask = np.zeros(len(times), dtype=bool)
    for s in sections:
        sid = str(s["section_id"])
        mask = (times >= float(s["start"])) & (times < float(s["end"]))
        if sid.startswith("verse"):
            verse_mask |= mask
        elif sid.startswith("jam"):
            jam_mask |= mask
    verse_db = float(np.mean(db[verse_mask])) if verse_mask.any() else None
    jam_db = float(np.mean(db[jam_mask])) if jam_mask.any() else None
    contrast = (verse_db - jam_db
                if verse_db is not None and jam_db is not None else None)
    return {
        "verse_mean_db": verse_db,
        "jam_mean_db": jam_db,
        "contrast_db": contrast,
    }


# ---------------------------------------------------------------------------
# Per-separator pipeline
# ---------------------------------------------------------------------------

def _run_separator(name: str, wav_path: Path, sections: list[dict],
                   song_offset: float) -> dict:
    print(f"Separator {name}: separate + envelope + detection ...", flush=True)
    if name == "demucs":
        vocals_path, cleanup_dir, rms_db = _separate_vocals(wav_path)
    else:
        vocals_path, cleanup_dir, rms_db = _separate_vocals_openunmix(wav_path)
    try:
        print(f"  vocal stem RMS: {rms_db:.1f} dB", flush=True)
        times, smoothed = _smoothed_envelope(vocals_path)
        times, smoothed = _trim_envelope_edges(times, smoothed)
        result = _detect(
            times, smoothed, song_offset, BASELINE_WINDOW_SEC,
            AUTO_OFFSET_DB, SUSTAIN_WINDOW_SEC, SUSTAIN_TOLERANCE_DB,
        )
        result["stem_rms_db"] = rms_db
        result["contrast"] = _vocal_jam_contrast(times, smoothed, sections)
        result["section_errors"] = _score_sections(sections, result["onsets_sec"])
        for sid in ANCHOR_SECTIONS:
            result[f"{sid}_verdict"] = _verdict(result["section_errors"], sid)
        result["envelope"] = _envelope_to_json(times, smoothed)
    finally:
        shutil.rmtree(cleanup_dir, ignore_errors=True)

    contrast = result["contrast"]["contrast_db"]
    print(
        f"  baseline {_fmt(result['baseline_db'], ' dB')}, "
        f"threshold {result['threshold_db']:.1f} dB, "
        f"{len(result['onsets_sec'])} onsets, "
        f"vocal-jam contrast {_fmt(contrast, ' dB')}",
        flush=True,
    )
    for sid in ANCHOR_SECTIONS:
        v = result[f"{sid}_verdict"]
        status = f"yes, error {v['error_sec']:+.2f}s" if v["detected"] else "no"
        print(f"  {sid} detected: {status}", flush=True)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _onset_error_cell(verdict: dict) -> str:
    return (f"{_fmt(verdict['nearest_onset_sec'])} / "
            f"{_fmt_signed(verdict['error_sec'])}")


def _comparison_table(results: dict[str, dict]) -> list[str]:
    lines = [
        "| separator | vocal-jam contrast (dB) | verse_1 onset / error "
        "| verse_5 onset / error | total onsets |",
        "|---|---|---|---|---|",
    ]
    for name in SEPARATORS:
        r = results[name]
        contrast = r["contrast"]["contrast_db"]
        lines.append(
            f"| {SEPARATOR_LABELS[name]} "
            f"| {_fmt(contrast, '')} "
            f"| {_onset_error_cell(r['verse_1_verdict'])} "
            f"| {_onset_error_cell(r['verse_5_verdict'])} "
            f"| {len(r['onsets_sec'])} |"
        )
    return lines


def _verdict_lines(results: dict[str, dict]) -> list[str]:
    proxy = results["open_unmix"]
    proxy_v5 = proxy["verse_5_verdict"]
    proxy_contrast = proxy["contrast"]["contrast_db"]
    demucs_contrast = results["demucs"]["contrast"]["contrast_db"]

    lines = ["## VERDICT", ""]
    if proxy_v5["detected"]:
        lines.append(
            f"- Open-Unmix (proxy for real-time/browser-grade separation) "
            f"detects verse_5 within ±{VERDICT_TOLERANCE_SEC:.0f}s? "
            f"**yes** — onset at {proxy_v5['nearest_onset_sec']:.2f}s, "
            f"error {proxy_v5['error_sec']:+.2f}s"
        )
    else:
        miss = ("no onsets found" if proxy_v5["error_sec"] is None else
                f"nearest onset off by {proxy_v5['error_sec']:+.2f}s, "
                f"tolerance ±{VERDICT_TOLERANCE_SEC:.0f}s")
        lines.append(
            f"- Open-Unmix (proxy for real-time/browser-grade separation) "
            f"detects verse_5 within ±{VERDICT_TOLERANCE_SEC:.0f}s? "
            f"**no** — {miss}"
        )

    if proxy_contrast is not None and demucs_contrast is not None:
        delta = proxy_contrast - demucs_contrast
        lines.append(
            f"- Vocal-jam contrast: Open-Unmix {proxy_contrast:.1f} dB vs "
            f"Demucs {demucs_contrast:.1f} dB ({delta:+.1f} dB vs reference)"
        )
    else:
        lines.append(
            f"- Vocal-jam contrast: Open-Unmix {_fmt(proxy_contrast, ' dB')} "
            f"vs Demucs {_fmt(demucs_contrast, ' dB')} (not comparable)"
        )

    if proxy_v5["detected"]:
        lines.append(
            "- Gate: a lower-quality separator keeps enough vocal-vs-jam "
            "contrast for energy-onset re-anchoring — the "
            "separate-then-detect live architecture is NOT blocked on "
            "Demucs-grade separation."
        )
    else:
        lines.append(
            "- Gate: lighter separation does not surface verse_5 — the "
            "separate-then-detect live architecture needs better real-time "
            "separation quality or a different detection signal."
        )
    lines.append("")
    return lines


def _separator_json(result: dict) -> dict:
    return {
        "stem_rms_db": result["stem_rms_db"],
        "baseline_db": result["baseline_db"],
        "threshold_db": result["threshold_db"],
        "onsets_sec": result["onsets_sec"],
        "above_threshold_intervals": [
            [s, e] for s, e in result["above_threshold_intervals"]
        ],
        "contrast": result["contrast"],
        "section_errors": result["section_errors"],
        "verse_1_verdict": result["verse_1_verdict"],
        "verse_5_verdict": result["verse_5_verdict"],
        "envelope": result["envelope"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True,
                   help="Ground-truth JSON with section starts "
                        "(e.g. tooling/ground_truth/peggy_o_band.json).")
    p.add_argument("--song-offset", type=float, default=20.0,
                   help="Ignore audio before this time -- pre-song noodling "
                        "(default 20.0).")
    p.add_argument("--output-md", type=Path, default=Path("stem_quality.md"))
    p.add_argument("--output-json", type=Path, default=Path("stem_quality.json"))
    args = p.parse_args(argv)

    sections = _load_sections(args.ground_truth)

    print(f"Converting {args.audio} to {SEPARATION_SAMPLE_RATE} Hz stereo WAV "
          f"for separation ...", flush=True)
    wav_path = _ensure_separation_wav(args.audio)
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        duration = float(info.frames) / float(info.samplerate)
        print(f"  {duration:.1f}s @ {info.samplerate} Hz", flush=True)

        results: dict[str, dict] = {}
        for name in SEPARATORS:
            results[name] = _run_separator(
                name, wav_path, sections, args.song_offset,
            )
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    # Markdown report.
    md = [
        "# Stem-quality feasibility probe (Demucs vs Open-Unmix)",
        "",
        f"- Audio: `{args.audio}` ({duration:.1f}s @ "
        f"{SEPARATION_SAMPLE_RATE} Hz)",
        f"- Song offset: {args.song_offset:.1f}s",
        f"- Ground truth: `{args.ground_truth}` "
        f"({len(sections)} scored sections)",
        f"- Separators: Demucs htdemucs two-stem vocals (reference) vs "
        f"Open-Unmix {OPENUNMIX_MODEL} vocals (real-time/browser-grade "
        f"proxy), identical input and detection",
        f"- Detection: {ENERGY_FRAME_MS}ms RMS → dB → "
        f"{ENERGY_SMOOTHING_SEC:.1f}s smoothing; threshold = baseline median "
        f"(first {BASELINE_WINDOW_SEC:.1f}s post-offset) + "
        f"{AUTO_OFFSET_DB:.1f} dB; sustained crossing = "
        f"{SUSTAIN_WINDOW_SEC:.1f}s window median within "
        f"{SUSTAIN_TOLERANCE_DB:.1f} dB of threshold",
        "",
    ]
    md += _comparison_table(results)
    md.append("")
    md += _verdict_lines(results)
    for name in SEPARATORS:
        md += _method_markdown(SEPARATOR_LABELS[name], results[name])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))
    print(f"Wrote markdown report to {args.output_md}", flush=True)

    # JSON report.
    report = {
        "audio": str(args.audio),
        "duration_sec": duration,
        "separation_sample_rate": SEPARATION_SAMPLE_RATE,
        "song_offset": args.song_offset,
        "ground_truth": str(args.ground_truth),
        "openunmix_model": OPENUNMIX_MODEL,
        "baseline_window_sec": BASELINE_WINDOW_SEC,
        "auto_offset_db": AUTO_OFFSET_DB,
        "sustain_window_sec": SUSTAIN_WINDOW_SEC,
        "sustain_tolerance_db": SUSTAIN_TOLERANCE_DB,
        "verdict_tolerance_sec": VERDICT_TOLERANCE_SEC,
        "separators": {
            name: _separator_json(results[name]) for name in SEPARATORS
        },
        "comparison": {
            "proxy_verse_5_detected":
                results["open_unmix"]["verse_5_verdict"]["detected"],
            "demucs_contrast_db":
                results["demucs"]["contrast"]["contrast_db"],
            "open_unmix_contrast_db":
                results["open_unmix"]["contrast"]["contrast_db"],
            "contrast_delta_db": (
                results["open_unmix"]["contrast"]["contrast_db"]
                - results["demucs"]["contrast"]["contrast_db"]
                if results["open_unmix"]["contrast"]["contrast_db"] is not None
                and results["demucs"]["contrast"]["contrast_db"] is not None
                else None
            ),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote JSON report to {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
