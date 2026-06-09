"""
raw_onset_probe.py
------------------
Probe: are vocal onsets detectable on RAW phone-mic audio, without Demucs?

The current re-anchoring path (detect_first_verse.py energy mode) depends
on Demucs vocal isolation, which cannot run live on a phone. This is a
de-risking experiment for the live app: detect onsets on the raw mixed
signal with two cheap methods and score them against ground-truth vocal
section starts. NO source separation anywhere in this script.

Methods (both: RMS in 100ms windows -> dB -> 1.0s moving average):
  full_spectrum  envelope of the raw signal as-is.
  vocal_band     Butterworth bandpass (default 250-3000 Hz) first, then the
                 envelope. Removing bass/kick and high cymbal energy should
                 suppress the instrumental jam relative to the vocal, which
                 sits in the midrange and is proximity-boosted on a vocal mic.

The two onsets that matter most are verse_1 (after the intro) and verse_5
(after the jam) -- the vocal entries that follow instrumental sections and
that live re-anchoring depends on. The above-threshold intervals are
reported too: if the jam and verse_5 merge into one continuous blob the
method fails; if they separate into distinct regions it works.

CLI:

    python tooling/raw_onset_probe.py \\
        --audio path/to/band_recording.mp3 \\
        --ground-truth tooling/ground_truth/peggy_o_band.json \\
        --song-offset 20.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Reuse helpers that live next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav, _moving_average  # noqa: E402

# Repo audio convention (template_builder / chroma_template).
SAMPLE_RATE = 22050

FRAME_MS = 100
SMOOTHING_SEC = 1.0
THRESHOLD_MIN_DB = -50.0
THRESHOLD_MAX_DB = -10.0
BANDPASS_ORDER = 4
# An onset within this distance of a ground-truth section start counts as
# "detected" for the verse_1 / verse_5 verdicts.
VERDICT_TOLERANCE_SEC = 5.0

METHODS = ("full_spectrum", "vocal_band")


# ---------------------------------------------------------------------------
# Signal -> envelope
# ---------------------------------------------------------------------------

def _bandpass(y: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Zero-phase Butterworth bandpass. Second-order sections rather than
    (b, a) polynomials -- numerically stable for a narrow band at 22 kHz.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(BANDPASS_ORDER, [low_hz, high_hz], btype="bandpass",
                 fs=sr, output="sos")
    return sosfiltfilt(sos, y)


def _envelope(y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS in FRAME_MS windows -> dBFS -> SMOOTHING_SEC moving average.

    Same shape as detect_first_verse._energy_envelope, but on an in-memory
    array (that helper reads the Demucs stem from disk) so the bandpassed
    signal doesn't need a temp file. Returns (times_sec, db).
    """
    frame_samples = int(sr * FRAME_MS / 1000)
    n_frames = len(y) // frame_samples
    if n_frames == 0:
        return np.zeros(0), np.zeros(0)
    trimmed = y[: n_frames * frame_samples].reshape(n_frames, frame_samples)
    rms = np.sqrt(np.mean(trimmed.astype(np.float64) ** 2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    times = np.arange(n_frames) * (FRAME_MS / 1000.0)
    smoothing_window = max(1, int(round(SMOOTHING_SEC / (FRAME_MS / 1000.0))))
    smoothed = _moving_average(db, smoothing_window)
    # Drop the frames where the moving average zero-pads past the signal
    # edge: dB values are negative, so the padding drags them toward 0 and
    # fakes a loud blip at the very start and end of the envelope.
    half = smoothing_window // 2
    if half and n_frames > 2 * half:
        times, smoothed = times[half:-half], smoothed[half:-half]
    return times, smoothed


# ---------------------------------------------------------------------------
# Auto-threshold onset detection
# ---------------------------------------------------------------------------

def _detect(times: np.ndarray, db: np.ndarray, song_offset: float,
            baseline_window_sec: float, auto_offset_db: float,
            sustain_window_sec: float, sustain_tolerance_db: float) -> dict:
    """Auto-threshold detection across the whole post-offset envelope.

    baseline_db = median dB in [song_offset, song_offset + baseline_window_sec]
    (assumed instrumental intro, no vocals). threshold = baseline + offset,
    clamped. Reports ALL sustained rising-edge crossings (median of the next
    sustain_window_sec above threshold - tolerance, same dip-tolerant rule as
    detect_first_verse) plus the raw above-threshold intervals.
    """
    mask = (times >= song_offset) & (times < song_offset + baseline_window_sec)
    baseline_window = db[mask]
    baseline_db = float(np.median(baseline_window)) if len(baseline_window) else None
    threshold_db = (
        float(np.clip(baseline_db + auto_offset_db,
                      THRESHOLD_MIN_DB, THRESHOLD_MAX_DB))
        if baseline_db is not None else THRESHOLD_MIN_DB
    )

    dt = float(times[1] - times[0]) if len(times) > 1 else FRAME_MS / 1000.0
    window_frames = max(1, int(np.ceil(sustain_window_sec / dt)))
    median_floor = threshold_db - sustain_tolerance_db
    n = len(times)

    onsets: list[float] = []
    intervals: list[tuple[float, float]] = []
    interval_start: Optional[float] = None
    # Pre-offset state counts as "below" so audio already loud at song_offset
    # still produces a rising edge there.
    prev_below = True
    for i in range(n):
        if times[i] < song_offset:
            continue
        above = bool(db[i] >= threshold_db)
        if above and prev_below:
            interval_start = float(times[i])
            window = db[i: min(i + window_frames, n)]
            if len(window) > 0 and float(np.median(window)) >= median_floor:
                onsets.append(float(times[i]))
        elif not above and not prev_below and interval_start is not None:
            intervals.append((interval_start, float(times[i])))
            interval_start = None
        prev_below = not above
    if interval_start is not None and n > 0:
        intervals.append((interval_start, float(times[n - 1])))

    return {
        "baseline_db": baseline_db,
        "threshold_db": threshold_db,
        "onsets_sec": onsets,
        "above_threshold_intervals": intervals,
    }


# ---------------------------------------------------------------------------
# Ground-truth comparison
# ---------------------------------------------------------------------------

ANCHOR_SECTIONS = ("verse_1", "verse_5")


def _load_sections(ground_truth_path: Path) -> list[dict]:
    """Section dicts from the ground-truth file, skipping the
    underscore-prefixed ones (pre-song noodling, post-song chatter) that
    the ground truth itself excludes from accuracy.
    """
    gt = json.loads(ground_truth_path.read_text())
    return [s for s in gt.get("sections", [])
            if not str(s.get("section_id", "")).startswith("_")]


def _score_sections(sections: list[dict], onsets: list[float]) -> list[dict]:
    rows = []
    for s in sections:
        gt_start = float(s["start"])
        nearest: Optional[float] = None
        error: Optional[float] = None
        if onsets:
            nearest = min(onsets, key=lambda o: abs(o - gt_start))
            error = nearest - gt_start
        rows.append({
            "section_id": s["section_id"],
            "gt_start_sec": gt_start,
            "nearest_onset_sec": nearest,
            "error_sec": error,
            "is_anchor": s["section_id"] in ANCHOR_SECTIONS,
        })
    return rows


def _verdict(rows: list[dict], section_id: str) -> dict:
    row = next((r for r in rows if r["section_id"] == section_id), None)
    if row is None or row["error_sec"] is None:
        return {"section_id": section_id, "detected": False,
                "nearest_onset_sec": None, "error_sec": None}
    detected = abs(row["error_sec"]) <= VERDICT_TOLERANCE_SEC
    return {
        "section_id": section_id,
        "detected": detected,
        "nearest_onset_sec": row["nearest_onset_sec"],
        "error_sec": row["error_sec"],
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float], suffix: str = "s") -> str:
    return "-" if value is None else f"{value:.2f}{suffix}"


def _fmt_signed(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:+.2f}s"


def _method_markdown(name: str, result: dict) -> list[str]:
    lines = [f"## Method: {name}", ""]
    baseline = result["baseline_db"]
    lines.append(f"- baseline_db: "
                 f"{'-' if baseline is None else f'{baseline:.1f} dB'}")
    lines.append(f"- threshold_db: {result['threshold_db']:.1f} dB")
    onsets = ", ".join(f"{t:.1f}s" for t in result["onsets_sec"]) or "(none)"
    lines.append(f"- Detected onsets ({len(result['onsets_sec'])}): {onsets}")
    intervals = ", ".join(
        f"({s:.1f}s, {e:.1f}s)" for s, e in result["above_threshold_intervals"]
    ) or "(none)"
    lines.append(
        f"- Above-threshold intervals "
        f"({len(result['above_threshold_intervals'])}): {intervals}"
    )
    lines += ["", "Onset vs ground truth:", "",
              "| section | gt_start | nearest_onset | error |",
              "|---|---|---|---|"]
    for row in result["section_errors"]:
        sid = f"**{row['section_id']}**" if row["is_anchor"] else row["section_id"]
        lines.append(
            f"| {sid} | {_fmt(row['gt_start_sec'])} "
            f"| {_fmt(row['nearest_onset_sec'])} "
            f"| {_fmt_signed(row['error_sec'])} |"
        )
    lines.append("")
    for sid in ANCHOR_SECTIONS:
        v = result[f"{sid}_verdict"]
        if v["detected"]:
            lines.append(
                f"- VERDICT: {sid} onset detected? **yes** — "
                f"error {v['error_sec']:+.2f}s "
                f"(onset at {v['nearest_onset_sec']:.2f}s)"
            )
        else:
            miss = ("no onsets found" if v["error_sec"] is None else
                    f"nearest onset off by {v['error_sec']:+.2f}s, "
                    f"tolerance ±{VERDICT_TOLERANCE_SEC:.0f}s")
            lines.append(f"- VERDICT: {sid} onset detected? **no** — {miss}")
    lines.append("")
    return lines


def _envelope_to_json(times: np.ndarray, db: np.ndarray) -> list[dict]:
    return [{"time": round(float(t), 2), "db": round(float(d), 2)}
            for t, d in zip(times, db)]


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
    p.add_argument("--output-md", type=Path, default=Path("raw_onset.md"))
    p.add_argument("--output-json", type=Path, default=Path("raw_onset.json"))
    p.add_argument("--vocal-band-low", type=float, default=250.0,
                   help="Bandpass low edge in Hz (default 250).")
    p.add_argument("--vocal-band-high", type=float, default=3000.0,
                   help="Bandpass high edge in Hz (default 3000).")
    p.add_argument("--auto-offset-db", type=float, default=20.0,
                   help="Offset above the baseline median for the threshold "
                        f"(default 20.0, clamped to [{THRESHOLD_MIN_DB:.0f}, "
                        f"{THRESHOLD_MAX_DB:.0f}] dB).")
    p.add_argument("--sustain-window-sec", type=float, default=2.0,
                   help="Post-crossing window whose median must stay above "
                        "the tolerance line (default 2.0).")
    p.add_argument("--sustain-tolerance-db", type=float, default=5.0,
                   help="Allowed dip below threshold for the post-crossing "
                        "median (default 5.0).")
    p.add_argument("--baseline-window-sec", type=float, default=8.0,
                   help="Length of the post-offset window used for the "
                        "baseline median (default 8.0). Assumed to be "
                        "instrumental intro -- no vocals yet.")
    args = p.parse_args(argv)

    import librosa

    # Pre-convert via ffmpeg like detect_first_verse does -- sidesteps the
    # librosa+audioread quirks on extensionless rehearsal MP3s.
    wav_path = _ensure_wav(args.audio, target_sr=SAMPLE_RATE)
    try:
        print(f"Loading {args.audio} (raw, no separation) ...", flush=True)
        y, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
    duration = len(y) / sr
    print(f"  {duration:.1f}s @ {sr} Hz", flush=True)

    sections = _load_sections(args.ground_truth)

    signals = {
        "full_spectrum": y,
        "vocal_band": _bandpass(y, sr, args.vocal_band_low, args.vocal_band_high),
    }

    results: dict[str, dict] = {}
    for name in METHODS:
        print(f"Method {name}: envelope + auto-threshold detection ...",
              flush=True)
        times, db = _envelope(signals[name], sr)
        result = _detect(
            times, db, args.song_offset, args.baseline_window_sec,
            args.auto_offset_db, args.sustain_window_sec,
            args.sustain_tolerance_db,
        )
        result["section_errors"] = _score_sections(sections, result["onsets_sec"])
        for sid in ANCHOR_SECTIONS:
            result[f"{sid}_verdict"] = _verdict(result["section_errors"], sid)
        result["envelope"] = _envelope_to_json(times, db)
        results[name] = result
        print(
            f"  baseline {result['baseline_db']:.1f} dB, "
            f"threshold {result['threshold_db']:.1f} dB, "
            f"{len(result['onsets_sec'])} onsets, "
            f"{len(result['above_threshold_intervals'])} intervals",
            flush=True,
        )
        for sid in ANCHOR_SECTIONS:
            v = result[f"{sid}_verdict"]
            status = (f"yes, error {v['error_sec']:+.2f}s" if v["detected"]
                      else "no")
            print(f"  {sid} detected: {status}", flush=True)

    # Markdown report.
    md = [
        "# Raw-mic onset probe (no Demucs)",
        "",
        f"- Audio: `{args.audio}` ({duration:.1f}s @ {sr} Hz)",
        f"- Song offset: {args.song_offset:.1f}s",
        f"- Vocal band: {args.vocal_band_low:.0f}-{args.vocal_band_high:.0f} Hz",
        f"- Ground truth: `{args.ground_truth}` "
        f"({len(sections)} scored sections)",
        "",
    ]
    for name in METHODS:
        md += _method_markdown(name, results[name])
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))
    print(f"Wrote markdown report to {args.output_md}", flush=True)

    # JSON report.
    report = {
        "audio": str(args.audio),
        "duration_sec": duration,
        "sample_rate": sr,
        "song_offset": args.song_offset,
        "ground_truth": str(args.ground_truth),
        "vocal_band_low_hz": args.vocal_band_low,
        "vocal_band_high_hz": args.vocal_band_high,
        "auto_offset_db": args.auto_offset_db,
        "sustain_window_sec": args.sustain_window_sec,
        "sustain_tolerance_db": args.sustain_tolerance_db,
        "baseline_window_sec": args.baseline_window_sec,
        "verdict_tolerance_sec": VERDICT_TOLERANCE_SEC,
        "methods": {
            name: {
                "baseline_db": r["baseline_db"],
                "threshold_db": r["threshold_db"],
                "onsets_sec": r["onsets_sec"],
                "above_threshold_intervals": [
                    [s, e] for s, e in r["above_threshold_intervals"]
                ],
                "section_errors": r["section_errors"],
                "verse_1_verdict": r["verse_1_verdict"],
                "verse_5_verdict": r["verse_5_verdict"],
                "envelope": r["envelope"],
            }
            for name, r in results.items()
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote JSON report to {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
