"""
vad_probe.py
------------
Probe: can Silero VAD detect SUNG vocals over a loud band on raw
phone-mic audio, with no source separation?

Rung 2 of the live vocal-onset ladder. Rung 1 (raw_onset_probe.py:
energy / bandpass envelopes) failed -- vocal-minus-jam energy contrast
was ~0 dB because the band dominates the raw mix. This rung tests a
lightweight neural voice-activity detector (~2MB, CPU-fast, live-viable)
instead: run Silero across the whole file, collect the per-window speech
probability, and measure the CONTRAST between vocal sections (verse_*)
and the instrumental jam. NO Demucs, no separation of any kind.

The model is fed consecutive 512-sample windows (~32ms at 16kHz) in
order, exactly as a live stream would, so the stateful LSTM sees the
same input it would on a phone.

The headline number is CONTRAST = mean vocal P(speech) - jam P(speech).
The critical detection case is verse_5 -- the vocal re-entry after the
jam that live re-anchoring depends on.

CLI:

    python tooling/vad_probe.py \\
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

# Sibling helpers (light imports; heavy deps in those modules are lazy).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav, _moving_average  # noqa: E402
from raw_onset_probe import _fmt, _fmt_signed, _load_sections  # noqa: E402

# Silero's required input format.
VAD_SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512  # ~32ms

SUSTAIN_SEC = 1.5
SWEEP_THRESHOLDS = (0.3, 0.5, 0.7)
ANCHOR_SECTIONS = ("verse_1", "verse_5")


# ---------------------------------------------------------------------------
# Model + probability curve
# ---------------------------------------------------------------------------

def _load_model() -> tuple[object, str]:
    """Silero VAD on CPU. ONNX runtime preferred (faster, and the backend
    a phone deployment would use); torch JIT is the fallback.
    """
    from silero_vad import load_silero_vad

    try:
        return load_silero_vad(onnx=True), "onnx"
    except Exception as exc:  # noqa: BLE001 - any onnx failure -> jit
        print(f"  onnx backend unavailable ({exc}); using torch jit",
              flush=True)
        return load_silero_vad(), "jit"


def _speech_prob_curve(model: object, y: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Per-window P(speech) across the whole signal.

    Windows are fed consecutively (the model is a stateful LSTM), the
    same way a live stream would feed it. Returns (times_sec, probs)
    where times are window starts at ~32ms resolution. The trailing
    partial window is dropped.
    """
    import torch

    n_windows = len(y) // VAD_WINDOW_SAMPLES
    probs = np.zeros(n_windows, dtype=np.float64)
    model.reset_states()
    for i in range(n_windows):
        chunk = torch.from_numpy(
            y[i * VAD_WINDOW_SAMPLES:(i + 1) * VAD_WINDOW_SAMPLES]
            .astype(np.float32)
        )
        probs[i] = float(model(chunk, VAD_SAMPLE_RATE).item())
    times = np.arange(n_windows) * (VAD_WINDOW_SAMPLES / VAD_SAMPLE_RATE)
    return times, probs


def _smooth(probs: np.ndarray, dt: float, smoothing_sec: float) -> np.ndarray:
    # No edge trimming needed here: probs are non-negative, so the
    # zero-padding in the moving average can only pull edges DOWN -- it
    # can't fabricate onsets the way it faked loud blips on dB envelopes.
    window = max(1, int(round(smoothing_sec / dt)))
    return _moving_average(probs, window)


# ---------------------------------------------------------------------------
# Section stats + contrast
# ---------------------------------------------------------------------------

def _section_medians(sections: list[dict], times: np.ndarray,
                     probs: np.ndarray) -> list[dict]:
    rows = []
    for s in sections:
        start, end = float(s["start"]), float(s["end"])
        window = probs[(times >= start) & (times < end)]
        sid = str(s["section_id"])
        rows.append({
            "section_id": sid,
            "start_sec": start,
            "end_sec": end,
            "median_prob": float(np.median(window)) if len(window) else None,
            "is_vocal": sid.startswith("verse_"),
            "is_jam": sid.startswith("jam"),
        })
    return rows


def _contrast(rows: list[dict]) -> dict:
    """Headline metric: mean of per-verse median P(speech), minus the jam
    median. Positive contrast = Silero sees singing the jam lacks.
    """
    vocal = [r["median_prob"] for r in rows
             if r["is_vocal"] and r["median_prob"] is not None]
    jam = [r["median_prob"] for r in rows
           if r["is_jam"] and r["median_prob"] is not None]
    vocal_mean = float(np.mean(vocal)) if vocal else None
    jam_median = float(np.median(jam)) if jam else None
    contrast = (vocal_mean - jam_median
                if vocal_mean is not None and jam_median is not None else None)
    return {
        "vocal_mean_prob": vocal_mean,
        "jam_prob": jam_median,
        "contrast": contrast,
    }


# ---------------------------------------------------------------------------
# Onset detection (threshold sweep)
# ---------------------------------------------------------------------------

def _detect_onsets(times: np.ndarray, probs: np.ndarray, threshold: float,
                   song_offset: float, sustain_sec: float) -> list[float]:
    """Starts of above-threshold runs lasting >= sustain_sec, after
    song_offset. Pre-offset state counts as below, so audio already above
    threshold at song_offset yields an onset there.
    """
    mask = times >= song_offset
    t = times[mask]
    above = probs[mask] >= threshold
    n = len(t)
    dt = float(t[1] - t[0]) if n > 1 else VAD_WINDOW_SAMPLES / VAD_SAMPLE_RATE
    onsets: list[float] = []
    run_start: Optional[int] = None
    for i in range(n + 1):
        if i < n and above[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            duration = (i - run_start) * dt
            if duration >= sustain_sec:
                onsets.append(float(t[run_start]))
            run_start = None
    return onsets


def _nearest(onsets: list[float], gt_start: float,
             tolerance_sec: float) -> dict:
    if not onsets:
        return {"nearest_onset_sec": None, "error_sec": None,
                "detected": False}
    nearest = min(onsets, key=lambda o: abs(o - gt_start))
    error = nearest - gt_start
    return {
        "nearest_onset_sec": nearest,
        "error_sec": error,
        "detected": abs(error) <= tolerance_sec,
    }


def _sweep(times: np.ndarray, smoothed: np.ndarray, sections: list[dict],
           contrast: dict, song_offset: float,
           tolerance_sec: float) -> list[dict]:
    """One result row per threshold: the fixed sweep plus the adaptive
    threshold halfway between the jam median and the vocal mean (skipped
    when either is unavailable).
    """
    thresholds: list[tuple[str, float]] = [
        (f"{t:.2f}", t) for t in SWEEP_THRESHOLDS
    ]
    if contrast["jam_prob"] is not None and contrast["vocal_mean_prob"] is not None:
        adaptive = contrast["jam_prob"] + 0.5 * (
            contrast["vocal_mean_prob"] - contrast["jam_prob"])
        adaptive = float(np.clip(adaptive, 0.01, 0.99))
        thresholds.append((f"adaptive ({adaptive:.2f})", adaptive))

    gt_starts = {
        sid: next((float(s["start"]) for s in sections
                   if s["section_id"] == sid), None)
        for sid in ANCHOR_SECTIONS
    }
    results = []
    for label, value in thresholds:
        onsets = _detect_onsets(times, smoothed, value, song_offset,
                                SUSTAIN_SEC)
        row = {"label": label, "threshold": value,
               "onsets_sec": onsets, "n_onsets": len(onsets)}
        for sid in ANCHOR_SECTIONS:
            gt = gt_starts[sid]
            row[sid] = (
                _nearest(onsets, gt, tolerance_sec) if gt is not None
                else {"nearest_onset_sec": None, "error_sec": None,
                      "detected": False}
            )
        results.append(row)
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt_prob(value: Optional[float]) -> str:
    return "-" if value is None else f"{value:.2f}"


def _headline(contrast: dict) -> str:
    c = contrast["contrast"]
    sign = f"{c:+.2f}" if c is not None else "-"
    return (f"HEADLINE: mean vocal P(speech) = "
            f"{_fmt_prob(contrast['vocal_mean_prob'])} | "
            f"jam P(speech) = {_fmt_prob(contrast['jam_prob'])} | "
            f"CONTRAST = {sign}")


def _markdown(args: argparse.Namespace, duration: float, backend: str,
              section_rows: list[dict], contrast: dict,
              sweep: list[dict]) -> str:
    lines = [
        "# Silero VAD onset probe (no separation)",
        "",
        f"- Audio: `{args.audio}` ({duration:.1f}s, resampled to "
        f"{VAD_SAMPLE_RATE} Hz mono)",
        f"- Song offset: {args.song_offset:.1f}s | smoothing: "
        f"{args.smoothing_sec:.1f}s | backend: {backend}",
        f"- Ground truth: `{args.ground_truth}`",
        "",
        f"**{_headline(contrast)}**",
        "",
        "## Per-section median P(speech)",
        "",
        "| section | start | end | median P(speech) |",
        "|---|---|---|---|",
    ]
    for r in section_rows:
        sid = r["section_id"]
        if r["is_vocal"]:
            sid = f"**{sid}** (vocal)"
        elif r["is_jam"]:
            sid = f"**{sid}** (jam)"
        lines.append(f"| {sid} | {_fmt(r['start_sec'])} | {_fmt(r['end_sec'])} "
                     f"| {_fmt_prob(r['median_prob'])} |")
    lines += [
        "",
        f"## Onset detection (smoothed curve, sustained >= {SUSTAIN_SEC}s)",
        "",
        "| threshold | verse_1 onset | error | verse_5 onset (critical) "
        "| error | total onsets |",
        "|---|---|---|---|---|---|",
    ]
    for row in sweep:
        v1, v5 = row["verse_1"], row["verse_5"]
        lines.append(
            f"| {row['label']} "
            f"| {_fmt(v1['nearest_onset_sec'])} | {_fmt_signed(v1['error_sec'])} "
            f"| {_fmt(v5['nearest_onset_sec'])} | {_fmt_signed(v5['error_sec'])} "
            f"| {row['n_onsets']} |"
        )
    detected_at = [row["label"] for row in sweep if row["verse_5"]["detected"]]
    c = contrast["contrast"]
    lines += ["", "## VERDICT", ""]
    if c is None:
        lines.append("- Contrast not computable (missing vocal or jam "
                     "sections in ground truth).")
    else:
        lines.append(
            f"- Singing vs jam contrast: **{c:+.2f}** "
            f"(vocal {_fmt_prob(contrast['vocal_mean_prob'])} vs jam "
            f"{_fmt_prob(contrast['jam_prob'])})."
        )
    if detected_at:
        lines.append(
            f"- verse_5 (vocal re-entry after the jam, THE critical case) "
            f"detected within +/-{args.verdict_tolerance_sec:.0f}s at: "
            f"{', '.join(detected_at)}."
        )
    else:
        lines.append(
            f"- verse_5 (vocal re-entry after the jam, THE critical case) "
            f"NOT detected within +/-{args.verdict_tolerance_sec:.0f}s at "
            f"any threshold."
        )
    lines.append("")
    return "\n".join(lines)


def _curve_to_json(times: np.ndarray, probs: np.ndarray) -> list[dict]:
    return [{"time": round(float(t), 3), "prob": round(float(p), 4)}
            for t, p in zip(times, probs)]


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
    p.add_argument("--output-md", type=Path, default=Path("vad_probe.md"))
    p.add_argument("--output-json", type=Path, default=Path("vad_probe.json"))
    p.add_argument("--smoothing-sec", type=float, default=1.0,
                   help="Moving-average window for the smoothed probability "
                        "curve (default 1.0).")
    p.add_argument("--verdict-tolerance-sec", type=float, default=5.0,
                   help="Max |onset - gt_start| for verse_1 / verse_5 to "
                        "count as detected (default 5.0).")
    args = p.parse_args(argv)

    import librosa

    # ffmpeg pre-conversion (same trick as the sibling probes) to Silero's
    # required 16kHz mono, sidestepping extensionless-MP3 librosa quirks.
    wav_path = _ensure_wav(args.audio, target_sr=VAD_SAMPLE_RATE)
    try:
        print(f"Loading {args.audio} (raw, no separation) ...", flush=True)
        y, sr = librosa.load(str(wav_path), sr=VAD_SAMPLE_RATE, mono=True)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
    duration = len(y) / sr
    print(f"  {duration:.1f}s @ {sr} Hz", flush=True)

    sections = _load_sections(args.ground_truth)

    print("Loading Silero VAD (CPU) ...", flush=True)
    model, backend = _load_model()
    print(f"  backend: {backend}", flush=True)
    print(f"Running VAD over {duration:.1f}s "
          f"({len(y) // VAD_WINDOW_SAMPLES} windows of "
          f"{VAD_WINDOW_SAMPLES} samples) ...", flush=True)
    times, raw_probs = _speech_prob_curve(model, y)
    dt = VAD_WINDOW_SAMPLES / VAD_SAMPLE_RATE
    smoothed = _smooth(raw_probs, dt, args.smoothing_sec)

    section_rows = _section_medians(sections, times, raw_probs)
    contrast = _contrast(section_rows)
    print(_headline(contrast), flush=True)

    sweep = _sweep(times, smoothed, sections, contrast, args.song_offset,
                   args.verdict_tolerance_sec)
    for row in sweep:
        parts = []
        for sid in ANCHOR_SECTIONS:
            v = row[sid]
            if v["detected"]:
                parts.append(f"{sid} detected, error {v['error_sec']:+.2f}s")
            else:
                parts.append(f"{sid} NOT detected")
        print(f"  threshold {row['label']}: {' | '.join(parts)} "
              f"({row['n_onsets']} onsets)", flush=True)

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(
        _markdown(args, duration, backend, section_rows, contrast, sweep))
    print(f"Wrote markdown report to {args.output_md}", flush=True)

    report = {
        "audio": str(args.audio),
        "duration_sec": duration,
        "sample_rate": VAD_SAMPLE_RATE,
        "vad_window_samples": VAD_WINDOW_SAMPLES,
        "backend": backend,
        "song_offset": args.song_offset,
        "ground_truth": str(args.ground_truth),
        "smoothing_sec": args.smoothing_sec,
        "sustain_sec": SUSTAIN_SEC,
        "verdict_tolerance_sec": args.verdict_tolerance_sec,
        "vocal_mean_prob": contrast["vocal_mean_prob"],
        "jam_prob": contrast["jam_prob"],
        "contrast": contrast["contrast"],
        "section_medians": section_rows,
        "thresholds": sweep,
        "verse_5_detected_at": [row["label"] for row in sweep
                                if row["verse_5"]["detected"]],
        "raw_curve": _curve_to_json(times, raw_probs),
        "smoothed_curve": _curve_to_json(times, smoothed),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote JSON report to {args.output_json}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
