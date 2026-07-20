"""
xrec_probe.py
-------------
Probe: does the rung-3 mel vocal-presence head GENERALIZE across
recordings? Train on one labeled band recording, detect on a different
one -- different day, different energy, never seen by the model.

Rung 3b of the live vocal-onset ladder. Rung 3 (svd_probe.py) showed
that a logistic head on 64-band log-mel features separates singing from
the jam within one recording (timesplit AUC 0.96, verse_5 detected at
+0.5s). This probe is the cross-performance version: the head is
trained on ALL labeled windows of the train recording and applied
blind to the test recording. That mirrors the strongest deployment
story -- calibrate per song from a rehearsal take, ship the weights in
the song template, detect live on the night.

The test recording usually has no ground-truth file (labeling is
manual); pass the couple of timestamps you know instead:
--expected-verse-1-sec / --expected-verse-5-sec. The report gives
nearest-onset errors against those, plus the onset-free gap right
before the detected re-entry (a long gap = the jam produced no false
alarms at that threshold).

NO source separation anywhere. Only librosa + scikit-learn -- no torch.

CLI:

    python tooling/xrec_probe.py \\
        --train-audio feb26.m4a \\
        --train-ground-truth tooling/ground_truth/peggy_o_band.json \\
        --test-audio new_take.m4a \\
        --expected-verse-1-sec 56 --expected-verse-5-sec 303
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Sibling helpers (light module-level imports; heavy deps stay lazy).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav, _moving_average  # noqa: E402
from raw_onset_probe import _load_sections  # noqa: E402
from svd_probe import _mel_features, _new_clf, _window_labels  # noqa: E402
from vad_probe import SUSTAIN_SEC, _curve_to_json, _detect_onsets  # noqa: E402

SWEEP_THRESHOLDS = (0.3, 0.5, 0.7)
BLOCK_MIN_SEC = 2.0


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def _extract(audio: Path, window_sec: float
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(features, counts, times, duration) for one recording."""
    import librosa

    wav = _ensure_wav(audio, target_sr=24000)
    try:
        duration = float(librosa.get_duration(path=str(wav)))
        n_windows = int(duration // window_sec)
        feats, counts = _mel_features(wav, window_sec, n_windows)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    return feats, counts, np.arange(n_windows) * window_sec, duration


def _vocal_blocks(times: np.ndarray, smoothed: np.ndarray,
                  threshold: float) -> list[tuple[float, float]]:
    blocks: list[tuple[float, float]] = []
    start: Optional[float] = None
    for i, above in enumerate(smoothed >= threshold):
        if above and start is None:
            start = float(times[i])
        elif not above and start is not None:
            blocks.append((start, float(times[i])))
            start = None
    if start is not None and len(times):
        blocks.append((start, float(times[-1])))
    return [(a, b) for a, b in blocks if b - a >= BLOCK_MIN_SEC]


def _expected_row(onsets: list[float], expected: Optional[float],
                  tolerance_sec: float) -> dict:
    if expected is None:
        return {"expected_sec": None, "nearest_onset_sec": None,
                "error_sec": None, "detected": None, "gap_before_sec": None}
    if not onsets:
        return {"expected_sec": expected, "nearest_onset_sec": None,
                "error_sec": None, "detected": False, "gap_before_sec": None}
    nearest = min(onsets, key=lambda o: abs(o - expected))
    prior = [o for o in onsets if o < nearest]
    return {
        "expected_sec": expected,
        "nearest_onset_sec": nearest,
        "error_sec": nearest - expected,
        "detected": abs(nearest - expected) <= tolerance_sec,
        # Onset-free stretch ending at the matched onset: a long gap means
        # the jam (or whatever preceded the entry) stayed quiet.
        "gap_before_sec": nearest - (max(prior) if prior else 0.0),
    }


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--train-audio", type=Path, required=True,
                   action="append",
                   help="TRAIN recording. Repeat the flag (paired with "
                        "--train-ground-truth) to train on several takes.")
    p.add_argument("--train-ground-truth", type=Path, required=True,
                   action="append",
                   help="Labeled sections for the corresponding "
                        "--train-audio (same order).")
    p.add_argument("--test-audio", type=Path, required=True)
    p.add_argument("--expected-verse-1-sec", type=float, default=None,
                   help="Known first-verse vocal start in the TEST "
                        "recording, for error reporting.")
    p.add_argument("--expected-verse-5-sec", type=float, default=None,
                   help="Known post-jam re-entry in the TEST recording -- "
                        "the critical timestamp.")
    p.add_argument("--window-sec", type=float, default=0.5)
    p.add_argument("--smoothing-sec", type=float, default=2.0)
    p.add_argument("--verdict-tolerance-sec", type=float, default=5.0)
    p.add_argument("--output-md", type=Path, default=Path("xrec_probe.md"))
    p.add_argument("--output-json", type=Path, default=Path("xrec_probe.json"))
    args = p.parse_args(argv)

    if len(args.train_audio) != len(args.train_ground_truth):
        p.error("--train-audio and --train-ground-truth must be given the "
                "same number of times, in matching order")

    train_x_parts: list[np.ndarray] = []
    train_y_parts: list[np.ndarray] = []
    train_meta: list[dict] = []
    for audio, gt in zip(args.train_audio, args.train_ground_truth):
        print(f"Extracting mel features: train ({audio}) ...", flush=True)
        x_i, _c, t_i, d_i = _extract(audio, args.window_sec)
        sections = _load_sections(gt)
        _sids, labels = _window_labels(sections, t_i, args.window_sec)
        labeled = labels >= 0
        n_voc = int((labels == 1).sum())
        n_ins = int((labels == 0).sum())
        print(f"  {d_i:.1f}s, {n_voc} vocal / {n_ins} instrumental windows",
              flush=True)
        train_x_parts.append(x_i[labeled])
        train_y_parts.append(labels[labeled])
        train_meta.append({"audio": str(audio), "ground_truth": str(gt),
                           "duration_sec": d_i, "vocal_windows": n_voc,
                           "instrumental_windows": n_ins})
    x_train = np.concatenate(train_x_parts)
    y_train = np.concatenate(train_y_parts)
    d_train = sum(m["duration_sec"] for m in train_meta)

    print(f"Training mel head on {len(train_meta)} recording(s) "
          f"({len(y_train)} labeled windows) ...", flush=True)
    clf = _new_clf()
    clf.fit(x_train, y_train)

    print(f"Extracting mel features: test ({args.test_audio}) ...",
          flush=True)
    x_test, _c2, t_test, d_test = _extract(args.test_audio, args.window_sec)
    print(f"  {d_test:.1f}s", flush=True)
    probs = clf.predict_proba(x_test)[:, 1]
    smooth_w = max(1, int(round(args.smoothing_sec / args.window_sec)))
    smoothed = _moving_average(probs, smooth_w)

    sweep = []
    for thr in SWEEP_THRESHOLDS:
        onsets = _detect_onsets(t_test, smoothed, thr, 0.0, SUSTAIN_SEC)
        row = {
            "threshold": thr,
            "onsets_sec": onsets,
            "n_onsets": len(onsets),
            "vocal_blocks": _vocal_blocks(t_test, smoothed, thr),
            "verse_1": _expected_row(onsets, args.expected_verse_1_sec,
                                     args.verdict_tolerance_sec),
            "verse_5": _expected_row(onsets, args.expected_verse_5_sec,
                                     args.verdict_tolerance_sec),
        }
        sweep.append(row)
        parts = []
        for name in ("verse_1", "verse_5"):
            r = row[name]
            if r["detected"] is None:
                continue
            if r["detected"]:
                parts.append(f"{name} error {r['error_sec']:+.1f}s "
                             f"(gap before: {r['gap_before_sec']:.0f}s)")
            else:
                parts.append(f"{name} MISSED")
        print(f"  threshold {thr:.1f}: {len(onsets)} onsets"
              + (" | " + " | ".join(parts) if parts else ""), flush=True)

    # Markdown.
    train_desc = "; ".join(
        f"`{m['audio']}` with `{m['ground_truth']}`" for m in train_meta)
    md = [
        "# Cross-recording probe (rung 3b, no separation)",
        "",
        f"- Train ({len(train_meta)} recording(s), {d_train:.1f}s total): "
        f"{train_desc}",
        f"- Test: `{args.test_audio}` ({d_test:.1f}s) -- never seen in "
        "training",
        f"- Features: 64-band log-mel mean+std per {args.window_sec:.1f}s "
        f"window | smoothing {args.smoothing_sec:.1f}s | sustain >= "
        f"{SUSTAIN_SEC}s",
        "",
        "| threshold | verse_1 error | verse_5 error (critical) | gap "
        "before re-entry | total onsets |",
        "|---|---|---|---|---|",
    ]
    def err_cell(r: dict) -> str:
        if r["detected"] is None:
            return "n/a"
        if r["error_sec"] is None:
            return "MISSED (no onsets)"
        if r["detected"]:
            return f"{r['error_sec']:+.1f}s"
        return f"MISSED (nearest {r['error_sec']:+.1f}s)"

    for row in sweep:
        v5 = row["verse_5"]
        gap = (f"{v5['gap_before_sec']:.0f}s"
               if v5["detected"] and v5["gap_before_sec"] is not None
               else "-")
        md.append(f"| {row['threshold']:.1f} | {err_cell(row['verse_1'])} "
                  f"| {err_cell(v5)} | {gap} | {row['n_onsets']} |")
    md += ["", "## Detected structure per threshold", ""]
    for row in sweep:
        md.append(f"### threshold {row['threshold']:.1f}")
        md.append("")
        md.append("Onsets: " + (", ".join(_mmss(o) for o in row["onsets_sec"])
                                or "(none)"))
        md.append("")
        md.append("Vocal blocks (>= 2s): "
                  + (", ".join(f"{_mmss(a)}-{_mmss(b)}"
                               for a, b in row["vocal_blocks"]) or "(none)"))
        md.append("")
    md.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))
    print(f"Wrote markdown report to {args.output_md}", flush=True)

    report = {
        "train_recordings": train_meta,
        "train_duration_sec": d_train,
        "test_audio": str(args.test_audio),
        "test_duration_sec": d_test,
        "window_sec": args.window_sec,
        "smoothing_sec": args.smoothing_sec,
        "verdict_tolerance_sec": args.verdict_tolerance_sec,
        "expected_verse_1_sec": args.expected_verse_1_sec,
        "expected_verse_5_sec": args.expected_verse_5_sec,
        "thresholds": [
            {**row,
             "vocal_blocks": [[a, b] for a, b in row["vocal_blocks"]]}
            for row in sweep
        ],
        "curve": _curve_to_json(t_test, probs),
        "smoothed_curve": _curve_to_json(t_test, smoothed),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote JSON report to {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
