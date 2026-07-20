"""
train_vocal_head.py
-------------------
Train the per-song vocal-presence head that ships to the browser, and
export it as a weights JSON that web/js/vocal-onset.js can execute.

This is the productization of rung 3 (svd_probe.py) / rung 3b
(xrec_probe.py): a logistic head on pooled log-mel windows, trained on
labeled band recordings, detecting sung-vocal onsets on raw phone-mic
audio with no source separation.

CRITICAL: features here are BROWSER-MATCHED, not the probe config. The
live AudioEngine captures at 16 kHz, streams (no STFT centering), and
can only normalize causally, so this trainer computes:

    sr=16000, n_fft=1024 (Hann, center=False), hop=512, n_mels=64
    (Slaney, fmax=8000), power spectrum -> dB with a RUNNING-max
    reference (causal stand-in for librosa's ref=np.max), floored at
    ref-80 dB, pooled per 0.5s window as mean+std (128 dims),
    z-scored with train-set stats, logistic -> P(vocal),
    TRAILING 2s moving average (causal, unlike the probes' centered
    smoothing), sustained-crossing onsets (threshold 0.5, >=1.5s).

Everything the browser needs -- filterbank config, scaler, weights,
detection constants, and a synthetic parity vector for a load-time
self-check -- goes into the output JSON. web/js/vocal-onset.js must
stay formula-identical to _stream_log_mel / _pool below.

CLI:

    python tooling/train_vocal_head.py \\
        --train-audio feb26.m4a --train-ground-truth .../peggy_o_band.json \\
        --train-audio take2.m4a --train-ground-truth .../peggy_o_band2.json \\
        --test-audio take3.m4a --test-ground-truth .../peggy_o_band3.json \\
        --output web/templates/peggy_o_vocal_head.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav  # noqa: E402
from raw_onset_probe import _load_sections  # noqa: E402
from svd_probe import _pool_windows, _window_labels  # noqa: E402
from vad_probe import _detect_onsets  # noqa: E402

MODEL_NAME = "vocal_onset_head_v1"

FEATURE_CONFIG = {
    "sample_rate": 16000,
    "n_fft": 1024,
    "hop": 512,
    "n_mels": 64,
    "fmin": 0.0,
    "fmax": 8000.0,
    "window_sec": 0.5,
    "pooling": "mean+std",
    "log_ref": "running_max",
    "log_floor_db": -80.0,
    "smoothing_sec": 2.0,
    "threshold": 0.5,
    "sustain_sec": 1.5,
}

# Gate mapping: spread = q90 - q10 of the smoothed P(vocal) curve over an
# EXPANDING window from song start (a trailing window would fill with lows
# during a long jam and zero the gate exactly when the re-entry matters).
# Below GATE_LOW the onset term gets zero weight; above GATE_HIGH, full
# weight; linear in between. Calibrated from the three real takes: crisp
# mixes (takes 1-2) measure ~0.87, the hot unreliable mix (take 3) ~0.67.
# Provisional with n=3 -- deliberately conservative: when in doubt, the
# tracker ignores the vocal term and falls back to chroma-DTW.
GATE_LOW = 0.70
GATE_HIGH = 0.85
GATE_MIN_SEC = 60.0


# ---------------------------------------------------------------------------
# Browser-matched feature pipeline
# ---------------------------------------------------------------------------

def _mel_power(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mel power spectrogram exactly as the browser computes it:
    center=False (streaming frames, no reflection padding). Returns
    (S shape (n_mels, T), frame_start_times).
    """
    import librosa

    cfg = FEATURE_CONFIG
    S = librosa.feature.melspectrogram(
        y=y, sr=cfg["sample_rate"], n_fft=cfg["n_fft"],
        hop_length=cfg["hop"], n_mels=cfg["n_mels"],
        fmin=cfg["fmin"], fmax=cfg["fmax"], power=2.0, center=False,
    )
    times = np.arange(S.shape[1]) * (cfg["hop"] / cfg["sample_rate"])
    return S, times


def _stream_log_mel(S: np.ndarray) -> np.ndarray:
    """Causal dB conversion: per-frame running max as the reference
    (librosa's ref=np.max needs the whole file; a live stream doesn't
    have it). Floored at ref + log_floor_db, matching power_to_db's
    top_db=80 behavior once the running max converges.
    """
    db = 10.0 * np.log10(np.maximum(S, 1e-10))
    frame_max = db.max(axis=0)
    running_ref = np.maximum.accumulate(frame_max)
    out = db - running_ref[None, :]
    return np.maximum(out, FEATURE_CONFIG["log_floor_db"])


def _features(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(pooled features (n_windows, 128), counts, window_start_times)."""
    S, frame_times = _mel_power(y)
    log_s = _stream_log_mel(S)
    w = FEATURE_CONFIG["window_sec"]
    n_windows = int((frame_times[-1] + FEATURE_CONFIG["hop"]
                     / FEATURE_CONFIG["sample_rate"]) // w) if len(frame_times) else 0
    feats, counts = _pool_windows(log_s.T, frame_times, w, n_windows,
                                  with_std=True)
    return feats, counts, np.arange(n_windows) * w


def _trailing_mean(p: np.ndarray, k: int) -> np.ndarray:
    """Causal smoothing: mean of the last k values (including current).
    The probes used centered smoothing; the browser can't see the
    future, so eval here must match the browser.
    """
    out = np.empty_like(p)
    for i in range(len(p)):
        out[i] = p[max(0, i - k + 1): i + 1].mean()
    return out


def _load_16k(audio: Path) -> np.ndarray:
    import librosa

    wav = _ensure_wav(audio, target_sr=FEATURE_CONFIG["sample_rate"])
    try:
        y, _sr = librosa.load(str(wav), sr=FEATURE_CONFIG["sample_rate"],
                              mono=True)
    finally:
        try:
            os.unlink(wav)
        except OSError:
            pass
    return y


# ---------------------------------------------------------------------------
# Parity vector (browser self-check)
# ---------------------------------------------------------------------------

def _parity_signal() -> np.ndarray:
    """Deterministic 2s synthetic both sides can generate exactly."""
    sr = FEATURE_CONFIG["sample_rate"]
    t = np.arange(2 * sr, dtype=np.float64) / sr
    y = (0.5 * np.sin(2 * np.pi * 440.0 * t)
         + 0.3 * np.sin(2 * np.pi * 1000.0 * t)
         + 0.2 * np.sin(2 * np.pi * 3000.0 * t))
    return y.astype(np.float32)


def _parity_vector() -> list[list[float]]:
    """Pooled log-mel (ABSOLUTE dB, ref=1 -- no running max, so the check
    isolates FFT + filterbank parity) for the synthetic signal's first
    3 windows.
    """
    S, frame_times = _mel_power(_parity_signal().astype(np.float64))
    log_s = 10.0 * np.log10(np.maximum(S, 1e-10))
    w = FEATURE_CONFIG["window_sec"]
    feats, _c = _pool_windows(log_s.T, frame_times, w, 3, with_std=True)
    return [[round(float(v), 3) for v in row] for row in feats]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def _anchor_start(sections: list[dict], sid: str) -> Optional[float]:
    return next((float(s["start"]) for s in sections
                 if s["section_id"] == sid), None)


def _evaluate(name: str, probs: np.ndarray, times: np.ndarray,
              sections: list[dict], tolerance_sec: float) -> dict:
    from sklearn.metrics import roc_auc_score

    cfg = FEATURE_CONFIG
    k = max(1, int(round(cfg["smoothing_sec"] / cfg["window_sec"])))
    sm = _trailing_mean(probs, k)

    _sids, labels = _window_labels(sections, times, cfg["window_sec"])
    labeled = labels >= 0
    auc = (float(roc_auc_score(labels[labeled], sm[labeled]))
           if len(np.unique(labels[labeled])) == 2 else None)
    vocal_med = float(np.median(sm[labeled & (labels == 1)])) \
        if (labels == 1).any() else None
    jam_windows = sm[labeled & (labels == 0)]
    instr_med = float(np.median(jam_windows)) if len(jam_windows) else None

    song_start = min(float(s["start"]) for s in sections)
    song_end = max(float(s["end"]) for s in sections if s["end"] < 9000)
    active = (times >= song_start) & (times < song_end)
    spread = (float(np.quantile(sm[active], 0.9)
                    - np.quantile(sm[active], 0.1)) if active.any() else None)

    onsets = _detect_onsets(times, sm, cfg["threshold"], song_start,
                            cfg["sustain_sec"])
    result = {"take": name, "auc": auc, "vocal_median": vocal_med,
              "instrumental_median": instr_med, "gate_spread": spread,
              "onsets_sec": [round(o, 1) for o in onsets]}
    for sid in ("verse_1", "verse_5"):
        gt = _anchor_start(sections, sid)
        if gt is None or not onsets:
            result[sid] = None
            continue
        nearest = min(onsets, key=lambda o: abs(o - gt))
        result[sid] = {"gt_sec": gt, "nearest_onset_sec": nearest,
                       "error_sec": round(nearest - gt, 2),
                       "detected": abs(nearest - gt) <= tolerance_sec}
    jam_ivs = [(float(s["start"]), float(s["end"])) for s in sections
               if str(s["section_id"]).startswith("jam")]
    result["jam_false_alarms"] = [
        round(o, 1) for o in onsets
        if any(a <= o < b - tolerance_sec for a, b in jam_ivs)
    ]
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--train-audio", type=Path, required=True, action="append")
    p.add_argument("--train-ground-truth", type=Path, required=True,
                   action="append")
    p.add_argument("--test-audio", type=Path, action="append", default=[])
    p.add_argument("--test-ground-truth", type=Path, action="append",
                   default=[])
    p.add_argument("--verdict-tolerance-sec", type=float, default=5.0)
    p.add_argument("--output", type=Path, required=True,
                   help="Weights JSON consumed by web/js/vocal-onset.js.")
    args = p.parse_args(argv)
    if len(args.train_audio) != len(args.train_ground_truth):
        p.error("--train-audio / --train-ground-truth count mismatch")
    if len(args.test_audio) != len(args.test_ground_truth):
        p.error("--test-audio / --test-ground-truth count mismatch")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cfg = FEATURE_CONFIG
    x_parts, y_parts, trained_on = [], [], []
    for audio, gt in zip(args.train_audio, args.train_ground_truth):
        print(f"Train take: {audio}", flush=True)
        y16 = _load_16k(audio)
        feats, _c, times = _features(y16)
        sections = _load_sections(gt)
        _sids, labels = _window_labels(sections, times, cfg["window_sec"])
        labeled = labels >= 0
        print(f"  {len(y16) / cfg['sample_rate']:.1f}s, "
              f"{int((labels == 1).sum())} vocal / "
              f"{int((labels == 0).sum())} instrumental windows", flush=True)
        x_parts.append(feats[labeled])
        y_parts.append(labels[labeled])
        trained_on.append({"audio": str(audio), "ground_truth": str(gt)})
    x_train = np.concatenate(x_parts)
    y_train = np.concatenate(y_parts)

    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(x_train), y_train)
    print(f"Trained on {len(y_train)} windows from {len(trained_on)} take(s)",
          flush=True)

    evals = []
    for audio, gt in zip(args.test_audio, args.test_ground_truth):
        print(f"Eval take: {audio}", flush=True)
        y16 = _load_16k(audio)
        feats, _c, times = _features(y16)
        probs = clf.predict_proba(scaler.transform(feats))[:, 1]
        sections = _load_sections(gt)
        e = _evaluate(str(audio), probs, times, sections,
                      args.verdict_tolerance_sec)
        evals.append(e)
        auc_s = "-" if e["auc"] is None else f"{e['auc']:.3f}"
        spread_s = ("-" if e["gate_spread"] is None
                    else f"{e['gate_spread']:.2f}")
        print(f"  AUC {auc_s} | gate spread {spread_s} | "
              f"jam FAs {len(e['jam_false_alarms'])}", flush=True)
        for sid in ("verse_1", "verse_5"):
            a = e[sid]
            if a is None:
                print(f"  {sid}: n/a", flush=True)
            else:
                print(f"  {sid}: {'ok' if a['detected'] else 'MISS'} "
                      f"error {a['error_sec']:+.1f}s", flush=True)

    head = {
        "model": MODEL_NAME,
        "song_id": json.loads(
            args.train_ground_truth[0].read_text()).get("song_id"),
        "feature_config": cfg,
        "gate": {"mode": "expanding", "low": GATE_LOW, "high": GATE_HIGH,
                 "min_sec": GATE_MIN_SEC},
        "scaler_mean": [round(float(v), 6) for v in scaler.mean_],
        "scaler_scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
        "trained_on": trained_on,
        "eval": evals,
        "parity": {
            "signal": "0.5*sin(2pi*440t) + 0.3*sin(2pi*1000t) + "
                      "0.2*sin(2pi*3000t), 2.0s at 16kHz, float32",
            "tolerance_db": 0.1,
            "expected_windows": _parity_vector(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(head, indent=1))
    print(f"Wrote head to {args.output} "
          f"({args.output.stat().st_size / 1024:.0f} KB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
