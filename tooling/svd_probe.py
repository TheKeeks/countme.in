"""
svd_probe.py
------------
Probe: is SUNG-vocal presence linearly decodable from raw phone-mic
audio -- no source separation -- using learned audio embeddings?

Rung 3 of the live vocal-onset ladder.
  Rung 1 (raw_onset_probe.py, energy/bandpass): FAILED -- vocal-minus-jam
    energy contrast ~0 dB; the band dominates the raw mix.
  Rung 2 (vad_probe.py, Silero VAD): FAILED -- speech-trained VAD scores
    P(speech) ~= 0.00 on singing over a band (it only fired on actual
    talking before/after the song).

This rung asks whether the *representation* was the problem: train a
tiny logistic head to separate singing windows from instrumental
windows, using the hand-labeled ground-truth sections as labels. NO
Demucs anywhere, not even for labeling. Two feature sets:

  mel   64-band log-mel mean+std per window. Trivially live-viable
        (a phone can compute mel frames in real time). If this works,
        the live path is nearly free.
  mert  m-a-p/MERT-v1-95M hidden states (the same music encoder
        build_song.py uses), one probe per layer. This is the signal
        CEILING: 95M params won't run in a browser tab as-is, but if
        vocal presence is decodable here, distillation to a small
        on-device model is a known path. If even MERT can't see it,
        the rung is dead.

Evaluation is leave-one-section-out: each labeled section's
predictions come from a classifier that never saw that section, so
verse_5 -- the vocal re-entry after the jam that live re-anchoring
depends on -- is scored by a model trained without it. The assembled
held-out P(vocal) curve then goes through the same threshold-sweep
onset detection as rung 2 for direct comparison.

CLI:

    python tooling/svd_probe.py \\
        --audio path/to/band_recording.m4a \\
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

# Sibling helpers (light module-level imports; heavy deps stay lazy).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav, _moving_average  # noqa: E402
from raw_onset_probe import _fmt, _load_sections  # noqa: E402
from vad_probe import (  # noqa: E402
    ANCHOR_SECTIONS, SUSTAIN_SEC, _contrast, _curve_to_json, _fmt_prob,
    _section_medians, _sweep,
)

MERT_MODEL = "m-a-p/MERT-v1-95M"  # same encoder as build_song.py
MERT_SAMPLE_RATE = 24000
MEL_SAMPLE_RATE = 22050
MEL_BANDS = 64
MEL_HOP = 512


# ---------------------------------------------------------------------------
# Windowing + labels
# ---------------------------------------------------------------------------

def _pool_windows(frames: np.ndarray, frame_times: np.ndarray,
                  window_sec: float, n_windows: int, with_std: bool = False
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Pool per-frame features (T, D) into per-window mean (and optional
    std) features on the absolute-time grid. Returns (features, counts).
    """
    idx = (frame_times // window_sec).astype(int)
    valid = (idx >= 0) & (idx < n_windows)
    idx, frames = idx[valid], frames[valid]
    d = frames.shape[1]
    sums = np.zeros((n_windows, d), dtype=np.float64)
    sumsq = np.zeros((n_windows, d), dtype=np.float64)
    counts = np.zeros(n_windows, dtype=np.float64)
    np.add.at(sums, idx, frames)
    np.add.at(sumsq, idx, frames.astype(np.float64) ** 2)
    np.add.at(counts, idx, 1.0)
    safe = np.maximum(counts, 1.0)[:, None]
    mean = sums / safe
    if not with_std:
        return mean.astype(np.float32), counts
    var = np.maximum(sumsq / safe - mean ** 2, 0.0)
    feats = np.concatenate([mean, np.sqrt(var)], axis=1)
    return feats.astype(np.float32), counts


def _window_labels(sections: list[dict], times: np.ndarray,
                   window_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-window (section_id or '', label) where label is 1 for vocal
    (verse_*), 0 for instrumental (any other labeled section), -1 for
    unlabeled (window not fully inside a labeled section).
    """
    section_ids = np.full(len(times), "", dtype=object)
    labels = np.full(len(times), -1, dtype=np.int64)
    for s in sections:
        sid = str(s["section_id"])
        inside = (times >= float(s["start"])) & \
                 (times + window_sec <= float(s["end"]))
        section_ids[inside] = sid
        labels[inside] = 1 if sid.startswith("verse_") else 0
    return section_ids, labels


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _mel_features(wav_path: Path, window_sec: float, n_windows: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    import librosa

    y, sr = librosa.load(str(wav_path), sr=MEL_SAMPLE_RATE, mono=True)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, hop_length=MEL_HOP, n_mels=MEL_BANDS)
    log_s = librosa.power_to_db(S, ref=np.max)  # (n_mels, T)
    frame_times = librosa.frames_to_time(
        np.arange(log_s.shape[1]), sr=sr, hop_length=MEL_HOP)
    return _pool_windows(log_s.T, frame_times, window_sec, n_windows,
                         with_std=True)


def _mert_features(wav_path: Path, window_sec: float, n_windows: int,
                   chunk_sec: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-window mean hidden state for EVERY MERT layer.

    Returns (features shape (n_layers, n_windows, dim), counts). Audio is
    processed in chunk_sec chunks (memory), frames assigned to absolute
    0.5s windows; boundary windows lose a little context, which is fine
    for a probe (and mirrors what a streaming deployment would see).
    """
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    y, sr = librosa.load(str(wav_path), sr=MERT_SAMPLE_RATE, mono=True)
    print(f"  loading {MERT_MODEL} ...", flush=True)
    processor = AutoFeatureExtractor.from_pretrained(
        MERT_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MERT_MODEL, trust_remote_code=True)
    model.eval()

    sums: Optional[np.ndarray] = None
    counts = np.zeros(n_windows, dtype=np.float64)
    chunk_samples = int(chunk_sec * sr)
    n_chunks = int(np.ceil(len(y) / chunk_samples))
    for k in range(n_chunks):
        seg = y[k * chunk_samples:(k + 1) * chunk_samples]
        if len(seg) < int(0.5 * sr):
            continue
        inputs = processor(seg, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states  # tuple of (1, T, D) per layer
        n_layers, dim = len(hidden), hidden[0].shape[-1]
        if sums is None:
            sums = np.zeros((n_layers, n_windows, dim), dtype=np.float64)
        t_frames = hidden[0].shape[1]
        fps = t_frames / (len(seg) / sr)
        frame_times = k * chunk_sec + (np.arange(t_frames) + 0.5) / fps
        idx = (frame_times // window_sec).astype(int)
        valid = (idx >= 0) & (idx < n_windows)
        np.add.at(counts, idx[valid], 1.0)
        for layer, h in enumerate(hidden):
            arr = h[0].numpy().astype(np.float64)
            np.add.at(sums[layer], idx[valid], arr[valid])
        print(f"  chunk {k + 1}/{n_chunks} done", flush=True)
    if sums is None:
        raise RuntimeError("audio too short for MERT feature extraction")
    safe = np.maximum(counts, 1.0)[None, :, None]
    return (sums / safe).astype(np.float32), counts


# ---------------------------------------------------------------------------
# Held-out probability curve (leave-one-section-out)
# ---------------------------------------------------------------------------

def _heldout_curve(features: np.ndarray, section_ids: np.ndarray,
                   labels: np.ndarray, valid: np.ndarray
                   ) -> tuple[np.ndarray, Optional[float]]:
    """Assemble a P(vocal) curve where every labeled window is predicted
    by a classifier that never saw its section, and unlabeled windows by
    a classifier trained on all labeled sections. Returns (probs, AUC on
    the pooled held-out labeled windows).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def new_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )

    labeled = (labels >= 0) & valid
    probs = np.zeros(len(labels), dtype=np.float64)
    for sec in sorted({s for s in section_ids[labeled]}):
        test = (section_ids == sec) & valid
        train = labeled & ~test
        if len(np.unique(labels[train])) < 2:
            continue
        clf = new_clf()
        clf.fit(features[train], labels[train])
        probs[test] = clf.predict_proba(features[test])[:, 1]
    unlabeled = ~labeled & valid
    if unlabeled.any() and len(np.unique(labels[labeled])) == 2:
        clf = new_clf()
        clf.fit(features[labeled], labels[labeled])
        probs[unlabeled] = clf.predict_proba(features[unlabeled])[:, 1]
    auc = (float(roc_auc_score(labels[labeled], probs[labeled]))
           if len(np.unique(labels[labeled])) == 2 else None)
    return probs, auc


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _headline(name: str, contrast: dict, auc: Optional[float]) -> str:
    c = contrast["contrast"]
    sign = f"{c:+.2f}" if c is not None else "-"
    auc_s = f"{auc:.3f}" if auc is not None else "-"
    return (f"{name}: held-out vocal P(vocal) = "
            f"{_fmt_prob(contrast['vocal_mean_prob'])} | jam P(vocal) = "
            f"{_fmt_prob(contrast['jam_prob'])} | CONTRAST = {sign} | "
            f"AUC = {auc_s}")


def _feature_markdown(name: str, note: str, result: dict,
                      tolerance_sec: float) -> list[str]:
    lines = [f"## Feature set: {name}", "", note, "",
             f"**{_headline(name, result['contrast'], result['auc'])}**", "",
             "| section | start | held-out median P(vocal) |",
             "|---|---|---|"]
    for r in result["section_medians"]:
        sid = r["section_id"]
        if r["is_vocal"]:
            sid = f"**{sid}** (vocal)"
        elif r["is_jam"]:
            sid = f"**{sid}** (jam)"
        lines.append(f"| {sid} | {_fmt(r['start_sec'])} "
                     f"| {_fmt_prob(r['median_prob'])} |")
    lines += ["",
              f"Onset detection (smoothed curve, sustained >= {SUSTAIN_SEC}s):",
              "",
              "| threshold | verse_1 onset | error | verse_5 onset (critical) "
              "| error | total onsets |",
              "|---|---|---|---|---|---|"]
    for row in result["thresholds"]:
        v1, v5 = row["verse_1"], row["verse_5"]
        e1 = "-" if v1["error_sec"] is None else f"{v1['error_sec']:+.2f}s"
        e5 = "-" if v5["error_sec"] is None else f"{v5['error_sec']:+.2f}s"
        lines.append(
            f"| {row['label']} | {_fmt(v1['nearest_onset_sec'])} | {e1} "
            f"| {_fmt(v5['nearest_onset_sec'])} | {e5} | {row['n_onsets']} |")
    detected_at = [row["label"] for row in result["thresholds"]
                   if row["verse_5"]["detected"]]
    if detected_at:
        lines.append(f"\n- verse_5 detected within +/-{tolerance_sec:.0f}s "
                     f"at: {', '.join(detected_at)}")
    else:
        lines.append(f"\n- verse_5 NOT detected within +/-{tolerance_sec:.0f}s "
                     f"at any threshold")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True,
                   help="Ground-truth JSON with section starts; the "
                        "hand-labeled sections are also the training labels.")
    p.add_argument("--song-offset", type=float, default=20.0)
    p.add_argument("--output-md", type=Path, default=Path("svd_probe.md"))
    p.add_argument("--output-json", type=Path, default=Path("svd_probe.json"))
    p.add_argument("--window-sec", type=float, default=0.5,
                   help="Feature window / label resolution (default 0.5).")
    p.add_argument("--smoothing-sec", type=float, default=2.0,
                   help="Moving average over the P(vocal) curve before onset "
                        "detection (default 2.0).")
    p.add_argument("--verdict-tolerance-sec", type=float, default=5.0)
    p.add_argument("--chunk-sec", type=float, default=30.0,
                   help="MERT processing chunk length (default 30.0).")
    p.add_argument("--features", choices=["both", "mel", "mert"],
                   default="both")
    args = p.parse_args(argv)

    import librosa

    wav_path = _ensure_wav(args.audio, target_sr=MERT_SAMPLE_RATE)
    try:
        duration = float(librosa.get_duration(path=str(wav_path)))
        n_windows = int(duration // args.window_sec)
        times = np.arange(n_windows) * args.window_sec
        print(f"Audio: {duration:.1f}s -> {n_windows} windows of "
              f"{args.window_sec:.1f}s", flush=True)

        sections = _load_sections(args.ground_truth)
        section_ids, labels = _window_labels(sections, times, args.window_sec)
        n_voc = int((labels == 1).sum())
        n_ins = int((labels == 0).sum())
        print(f"Labels: {n_voc} vocal windows, {n_ins} instrumental, "
              f"{int((labels == -1).sum())} unlabeled", flush=True)

        results: dict[str, dict] = {}
        smooth_w = max(1, int(round(args.smoothing_sec / args.window_sec)))

        def analyze(name: str, probs: np.ndarray, auc: Optional[float],
                    extra: dict) -> None:
            smoothed = _moving_average(probs, smooth_w)
            rows = _section_medians(sections, times, probs)
            contrast = _contrast(rows)
            sweep = _sweep(times, smoothed, sections, contrast,
                           args.song_offset, args.verdict_tolerance_sec)
            results[name] = {
                "auc": auc, "section_medians": rows, "contrast": contrast,
                "thresholds": sweep,
                "verse_5_detected_at": [r["label"] for r in sweep
                                        if r["verse_5"]["detected"]],
                "curve": _curve_to_json(times, probs),
                "smoothed_curve": _curve_to_json(times, smoothed),
                **extra,
            }
            print(_headline(name, contrast, auc), flush=True)
            for row in sweep:
                v5 = row["verse_5"]
                status = (f"detected, error {v5['error_sec']:+.2f}s"
                          if v5["detected"] else "NOT detected")
                print(f"  threshold {row['label']}: verse_5 {status} "
                      f"({row['n_onsets']} onsets)", flush=True)

        if args.features in ("both", "mel"):
            print("Extracting log-mel features (live-viable baseline) ...",
                  flush=True)
            mel_feats, mel_counts = _mel_features(
                wav_path, args.window_sec, n_windows)
            probs, auc = _heldout_curve(
                mel_feats, section_ids, labels, mel_counts > 0)
            analyze("mel", probs, auc, {})

        if args.features in ("both", "mert"):
            print("Extracting MERT hidden states (signal ceiling) ...",
                  flush=True)
            mert_feats, mert_counts = _mert_features(
                wav_path, args.window_sec, n_windows, args.chunk_sec)
            valid = mert_counts > 0
            print(f"Probing {mert_feats.shape[0]} MERT layers "
                  f"(leave-one-section-out logistic each) ...", flush=True)
            layer_results = []
            for layer in range(mert_feats.shape[0]):
                probs, auc = _heldout_curve(
                    mert_feats[layer], section_ids, labels, valid)
                layer_results.append((layer, probs, auc))
                print(f"  layer {layer:2d}: AUC = "
                      f"{auc if auc is None else round(auc, 3)}", flush=True)
            scored = [(l, p, a) for l, p, a in layer_results if a is not None]
            best_layer, best_probs, best_auc = max(scored, key=lambda r: r[2])
            print(f"Best MERT layer: {best_layer} (AUC {best_auc:.3f})",
                  flush=True)
            analyze(f"mert_layer_{best_layer}", best_probs, best_auc, {
                "layer_aucs": [{"layer": l, "auc": a}
                               for l, _p, a in layer_results],
                "best_layer": best_layer,
            })
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    # Markdown.
    md = [
        "# Singing-voice detection probe (rung 3, no separation)",
        "",
        f"- Audio: `{args.audio}` ({duration:.1f}s)",
        f"- Window: {args.window_sec:.1f}s | smoothing: "
        f"{args.smoothing_sec:.1f}s | song offset: {args.song_offset:.1f}s",
        f"- Ground truth (labels + eval): `{args.ground_truth}` "
        f"({n_voc} vocal / {n_ins} instrumental windows)",
        "- Protocol: leave-one-section-out -- every labeled section is "
        "scored by a classifier that never saw it.",
        "",
    ]
    for line in [
        _headline(name, r["contrast"], r["auc"]) for name, r in results.items()
    ]:
        md.append(f"**{line}**")
        md.append("")
    notes = {
        "mel": "64-band log-mel mean+std per window -- trivially live-viable "
               "on a phone.",
    }
    for name, r in results.items():
        note = notes.get(
            name,
            f"MERT hidden states, best layer by held-out AUC "
            f"(layer sweep in JSON) -- the signal ceiling; needs "
            f"distillation for on-device use.")
        md += _feature_markdown(name, note, r, args.verdict_tolerance_sec)

    md += ["## VERDICT", ""]
    for name, r in results.items():
        c = r["contrast"]["contrast"]
        det = r["verse_5_detected_at"]
        md.append(
            f"- {name}: contrast "
            f"{'-' if c is None else f'{c:+.2f}'}, verse_5 "
            + (f"detected at {', '.join(det)}." if det
               else "NOT detected at any threshold."))
    md.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))
    print(f"Wrote markdown report to {args.output_md}", flush=True)

    report = {
        "audio": str(args.audio),
        "duration_sec": duration,
        "window_sec": args.window_sec,
        "smoothing_sec": args.smoothing_sec,
        "song_offset": args.song_offset,
        "ground_truth": str(args.ground_truth),
        "verdict_tolerance_sec": args.verdict_tolerance_sec,
        "mert_model": MERT_MODEL,
        "n_vocal_windows": n_voc,
        "n_instrumental_windows": n_ins,
        "features": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote JSON report to {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
