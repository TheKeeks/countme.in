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

Two evaluation protocols:

  timesplit (PRIMARY, deployment-shaped) -- train on everything before
    a cut point (default: the middle of the jam), test on everything
    after. This mirrors live use: by mid-jam the app has heard the
    intro, four verses, and jam material (whose rough section identity
    the template + tap already give it), and the question is whether a
    head calibrated on that past detects the verse_5 vocal re-entry.
  loso (secondary, strict) -- leave-one-section-out. Honest about
    cross-section generalization but too harsh for the negative class
    here: with only two instrumental sections, holding out the jam
    leaves a single 13s intro as the only instrumental training
    example, so "jam looks vocal" is expected, not a finding.

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


def _clip_sections(sections: list[dict], from_sec: float) -> list[dict]:
    """Sections restricted to [from_sec, inf): drop those that end before
    it, clip the start of the one straddling it.
    """
    clipped = []
    for s in sections:
        if float(s["end"]) <= from_sec:
            continue
        c = dict(s)
        c["start"] = max(float(s["start"]), from_sec)
        clipped.append(c)
    return clipped


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
    windows; boundary windows lose a little context, which is fine for a
    probe (and mirrors what a streaming deployment would see).
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
# Classifiers / evaluation protocols
# ---------------------------------------------------------------------------

def _new_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def _auc(labels: np.ndarray, probs: np.ndarray,
         mask: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels[mask])) < 2:
        return None
    return float(roc_auc_score(labels[mask], probs[mask]))


def _timesplit_curve(features: np.ndarray, labels: np.ndarray,
                     valid: np.ndarray, times: np.ndarray,
                     train_until: float
                     ) -> tuple[np.ndarray, Optional[float]]:
    """PRIMARY protocol. Train on labeled windows before `train_until`,
    predict everything after it. Returns (probs -- zero before the cut,
    AUC on the labeled test windows).
    """
    train = (labels >= 0) & valid & (times < train_until)
    test = valid & (times >= train_until)
    probs = np.zeros(len(labels), dtype=np.float64)
    if len(np.unique(labels[train])) < 2 or not test.any():
        return probs, None
    clf = _new_clf()
    clf.fit(features[train], labels[train])
    probs[test] = clf.predict_proba(features[test])[:, 1]
    return probs, _auc(labels, probs, (labels >= 0) & test)


def _loso_curve(features: np.ndarray, section_ids: np.ndarray,
                labels: np.ndarray, valid: np.ndarray
                ) -> tuple[np.ndarray, Optional[float]]:
    """Secondary protocol: every labeled section predicted by a
    classifier that never saw it; unlabeled windows by one trained on
    all labeled sections.
    """
    labeled = (labels >= 0) & valid
    probs = np.zeros(len(labels), dtype=np.float64)
    for sec in sorted({s for s in section_ids[labeled]}):
        test = (section_ids == sec) & valid
        train = labeled & ~test
        if len(np.unique(labels[train])) < 2:
            continue
        clf = _new_clf()
        clf.fit(features[train], labels[train])
        probs[test] = clf.predict_proba(features[test])[:, 1]
    unlabeled = ~labeled & valid
    if unlabeled.any() and len(np.unique(labels[labeled])) == 2:
        clf = _new_clf()
        clf.fit(features[labeled], labels[labeled])
        probs[unlabeled] = clf.predict_proba(features[unlabeled])[:, 1]
    return probs, _auc(labels, probs, labeled)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _headline(name: str, contrast: dict, auc: Optional[float]) -> str:
    c = contrast["contrast"]
    sign = f"{c:+.2f}" if c is not None else "-"
    auc_s = f"{auc:.3f}" if auc is not None else "-"
    return (f"{name}: vocal P(vocal) = "
            f"{_fmt_prob(contrast['vocal_mean_prob'])} | jam P(vocal) = "
            f"{_fmt_prob(contrast['jam_prob'])} | CONTRAST = {sign} | "
            f"AUC = {auc_s}")


def _sweep_markdown(sweep: list[dict], tolerance_sec: float,
                    verse_1_in_test: bool) -> list[str]:
    lines = ["| threshold | verse_1 onset | error | verse_5 onset (critical) "
             "| error | total onsets |",
             "|---|---|---|---|---|---|"]
    for row in sweep:
        v1, v5 = row["verse_1"], row["verse_5"]
        if verse_1_in_test:
            c1 = (f"{_fmt(v1['nearest_onset_sec'])} | "
                  + ("-" if v1["error_sec"] is None
                     else f"{v1['error_sec']:+.2f}s"))
        else:
            c1 = "n/a (in train region) | n/a"
        e5 = "-" if v5["error_sec"] is None else f"{v5['error_sec']:+.2f}s"
        lines.append(f"| {row['label']} | {c1} "
                     f"| {_fmt(v5['nearest_onset_sec'])} | {e5} "
                     f"| {row['n_onsets']} |")
    detected_at = [row["label"] for row in sweep
                   if row["verse_5"]["detected"]]
    if detected_at:
        lines.append(f"\n- verse_5 detected within +/-{tolerance_sec:.0f}s "
                     f"at: {', '.join(detected_at)}")
    else:
        lines.append(f"\n- verse_5 NOT detected within "
                     f"+/-{tolerance_sec:.0f}s at any threshold")
    return lines


def _medians_markdown(rows: list[dict]) -> list[str]:
    lines = ["| section | start | median P(vocal) |", "|---|---|---|"]
    for r in rows:
        sid = r["section_id"]
        if r["is_vocal"]:
            sid = f"**{sid}** (vocal)"
        elif r["is_jam"]:
            sid = f"**{sid}** (jam)"
        lines.append(f"| {sid} | {_fmt(r['start_sec'])} "
                     f"| {_fmt_prob(r['median_prob'])} |")
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
    p.add_argument("--train-until", type=float, default=None,
                   help="Cut point for the timesplit protocol. Default: the "
                        "midpoint of the first jam section from the ground "
                        "truth.")
    args = p.parse_args(argv)

    import librosa

    sections = _load_sections(args.ground_truth)
    train_until = args.train_until
    if train_until is None:
        jam = next((s for s in sections
                    if str(s["section_id"]).startswith("jam")), None)
        if jam is None:
            p.error("--train-until not given and no jam_* section in "
                    "ground truth to derive it from")
        train_until = (float(jam["start"]) + float(jam["end"])) / 2.0
    test_sections = _clip_sections(sections, train_until)

    wav_path = _ensure_wav(args.audio, target_sr=MERT_SAMPLE_RATE)
    try:
        duration = float(librosa.get_duration(path=str(wav_path)))
        n_windows = int(duration // args.window_sec)
        times = np.arange(n_windows) * args.window_sec
        print(f"Audio: {duration:.1f}s -> {n_windows} windows of "
              f"{args.window_sec:.1f}s", flush=True)
        print(f"Timesplit protocol: train < {train_until:.1f}s, "
              f"test >= {train_until:.1f}s", flush=True)

        section_ids, labels = _window_labels(sections, times, args.window_sec)
        n_voc = int((labels == 1).sum())
        n_ins = int((labels == 0).sum())
        print(f"Labels: {n_voc} vocal windows, {n_ins} instrumental, "
              f"{int((labels == -1).sum())} unlabeled", flush=True)

        results: dict[str, dict] = {}
        smooth_w = max(1, int(round(args.smoothing_sec / args.window_sec)))

        def analyze(name: str, features: np.ndarray, valid: np.ndarray,
                    extra: dict) -> None:
            # PRIMARY: deployment-shaped timesplit.
            ts_probs, ts_auc = _timesplit_curve(
                features, labels, valid, times, train_until)
            ts_smoothed = _moving_average(ts_probs, smooth_w)
            ts_rows = _section_medians(test_sections, times, ts_probs)
            ts_contrast = _contrast(ts_rows)
            ts_sweep = _sweep(times, ts_smoothed, sections, ts_contrast,
                              train_until, args.verdict_tolerance_sec)
            # SECONDARY: strict leave-one-section-out.
            lo_probs, lo_auc = _loso_curve(
                features, section_ids, labels, valid)
            lo_smoothed = _moving_average(lo_probs, smooth_w)
            lo_rows = _section_medians(sections, times, lo_probs)
            lo_contrast = _contrast(lo_rows)
            lo_sweep = _sweep(times, lo_smoothed, sections, lo_contrast,
                              args.song_offset, args.verdict_tolerance_sec)
            results[name] = {
                "timesplit": {
                    "train_until_sec": train_until,
                    "auc": ts_auc, "contrast": ts_contrast,
                    "section_medians": ts_rows, "thresholds": ts_sweep,
                    "verse_5_detected_at": [r["label"] for r in ts_sweep
                                            if r["verse_5"]["detected"]],
                    "curve": _curve_to_json(times, ts_probs),
                    "smoothed_curve": _curve_to_json(times, ts_smoothed),
                },
                "loso": {
                    "auc": lo_auc, "contrast": lo_contrast,
                    "section_medians": lo_rows, "thresholds": lo_sweep,
                    "verse_5_detected_at": [r["label"] for r in lo_sweep
                                            if r["verse_5"]["detected"]],
                    "curve": _curve_to_json(times, lo_probs),
                    "smoothed_curve": _curve_to_json(times, lo_smoothed),
                },
                **extra,
            }
            print("  " + _headline(f"{name} [timesplit]", ts_contrast,
                                   ts_auc), flush=True)
            for row in ts_sweep:
                v5 = row["verse_5"]
                status = (f"detected, error {v5['error_sec']:+.2f}s"
                          if v5["detected"] else "NOT detected")
                print(f"    threshold {row['label']}: verse_5 {status} "
                      f"({row['n_onsets']} onsets)", flush=True)
            print("  " + _headline(f"{name} [loso]", lo_contrast, lo_auc),
                  flush=True)

        if args.features in ("both", "mel"):
            print("Extracting log-mel features (live-viable baseline) ...",
                  flush=True)
            mel_feats, mel_counts = _mel_features(
                wav_path, args.window_sec, n_windows)
            analyze("mel", mel_feats, mel_counts > 0, {})

        if args.features in ("both", "mert"):
            print("Extracting MERT hidden states (signal ceiling) ...",
                  flush=True)
            mert_feats, mert_counts = _mert_features(
                wav_path, args.window_sec, n_windows, args.chunk_sec)
            valid = mert_counts > 0
            print(f"Probing {mert_feats.shape[0]} MERT layers ...",
                  flush=True)
            layer_aucs = []
            for layer in range(mert_feats.shape[0]):
                _pr, ts_auc = _timesplit_curve(
                    mert_feats[layer], labels, valid, times, train_until)
                _pr2, lo_auc = _loso_curve(
                    mert_feats[layer], section_ids, labels, valid)
                layer_aucs.append({"layer": layer, "timesplit_auc": ts_auc,
                                   "loso_auc": lo_auc})
                print(f"  layer {layer:2d}: timesplit AUC = "
                      f"{'-' if ts_auc is None else f'{ts_auc:.3f}'}  "
                      f"loso AUC = "
                      f"{'-' if lo_auc is None else f'{lo_auc:.3f}'}",
                      flush=True)
            scored = [r for r in layer_aucs
                      if r["timesplit_auc"] is not None]
            best = max(scored, key=lambda r: r["timesplit_auc"])
            best_layer = best["layer"]
            print(f"Best MERT layer by timesplit AUC: {best_layer} "
                  f"({best['timesplit_auc']:.3f})", flush=True)
            analyze(f"mert_layer_{best_layer}", mert_feats[best_layer],
                    valid, {"layer_aucs": layer_aucs,
                            "best_layer": best_layer})
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
        "- PRIMARY protocol `timesplit` (deployment-shaped): train on "
        f"everything before {train_until:.1f}s (intro + verses 1-4 + first "
        "half of the jam -- what the app has heard by mid-jam), test on "
        "everything after. verse_5 is predicted by a head that never saw "
        "past the jam midpoint.",
        "- Secondary protocol `loso` (strict leave-one-section-out): "
        "reported for reference; with only two instrumental sections, "
        "holding out the jam leaves a lone 13s intro as instrumental "
        "training data, so weak jam scores there are expected.",
        "",
    ]
    for name, r in results.items():
        headline = _headline(f"{name} [timesplit]",
                             r["timesplit"]["contrast"],
                             r["timesplit"]["auc"])
        md.append(f"**{headline}**")
        md.append("")

    for name, r in results.items():
        md += [f"## Feature set: {name}", ""]
        if "layer_aucs" in r:
            md += ["MERT layer sweep (AUC per layer, both protocols):", "",
                   "| layer | timesplit AUC | loso AUC |", "|---|---|---|"]
            for row in r["layer_aucs"]:
                ts = ("-" if row["timesplit_auc"] is None
                      else f"{row['timesplit_auc']:.3f}")
                lo = ("-" if row["loso_auc"] is None
                      else f"{row['loso_auc']:.3f}")
                marker = " (best)" if row["layer"] == r["best_layer"] else ""
                md.append(f"| {row['layer']}{marker} | {ts} | {lo} |")
            md.append("")
        for proto, title in [("timesplit",
                              f"Protocol: timesplit (train < "
                              f"{train_until:.1f}s) -- PRIMARY"),
                             ("loso", "Protocol: leave-one-section-out -- "
                                      "secondary")]:
            pr = r[proto]
            proto_headline = _headline(f"{name} [{proto}]", pr["contrast"],
                                       pr["auc"])
            md += [f"### {title}", "", f"**{proto_headline}**", ""]
            md += _medians_markdown(pr["section_medians"])
            md += ["", f"Onset detection (smoothed curve, sustained >= "
                       f"{SUSTAIN_SEC}s):", ""]
            md += _sweep_markdown(pr["thresholds"],
                                  args.verdict_tolerance_sec,
                                  verse_1_in_test=(proto == "loso"))
            md.append("")

    md += ["## VERDICT", ""]
    for name, r in results.items():
        ts = r["timesplit"]
        c = ts["contrast"]["contrast"]
        det = ts["verse_5_detected_at"]
        auc_s = "-" if ts["auc"] is None else f"{ts['auc']:.3f}"
        c_s = "-" if c is None else f"{c:+.2f}"
        md.append(
            f"- {name} (timesplit): AUC {auc_s}, contrast {c_s}, verse_5 "
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
        "train_until_sec": train_until,
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
