"""
dtw_align.py
------------
Offline DTW alignment of a test audio file against a template's chroma
reference. Per-test-frame predictions go to a Markdown + JSON report
that mirrors validate_position.py's shape so the two signals can be
compared apples-to-apples during the chroma+DTW transition.

CLI:
    python tooling/dtw_align.py \\
        --template tooling/songs/peggy_o_aligned.json \\
        --audio path/to/test.mp3 \\
        [--ground-truth tooling/ground_truth/peggy_o.json] \\
        [--song-offset 0.0] \\
        [--output-md dtw_validation.md] \\
        [--output-json dtw_validation.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Optional

import librosa
import numpy as np

# detect_first_verse owns vocal-isolation + energy onset detection. We import
# its private helpers rather than reimplementing -- single source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_first_verse as dfv  # noqa: E402


log = logging.getLogger("dtw_align")


# ---------------------------------------------------------------------------
# Helpers (deliberately self-contained -- the spec asks us not to touch
# validate_position.py, so we duplicate small pieces of its audio + GT
# plumbing rather than coupling to its private API)
# ---------------------------------------------------------------------------

def _ensure_wav(audio_path: Path, target_sr: int) -> Path:
    """Pre-convert any input to a clean mono WAV at `target_sr` via ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
         "-ar", str(target_sr), "-ac", "1", "-f", "wav", str(wav_path)],
        check=True,
    )
    return wav_path


def _load_ground_truth(path: Path) -> list[tuple[str, float, float]]:
    data = json.loads(path.read_text())
    out: list[tuple[str, float, float]] = []
    for s in data.get("sections", []):
        if "section_id" in s and "start" in s and "end" in s:
            out.append((s["section_id"], float(s["start"]), float(s["end"])))
    return out


def _gt_section_at(t: float, gt: list[tuple[str, float, float]]) -> Optional[str]:
    """Return the GT section_id whose [start, end) contains t, or None.

    Underscore-prefixed sections ('_silence', '_noodling' on band recordings,
    etc.) are intentionally returned as None so they surface in the JSON as
    gt_section_id=null and drop out of accuracy. The 'excluded' count in
    the report is the difference between rows and rows-with-non-null GT.
    """
    for sid, gs, ge in gt:
        if gs <= t < ge:
            if sid.startswith("_"):
                return None
            return sid
    return None


def _is_excluded(section_id: Optional[str]) -> bool:
    return section_id is None or section_id.startswith("_")


def _fmt_mmss(t: float) -> str:
    m = int(t) // 60
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


# ---------------------------------------------------------------------------
# Chroma + DTW
# ---------------------------------------------------------------------------

def _read_chroma_reference(template: dict) -> tuple[np.ndarray, int, int, float]:
    """Return (matrix shape (12, n_frames), sample_rate, hop_length, fps).

    Note the matrix is transposed for librosa.sequence.dtw which expects
    (n_features, n_frames). The JSON stores it row-per-frame.
    """
    chroma = template.get("chroma_reference")
    if not chroma or "data" not in chroma:
        raise RuntimeError(
            "Template has no chroma_reference field. Run tooling/chroma_template.py "
            "against this template first."
        )
    data = np.array(chroma["data"], dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 12:
        raise RuntimeError(
            f"chroma_reference.data shape {data.shape!r} is not (n_frames, 12)."
        )
    sr = int(chroma["sample_rate"])
    hop = int(chroma["hop_length"])
    fps = float(chroma["frames_per_sec"])
    return data.T, sr, hop, fps


def _section_lookup(section_map: list[tuple[str, int, int]],
                    ref_frame: int) -> Optional[str]:
    """First section in `section_map` whose [chroma_start_frame, chroma_end_frame)
    contains `ref_frame`. Returns None if no section covers it.

    `section_map` is built once at startup by `_build_section_map` so the
    DTW per-frame loop doesn't re-walk template["structure"] for every
    test frame.
    """
    for sid, s, e in section_map:
        if s <= ref_frame < e:
            return sid
    return None


def _log_section_bounds(template: dict) -> None:
    """At startup, dump every section's bounds so the chroma <-> section
    handoff is visible whether the indices were stamped by
    chroma_template.py or derived on the fly below.
    """
    sections = template.get("structure") or []
    log.info("Template sections found: %d", len(sections))
    for s in sections:
        sid = s.get("section_id", "?")
        log.info(
            "  %s: start_time=%s end_time=%s "
            "chroma_start_frame=%s chroma_end_frame=%s",
            sid,
            s.get("start_time"),
            s.get("end_time"),
            s.get("chroma_start_frame"),
            s.get("chroma_end_frame"),
        )
        if s.get("chroma_start_frame") is None or s.get("chroma_end_frame") is None:
            log.warning(
                "  WARNING: section %s has no chroma frame indices", sid,
            )


def _build_section_map(template: dict,
                       fps: float
                       ) -> list[tuple[str, int, int]]:
    """Build the (section_id, chroma_start_frame, chroma_end_frame) list
    the per-frame lookup uses.

    Preference order per section:
      1. Stamped fields (chroma_start_frame / chroma_end_frame).
      2. Derived from start_time / end_time and the chroma_reference's fps.
      3. Skip the section -- it can't be mapped to any frame range.

    Logs a one-time WARNING when any section had to use the derived
    fallback, so the user knows to re-run build-chroma to bake the
    indices in.
    """
    out: list[tuple[str, int, int]] = []
    fallback_triggered = False
    for section in template.get("structure") or []:
        sid = section.get("section_id")
        if sid is None:
            continue
        cs = section.get("chroma_start_frame")
        ce = section.get("chroma_end_frame")
        if cs is not None and ce is not None:
            out.append((sid, int(cs), int(ce)))
            continue
        st = section.get("start_time")
        et = section.get("end_time")
        if st is not None and et is not None:
            fallback_triggered = True
            out.append((
                sid,
                int(round(float(st) * fps)),
                int(round(float(et) * fps)),
            ))
    if fallback_triggered:
        log.warning(
            "WARNING: at least one section missing chroma_start_frame/"
            "chroma_end_frame; deriving from start_time/end_time. Re-run "
            "build-chroma to bake them in.",
        )
    return out


def _compute_test_chroma(audio_path: Path, sample_rate: int, hop_length: int,
                         song_offset: float) -> np.ndarray:
    """Decode audio via ffmpeg-pre-convert (matches validate_position's
    defensive approach), drop the first `song_offset` seconds, and return
    a (12, n_frames) chroma_cqt matrix at the reference's hop_length.
    """
    wav_path = _ensure_wav(audio_path, sample_rate)
    try:
        y, _ = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
    if song_offset > 0:
        skip = int(round(song_offset * sample_rate))
        if skip >= len(y):
            return np.zeros((12, 0), dtype=np.float32)
        y = y[skip:]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sample_rate, hop_length=hop_length)
    return chroma.astype(np.float32)


def _run_dtw(test_chroma: np.ndarray, ref_chroma: np.ndarray) -> np.ndarray:
    """Returns the warping path as an (L, 2) array of (test_idx, ref_idx)
    pairs in forward time order (path starts at (0, 0)).
    """
    _D, wp = librosa.sequence.dtw(
        X=test_chroma, Y=ref_chroma, subseq=False, backtrack=True,
    )
    # librosa returns the path end-to-start.
    return np.asarray(wp[::-1], dtype=np.int64)


def _test_to_ref_mapping(wp: np.ndarray, n_test: int) -> np.ndarray:
    """For each test frame, the first reference frame it's mapped to in the
    warping path. (-1 for frames that never appeared, which shouldn't
    happen under subseq=False but we degrade gracefully.)
    """
    mapping = np.full(n_test, -1, dtype=np.int64)
    for test_i, ref_j in wp:
        idx = int(test_i)
        if 0 <= idx < n_test and mapping[idx] < 0:
            mapping[idx] = int(ref_j)
    # Forward-fill any -1 gaps from the previous valid mapping; the
    # warping path is monotone, so propagating last-seen is safe.
    last = -1
    for i in range(n_test):
        if mapping[i] < 0:
            mapping[i] = last
        else:
            last = mapping[i]
    return mapping


# ---------------------------------------------------------------------------
# Re-anchoring via vocal-onset detection
# ---------------------------------------------------------------------------

REANCHOR_SEARCH_WINDOW_SEC = 45.0
REANCHOR_FALLBACK_THRESHOLD_DB = -35.0


def _dtw_guess_test_time(test_to_ref: np.ndarray, target_ref_frame: int,
                         fps: float, song_offset: float) -> Optional[float]:
    """Where does the single-pass DTW currently *think* `target_ref_frame`
    happens in the test audio? Returns the absolute test time (seconds) of
    the first test frame whose mapping crosses target_ref_frame, or None
    when DTW never reaches it.
    """
    for i, rf in enumerate(test_to_ref):
        if rf >= target_ref_frame:
            return float(i) / fps + song_offset
    return None


def _detect_reanchor_onsets(audio_path: Path,
                            sections_to_anchor: list[str],
                            template_sections: list[dict],
                            test_to_ref: np.ndarray,
                            fps: float,
                            song_offset: float,
                            separator: str = "demucs"
                            ) -> tuple[list[tuple[str, float, float]], dict]:
    """For each named section, find a vocal onset in the test audio within
    ±REANCHOR_SEARCH_WINDOW_SEC of DTW's current guess for that section.

    Reuses tooling/detect_first_verse: the separator runs once, the smoothed
    envelope is computed once, and the auto-threshold is calibrated from
    a single baseline window so all anchor searches share consistent
    sensitivity.

    `separator` picks which model produces the vocal stem the onset search
    runs on: "demucs" (htdemucs, the known-good default) or "open_unmix"
    (umxhq, the lower-quality real-time/browser-grade proxy from
    stem_quality_probe). Everything downstream of the stem -- the search
    window, the energy-onset logic, the segmented DTW -- is identical.

    Returns (anchors, info) where:
      anchors: list of (section_id, ref_start_sec, detected_test_sec)
               for sections where detection succeeded
      info: per-section dict including search window, detected onset,
            and reason on failure -- for the report.
    """
    section_starts = {
        s["section_id"]: s.get("start_time")
        for s in template_sections
    }

    info: dict = {"per_section": [], "separator": separator,
                  "demucs_used": separator == "demucs",
                  "threshold_db": None, "baseline_db": None,
                  "fallback_triggered": False, "vocal_rms_db": None}

    sep_wav: Optional[Path] = None
    if separator == "open_unmix":
        # stem_quality_probe owns the Open-Unmix path; its separator reads
        # via soundfile, so pre-convert the (possibly extensionless) input
        # to the 44.1 kHz WAV it expects. Demucs decodes inputs itself.
        import stem_quality_probe as sqp  # noqa: PLC0415
        log.info("Re-anchor: isolating vocals via Open-Unmix (%s) ...",
                 sqp.OPENUNMIX_MODEL)
        sep_wav = sqp._ensure_separation_wav(audio_path)
        vocals_path, stem_dir, vocal_rms_db = sqp._separate_vocals_openunmix(sep_wav)
    else:
        log.info("Re-anchor: isolating vocals via Demucs ...")
        vocals_path, stem_dir, vocal_rms_db = dfv._separate_vocals(audio_path)
    info["vocal_rms_db"] = round(vocal_rms_db, 2)
    log.info("Re-anchor: vocal stem RMS %.1f dB", vocal_rms_db)
    try:
        times, smoothed = dfv._smoothed_envelope(vocals_path)
        auto = dfv._compute_auto_threshold(
            times, smoothed,
            tap_time=song_offset,
            window_sec=dfv.AUTO_BASELINE_WINDOW_SEC,
            offset_db=dfv.AUTO_THRESHOLD_OFFSET_DB,
            fallback_db=REANCHOR_FALLBACK_THRESHOLD_DB,
        )
        threshold_db = float(auto["threshold_db"])
        info["threshold_db"] = round(threshold_db, 2)
        info["baseline_db"] = (
            round(float(auto["baseline_db"]), 2)
            if auto.get("baseline_db") is not None else None
        )
        info["fallback_triggered"] = bool(auto["fallback_triggered"])
        log.info(
            "Re-anchor threshold: %.1f dB (baseline %s, fallback=%s)",
            threshold_db,
            f"{auto['baseline_db']:.1f}" if auto.get("baseline_db") is not None else "n/a",
            auto["fallback_triggered"],
        )

        anchors: list[tuple[str, float, float]] = []
        for sid in sections_to_anchor:
            ref_start = section_starts.get(sid)
            entry: dict = {"section_id": sid, "ref_start_sec": ref_start}
            if ref_start is None:
                entry["reason"] = "section has no start_time in template"
                log.warning("Re-anchor: %s missing start_time; skipping", sid)
                info["per_section"].append(entry)
                continue
            target_ref_frame = int(round(float(ref_start) * fps))
            t_guess = _dtw_guess_test_time(test_to_ref, target_ref_frame, fps, song_offset)
            entry["dtw_guess_sec"] = round(t_guess, 2) if t_guess is not None else None
            if t_guess is None:
                entry["reason"] = "DTW never reached the section in single-pass"
                log.warning("Re-anchor: no DTW guess for %s; skipping", sid)
                info["per_section"].append(entry)
                continue

            search_lo = max(song_offset, t_guess - REANCHOR_SEARCH_WINDOW_SEC)
            search_hi = t_guess + REANCHOR_SEARCH_WINDOW_SEC
            entry["search_window_sec"] = [round(search_lo, 2), round(search_hi, 2)]
            mask = (times >= search_lo) & (times < search_hi)
            sub_times = times[mask]
            sub_db = smoothed[mask]
            if len(sub_times) == 0:
                entry["reason"] = "envelope has no samples in search window"
                log.warning("Re-anchor: empty envelope window for %s", sid)
                info["per_section"].append(entry)
                continue
            onset = dfv._find_sustained_crossing(
                sub_times, sub_db, threshold_db,
                dfv.ENERGY_SUSTAIN_WINDOW_SEC,
                dfv.ENERGY_SUSTAIN_TOLERANCE_DB,
                tap_time=search_lo,
            )
            if onset is None:
                entry["reason"] = (
                    f"no vocal onset above {threshold_db:.1f} dB sustained for "
                    f"{dfv.ENERGY_SUSTAIN_WINDOW_SEC:.1f}s in window"
                )
                log.warning(
                    "Re-anchor: no vocal onset for %s in [%.1f, %.1f]s; "
                    "falling back to single-pass at this boundary",
                    sid, search_lo, search_hi,
                )
                info["per_section"].append(entry)
                continue
            entry["detected_onset_sec"] = round(float(onset), 2)
            log.info(
                "Re-anchor: %s detected test onset %.2fs -> reference %.1fs",
                sid, onset, ref_start,
            )
            anchors.append((sid, float(ref_start), float(onset)))
            info["per_section"].append(entry)
    finally:
        import shutil  # noqa: PLC0415
        shutil.rmtree(stem_dir, ignore_errors=True)
        if sep_wav is not None:
            try:
                os.unlink(sep_wav)
            except OSError:
                pass

    return anchors, info


def _segmented_dtw(test_chroma: np.ndarray, ref_chroma: np.ndarray,
                   anchors: list[tuple[str, float, float]],
                   fps: float, song_offset: float
                   ) -> tuple[np.ndarray, list[dict]]:
    """Run DTW once per segment defined by the anchors. Returns the
    concatenated forward warping path plus a list of segment-info dicts
    for the report. Anchor frame indices are clamped to the available
    range and dedupe'd to keep segments strictly monotone.

    Frame indices in the returned path are relative to the start of
    test_chroma (frame 0 = song_offset wall-clock), same convention as
    single-pass output.
    """
    n_test = int(test_chroma.shape[1])
    n_ref = int(ref_chroma.shape[1])
    boundaries: list[tuple[int, int]] = [(0, 0)]
    for _sid, ref_t, test_t in sorted(anchors, key=lambda x: x[2]):
        tf = int(round((test_t - song_offset) * fps))
        rf = int(round(ref_t * fps))
        tf = max(0, min(tf, n_test))
        rf = max(0, min(rf, n_ref))
        if tf > boundaries[-1][0] and rf > boundaries[-1][1]:
            boundaries.append((tf, rf))
        else:
            log.warning(
                "Re-anchor: skipping non-monotone anchor (test_frame=%d, ref_frame=%d) "
                "after previous (%d, %d)",
                tf, rf, boundaries[-1][0], boundaries[-1][1],
            )
    if boundaries[-1] != (n_test, n_ref):
        boundaries.append((n_test, n_ref))

    full_wp_parts: list[np.ndarray] = []
    segments: list[dict] = []
    for (t_lo, r_lo), (t_hi, r_hi) in zip(boundaries, boundaries[1:]):
        if t_hi <= t_lo or r_hi <= r_lo:
            continue
        sub_test = test_chroma[:, t_lo:t_hi]
        sub_ref = ref_chroma[:, r_lo:r_hi]
        sub_wp = _run_dtw(sub_test, sub_ref)
        sub_wp_abs = sub_wp + np.array([t_lo, r_lo], dtype=np.int64)
        full_wp_parts.append(sub_wp_abs)
        segments.append({
            "test_frames": [int(t_lo), int(t_hi)],
            "ref_frames": [int(r_lo), int(r_hi)],
            "test_sec": [round(t_lo / fps + song_offset, 2),
                         round(t_hi / fps + song_offset, 2)],
            "ref_sec": [round(r_lo / fps, 2), round(r_hi / fps, 2)],
        })
    if not full_wp_parts:
        return np.zeros((0, 2), dtype=np.int64), segments
    return np.concatenate(full_wp_parts, axis=0), segments


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _accuracy_block(rows: list[dict]) -> tuple[dict, list[str]]:
    """Returns (stats_dict, md_lines)."""
    scored = [r for r in rows if not _is_excluded(r.get("gt_section_id"))]
    excluded = len(rows) - len(scored)
    n = len(scored)
    correct = sum(1 for r in scored if r["predicted_section"] == r["gt_section_id"])
    acc = (100.0 * correct / n) if n else 0.0

    per_section: dict[str, dict[str, int]] = {}
    confusions: Counter = Counter()
    for r in scored:
        gt = r["gt_section_id"]
        b = per_section.setdefault(gt, {"total": 0, "correct": 0})
        b["total"] += 1
        if r["predicted_section"] == gt:
            b["correct"] += 1
        else:
            confusions[(gt, r["predicted_section"])] += 1

    section_order: list[str] = []
    for r in scored:
        if r["gt_section_id"] not in section_order:
            section_order.append(r["gt_section_id"])

    md: list[str] = [
        "## Accuracy vs ground truth",
        "",
        f"- DTW-predicted section accuracy: {acc:.1f}%",
        f"- Excluded windows: {excluded}",
        "",
        "### Per-section accuracy",
        "",
        "| section_id | accuracy | window count |",
        "| --- | ---: | ---: |",
    ]
    for sid in section_order:
        b = per_section[sid]
        pct = (100.0 * b["correct"] / b["total"]) if b["total"] else 0.0
        md.append(f"| `{sid}` | {pct:.1f}% | {b['total']} |")

    md += ["", "### Top confusions", ""]
    if confusions:
        for (gt, pred), c in confusions.most_common(5):
            label = f"`{pred}`" if pred is not None else "_(none)_"
            md.append(f"- `{gt}` → {label}: {c} windows")
    else:
        md.append("- _none_")

    stats = {
        "accuracy_pct": round(acc, 2),
        "excluded_windows": excluded,
        "scored_windows": n,
        "per_section_accuracy": [
            {
                "section_id": sid,
                "accuracy_pct": round(100.0 * per_section[sid]["correct"]
                                      / max(1, per_section[sid]["total"]), 2),
                "window_count": per_section[sid]["total"],
            }
            for sid in section_order
        ],
        "top_confusions": [
            {"true": gt, "predicted": pred, "windows": c}
            for (gt, pred), c in confusions.most_common(5)
        ],
    }
    return stats, md


def _build_report(template: dict, audio_path: Path,
                  chroma_meta: dict,
                  rows: list[dict],
                  song_offset: float,
                  gt_enabled: bool,
                  reanchor_info: Optional[dict] = None) -> tuple[str, dict]:
    by_section = Counter(r["predicted_section"] for r in rows)
    n = len(rows)

    out: list[str] = [
        "# DTW alignment report",
        "",
        f"- Template: `{template.get('song_id', '?')}`",
        f"- Audio: `{audio_path.name}`",
        f"- Chroma: {chroma_meta['feature_type']} at "
        f"{chroma_meta['frames_per_sec']:.1f} frames/sec",
        f"- Test frames: {chroma_meta['n_test_frames']}",
        f"- Reference frames: {chroma_meta['n_ref_frames']}",
        f"- Song offset: {song_offset:.1f}s",
    ]

    if reanchor_info is not None:
        requested = [e["section_id"] for e in reanchor_info.get("per_section", [])]
        anchors = reanchor_info.get("anchors", [])
        segments = reanchor_info.get("segments", [])
        separator_labels = {
            "demucs": "demucs (htdemucs, high-quality reference)",
            "open_unmix": "open_unmix (umxhq, real-time/browser-grade proxy)",
        }
        separator = reanchor_info.get("separator", "demucs")
        out += ["", "## Re-anchoring", ""]
        out.append(
            f"- Onset-detection separator: "
            f"{separator_labels.get(separator, separator)}"
        )
        out.append(f"- Re-anchor sections: {', '.join(requested) or '(none)'}")
        if anchors:
            for a in anchors:
                out.append(
                    f"  - `{a['section_id']}`: detected test onset "
                    f"{a['detected_test_sec']}s → anchored to reference "
                    f"{a['ref_start_sec']}s"
                )
        # Note any sections that failed onset detection (no entry in `anchors`).
        anchored_ids = {a["section_id"] for a in anchors}
        for entry in reanchor_info.get("per_section", []):
            if entry["section_id"] not in anchored_ids:
                reason = entry.get("reason", "unknown")
                out.append(
                    f"  - `{entry['section_id']}`: re-anchor failed "
                    f"({reason}); fell back to single-pass for this boundary"
                )
        if segments:
            out.append(
                f"- Segmented DTW: {len(segments)} segment(s)"
            )
            for i, seg in enumerate(segments):
                out.append(
                    f"  - segment {i}: test {seg['test_sec'][0]}–"
                    f"{seg['test_sec'][1]}s → ref {seg['ref_sec'][0]}–"
                    f"{seg['ref_sec'][1]}s "
                    f"({seg['test_frames'][1]-seg['test_frames'][0]} test, "
                    f"{seg['ref_frames'][1]-seg['ref_frames'][0]} ref frames)"
                )

    stats: dict = {}
    accuracy_stats: Optional[dict] = None
    if gt_enabled:
        accuracy_stats, acc_md = _accuracy_block(rows)
        out += [""] + acc_md

    out += ["", "## Predicted section distribution", ""]
    if n == 0:
        out.append("_No frames scored._")
    else:
        for sid, c in by_section.most_common():
            label = sid if sid is not None else "(none)"
            out.append(f"- `{label}`: {c} frames ({100*c/n:.1f}%)")

    out += ["", "## Timeline (every 5s)", ""]
    last_t = -10.0
    for r in rows:
        if r["test_time"] - last_t >= 5.0:
            sid = r["predicted_section"] or "(none)"
            out.append(
                f"- `{_fmt_mmss(r['test_time'])}` → **{sid}** "
                f"(reference_frame={r['reference_frame']})"
            )
            last_t = r["test_time"]

    report_json = {
        "template_id": template.get("song_id"),
        "audio": str(audio_path),
        "chroma": chroma_meta,
        "song_offset_sec": song_offset,
        "predicted_section_distribution": [
            {"section_id": sid, "frames": c} for sid, c in by_section.most_common()
        ],
        "warping_path": rows,
    }
    if accuracy_stats is not None:
        report_json["accuracy"] = accuracy_stats
    if reanchor_info is not None:
        report_json["reanchor"] = reanchor_info
    return "\n".join(out) + "\n", report_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--song-offset", type=float, default=0.0,
                   help="Seconds of pre-song audio to skip on the test side "
                        "before computing chroma (default 0.0).")
    p.add_argument("--output-md", type=Path, default=Path("dtw_validation.md"))
    p.add_argument("--output-json", type=Path, default=Path("dtw_validation.json"))
    p.add_argument("--reanchor-sections", default="",
                   help="Comma-separated section_ids to use as hard re-anchor "
                        "points via vocal-onset detection (e.g. 'verse_5'). "
                        "Each becomes a boundary at which the test audio is "
                        "split and DTW is run independently per segment. "
                        "Empty (default) keeps single-pass DTW behaviour.")
    p.add_argument("--reanchor-separator", choices=["demucs", "open_unmix"],
                   default="demucs",
                   help="Which separator produces the vocal stem that "
                        "re-anchor onset detection runs on. demucs (default): "
                        "htdemucs, the known-good high-quality path. "
                        "open_unmix: umxhq, the lower-quality real-time/"
                        "browser-grade proxy from stem_quality_probe. Only "
                        "used when --reanchor-sections is set.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    template = json.loads(args.template.read_text())
    _log_section_bounds(template)
    ref_chroma, sr, hop_length, fps = _read_chroma_reference(template)
    section_map = _build_section_map(template, fps)
    n_ref = int(ref_chroma.shape[1])
    log.info(
        "Reference: %d frames at %.1f fps (%.1fs) [sample_rate=%d, hop=%d]",
        n_ref, fps, n_ref / fps, sr, hop_length,
    )

    log.info("Loading test audio %s ...", args.audio)
    test_chroma = _compute_test_chroma(
        args.audio, sample_rate=sr, hop_length=hop_length,
        song_offset=args.song_offset,
    )
    n_test = int(test_chroma.shape[1])
    if n_test == 0:
        log.error("Test audio yielded zero frames (song offset past end?).")
        return 1
    log.info("Test: %d frames (%.1fs after song_offset)", n_test, n_test / fps)

    log.info("Running DTW...")
    wp = _run_dtw(test_chroma, ref_chroma)
    test_to_ref = _test_to_ref_mapping(wp, n_test)

    # Optional re-anchoring: detect vocal onsets at named sections, then
    # rerun DTW segment-by-segment so a per-section duration mismatch
    # (e.g. a band jam that's much shorter than the studio reference)
    # can't drag every subsequent section out of alignment.
    sections_to_anchor = [s.strip() for s in args.reanchor_sections.split(",") if s.strip()]
    reanchor_info: Optional[dict] = None
    segments_info: list[dict] = []
    if sections_to_anchor:
        log.info("Re-anchor: requested sections = %s (separator=%s)",
                 sections_to_anchor, args.reanchor_separator)
        anchors, reanchor_info = _detect_reanchor_onsets(
            args.audio,
            sections_to_anchor,
            template.get("structure") or [],
            test_to_ref, fps, args.song_offset,
            separator=args.reanchor_separator,
        )
        if anchors:
            log.info("Re-anchor: %d anchor(s) found; running segmented DTW",
                     len(anchors))
            wp, segments_info = _segmented_dtw(
                test_chroma, ref_chroma, anchors, fps, args.song_offset,
            )
            test_to_ref = _test_to_ref_mapping(wp, n_test)
            log.info("Re-anchor: %d segment(s) aligned independently",
                     len(segments_info))
        else:
            log.warning("Re-anchor: no usable anchors; keeping single-pass DTW")
        reanchor_info["anchors"] = [
            {"section_id": sid, "ref_start_sec": round(rt, 2),
             "detected_test_sec": round(tt, 2)}
            for sid, rt, tt in anchors
        ]
        reanchor_info["segments"] = segments_info

    # Ground truth (optional). Errors loudly when --ground-truth was given
    # but the file doesn't exist; otherwise downstream gt_section_id values
    # would silently be null and the user would think the workflow worked.
    gt_intervals: list[tuple[str, float, float]] = []
    gt_enabled = args.ground_truth is not None
    if gt_enabled:
        if not args.ground_truth.exists():
            log.error("ERROR: ground truth file not found: %s", args.ground_truth)
            return 1
        gt_intervals = _load_ground_truth(args.ground_truth)
        n_excluded = sum(1 for sid, _, _ in gt_intervals if sid.startswith("_"))
        log.info(
            "Ground truth: %s (%d sections, %d excluded)",
            args.ground_truth, len(gt_intervals), n_excluded,
        )
    else:
        log.info("Ground truth: none provided")

    rows: list[dict] = []
    for i in range(n_test):
        ref_frame = int(test_to_ref[i])
        test_time = float(i) / fps + args.song_offset
        ref_time = float(ref_frame) / fps if ref_frame >= 0 else None
        pred_section = _section_lookup(section_map, ref_frame) if ref_frame >= 0 else None
        gt_sid = _gt_section_at(test_time, gt_intervals) if gt_enabled else None
        rows.append({
            "test_frame": i,
            "test_time": round(test_time, 3),
            "reference_frame": ref_frame,
            "reference_time": round(ref_time, 3) if ref_time is not None else None,
            "predicted_section": pred_section,
            "gt_section_id": gt_sid,
        })

    chroma_meta = {
        "feature_type": template["chroma_reference"].get("feature_type"),
        "sample_rate": sr,
        "hop_length": hop_length,
        "frames_per_sec": fps,
        "n_test_frames": n_test,
        "n_ref_frames": n_ref,
    }

    md_text, report_json = _build_report(
        template, args.audio, chroma_meta, rows, args.song_offset, gt_enabled,
        reanchor_info=reanchor_info,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md_text)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report_json, indent=2))
    log.info("Wrote %s and %s (%d frames)",
             args.output_md, args.output_json, n_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
