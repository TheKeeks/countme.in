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
    for sid, gs, ge in gt:
        if gs <= t < ge:
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


def _section_lookup(template: dict, ref_frame: int) -> Optional[str]:
    """First section whose [chroma_start_frame, chroma_end_frame) contains
    ref_frame. Returns None if no section covers it (gap or template-end).
    """
    for section in template.get("structure", []):
        s = section.get("chroma_start_frame")
        e = section.get("chroma_end_frame")
        if s is None or e is None:
            continue
        if int(s) <= ref_frame < int(e):
            return section["section_id"]
    return None


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
                  gt_enabled: bool) -> tuple[str, dict]:
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
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    template = json.loads(args.template.read_text())
    ref_chroma, sr, hop_length, fps = _read_chroma_reference(template)
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

    # Ground truth (optional)
    gt_intervals: list[tuple[str, float, float]] = []
    gt_enabled = args.ground_truth is not None
    if gt_enabled:
        gt_intervals = _load_ground_truth(args.ground_truth)
        log.info("Loaded %d ground-truth intervals from %s",
                 len(gt_intervals), args.ground_truth)

    rows: list[dict] = []
    for i in range(n_test):
        ref_frame = int(test_to_ref[i])
        test_time = float(i) / fps + args.song_offset
        ref_time = float(ref_frame) / fps if ref_frame >= 0 else None
        pred_section = _section_lookup(template, ref_frame) if ref_frame >= 0 else None
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
