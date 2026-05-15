"""
validate_position.py
--------------------
Offline simulator of the (not-yet-built) live position tracker. Slides a
2-second window across an audio file with a 0.5-second hop, embeds each
window with MERT, and for each section asks: "how close is this window
to the closest frame in this section's audio_embedding_sequence?".

The window's predicted section is the one with the highest max-cosine
similarity. Confidence is the top score; margin is the gap to second
place. Writes three sibling reports with the same prefix:

    <out>.csv   per-window predictions (raw data for analysis)
    <out>.md    human-readable summary -- distribution, confidence,
                most-confused pairs, 5-second timeline
    <out>.html  inline-SVG timeline coloured by predicted section,
                opacity by confidence

Used to validate the embedding-matching approach end-to-end before we
port it into the browser runtime. Out of scope here: line-level
prediction and whisper fusion.

Adds an optional Viterbi smoothing pass on top of the per-window
argmax: a small section-order prior pulls the predicted sequence
toward "stay in this section, or move on to the next one", which
rescues short bursts of confusion between sections that sound alike.
Disable with `--transition-penalty 0.0`.

Adds an optional VAD pass that adjusts log-emissions before Viterbi:
when a window has vocals it's penalised against instrumental sections
(intro / jam / outro), and vice versa. webrtcvad does the per-frame
speech detection; the script aggregates with a majority vote inside
each 2 s scoring window. Disable with `--vad-penalty 0.0`.

Adds an optional time prior: at real-time stage use the user knows
which song is playing and when it started, so the reference timeline
already tells us which section to expect at each elapsed second. The
prior adds a bonus to the expected section's log-emission before
Viterbi runs. Disable with `--time-prior-weight 0.0`.

CLI:
    python tooling/validate_position.py \\
        --template web/templates/peggy_o_aligned.json \\
        --audio path/to/band_recording.mp3 \\
        --out validation_report
"""

from __future__ import annotations

import argparse
import csv
import html as html_lib
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

# build_song lives next to this file -- reuse its EmbeddingExtractor so
# the model + sample rate match exactly what produced the template.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_song as bs  # noqa: E402

log = logging.getLogger("validate_position")

WINDOW_SEC = 2.0
HOP_SEC = 0.5
EMBED_SR = bs.EMBED_SAMPLE_RATE  # 24 kHz, matches MERT-v1-95M


# ---------------------------------------------------------------------------
# Embedding + scoring helpers
# ---------------------------------------------------------------------------

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-9 else v / n


FORMAT_EXTENSIONS = {
    "mp3": ".mp3",
    "mp4": ".m4a",
    "wav": ".wav",
    "flac": ".flac",
    "ogg": ".ogg",
    "aac": ".aac",
    "caf": ".caf",
}


def _detect_audio_format(path: Path) -> Optional[str]:
    """Sniff the actual container format from file magic bytes.

    Returns one of: 'mp3', 'mp4', 'wav', 'flac', 'ogg', 'aac', 'caf',
    or None if the header doesn't match anything we know.
    """
    with open(path, "rb") as f:
        header = f.read(64)
    if header.startswith(b"ID3"):
        return "mp3"
    # AAC's ADTS sync (\xff\xf1 / \xff\xf9) also satisfies the loose MP3 rule
    # below, so check AAC first.
    if header.startswith(b"\xff\xf1") or header.startswith(b"\xff\xf9"):
        return "aac"
    if len(header) >= 2 and header[0] == 0xFF and (header[1] & 0xE0) == 0xE0:
        return "mp3"
    if b"ftyp" in header[:20]:
        return "mp4"   # m4a, mp4, voice memos
    if header.startswith(b"RIFF") and b"WAVE" in header[:12]:
        return "wav"
    if header.startswith(b"fLaC"):
        return "flac"
    if header.startswith(b"OggS"):
        return "ogg"   # ogg vorbis or opus
    if header.startswith(b"caff"):
        return "caf"   # Apple Core Audio
    return None


def _load_audio(path: Path, sr: int = EMBED_SR) -> np.ndarray:
    """Decode `path` to mono float32 at `sr`, going through ffmpeg first.

    librosa+audioread chokes on some non-studio MP3 encodings (the rehearsal
    recording that triggered this divided by a sample_rate of 0). Pre-converting
    to a clean WAV via ffmpeg sidesteps that path entirely. We also sniff the
    container format from magic bytes up front so a misleading filename
    extension (or no extension at all) doesn't trip ffmpeg's autodetection,
    and ffprobe the (possibly renamed) file so the workflow log records
    what we actually got.
    """
    import json as _json
    import os
    import shutil
    import subprocess
    import tempfile

    import librosa

    detected = _detect_audio_format(path)
    print(
        f"  detected audio format: {detected or 'unknown'} "
        f"(filename was: {path.name})",
        flush=True,
    )

    cleanup_paths: list[str] = []
    probe_path = str(path)
    if detected is not None:
        target_ext = FORMAT_EXTENSIONS[detected]
        if path.suffix.lower() != target_ext:
            renamed = tempfile.NamedTemporaryFile(suffix=target_ext, delete=False)
            renamed.close()
            shutil.copy(str(path), renamed.name)
            probe_path = renamed.name
            cleanup_paths.append(renamed.name)
            print(
                f"  copied to {Path(renamed.name).name} so ffmpeg sees the "
                f"right extension",
                flush=True,
            )
    else:
        print(
            "  could not identify format from magic bytes; trying ffmpeg anyway",
            flush=True,
        )

    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json",
             "-show_format", "-show_streams", probe_path],
            capture_output=True, text=True, check=True,
        )
        meta = _json.loads(probe.stdout)
        streams = meta.get("streams", [])
        if streams:
            s0 = streams[0]
            print(
                f"  audio metadata: codec={s0.get('codec_name')}, "
                f"sample_rate={s0.get('sample_rate')}, "
                f"channels={s0.get('channels')}, "
                f"duration={meta.get('format', {}).get('duration')}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001 -- diagnostic only
        print(f"  ffprobe failed (continuing anyway): {exc}", flush=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    cleanup_paths.append(wav_path)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", probe_path,
             "-ar", str(sr), "-ac", "1", "-f", "wav", wav_path],
            check=True,
        )
        y, _ = librosa.load(wav_path, sr=sr, mono=True)
    finally:
        for p in cleanup_paths:
            if os.path.exists(p):
                os.unlink(p)
    return y


def _collect_section_frames(template: dict) -> list[tuple[str, str, np.ndarray]]:
    """Return [(section_id, section_type, frames_matrix), ...] for sections
    with non-empty embedding sequences. Frames are re-L2-normalized
    defensively so cosine similarity is just a dot product."""
    out: list[tuple[str, str, np.ndarray]] = []
    for s in template.get("structure", []):
        seq = s.get("audio_embedding_sequence") or []
        if not seq:
            continue
        frames = np.array(
            [_l2_norm(np.array(f, dtype=np.float32)) for f in seq],
            dtype=np.float32,
        )
        out.append((s["section_id"], s.get("section_type", ""), frames))
    return out


def _score_window(query_vec: np.ndarray,
                  sections: list[tuple[str, str, np.ndarray]]) -> np.ndarray:
    """Max-cosine similarity per section. Returns shape (n_sections,)."""
    out = np.empty(len(sections), dtype=np.float32)
    for i, (_sid, _stype, frames) in enumerate(sections):
        out[i] = float((frames @ query_vec).max())
    return out


# ---------------------------------------------------------------------------
# Viterbi smoothing
# ---------------------------------------------------------------------------

SOFTMAX_TEMPERATURE = 0.1
FORWARD_LOG_PROB = -1.0


def _log_emissions(sims_matrix: np.ndarray,
                   temperature: float = SOFTMAX_TEMPERATURE) -> np.ndarray:
    """Per-window softmax(sims/T) in log space. Shape (W, S) -> (W, S)."""
    scaled = sims_matrix / temperature
    # logsumexp along axis=1, numerically stable.
    m = scaled.max(axis=1, keepdims=True)
    lse = m.squeeze(-1) + np.log(np.exp(scaled - m).sum(axis=1))
    return scaled - lse[:, None]


def _transition_log_probs(n_sections: int, transition_penalty: float) -> np.ndarray:
    """log P(j | i) under the simple section-order prior.

    Stay-in-section: 0. Forward to the next section in song order: -1.0.
    Any other jump (including backward): -transition_penalty.
    The last section has no "next", so all off-diagonals from it pay the
    full penalty.
    """
    t = np.full((n_sections, n_sections), -transition_penalty, dtype=np.float64)
    for i in range(n_sections):
        t[i, i] = 0.0
        if i + 1 < n_sections:
            t[i, i + 1] = FORWARD_LOG_PROB
    return t


def _viterbi(log_emissions: np.ndarray,
             log_transitions: np.ndarray) -> np.ndarray:
    """Standard Viterbi. log_emissions: (W, S). log_transitions: (S, S).

    Returns the most-likely state sequence as a (W,) int array.
    """
    n_windows, n_states = log_emissions.shape
    if n_windows == 0:
        return np.zeros(0, dtype=np.int32)
    dp = np.empty((n_windows, n_states), dtype=np.float64)
    back = np.zeros((n_windows, n_states), dtype=np.int32)
    dp[0] = log_emissions[0]
    for t in range(1, n_windows):
        # scores[prev, curr] = dp[t-1, prev] + log P(curr | prev)
        scores = dp[t - 1][:, None] + log_transitions
        back[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emissions[t]
    seq = np.empty(n_windows, dtype=np.int32)
    seq[-1] = int(dp[-1].argmax())
    for t in range(n_windows - 2, -1, -1):
        seq[t] = back[t + 1, seq[t + 1]]
    return seq


def _count_changes(seq) -> int:
    return sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])


def _mean_run_length_sec(seq, hop_sec: float) -> float:
    if not len(seq):
        return 0.0
    n_runs = 1 + _count_changes(seq)
    return len(seq) * hop_sec / n_runs


def _changed_pct(raw_seq, smoothed_seq) -> float:
    n = len(raw_seq)
    if n == 0:
        return 0.0
    diffs = sum(1 for r, s in zip(raw_seq, smoothed_seq) if r != s)
    return 100.0 * diffs / n


# ---------------------------------------------------------------------------
# Voice activity detection
# ---------------------------------------------------------------------------

VAD_SAMPLE_RATE = 16000   # webrtcvad supports 8 / 16 / 32 / 48 kHz
VAD_FRAME_MS = 30         # 10 / 20 / 30 only
VAD_AGGRESSIVENESS = 2    # 0..3, higher = stricter (fewer false positives)


def _compute_vad(y: np.ndarray, sr: int, times: list[float],
                 window_sec: float) -> np.ndarray:
    """Per-window bool: True iff a majority of webrtcvad frames inside
    [t, t+window_sec) flagged speech.

    Resamples to 16 kHz int16 PCM in-memory; the input array is left alone.
    """
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "webrtcvad is required when --vad-penalty > 0. "
            "Install with `pip install webrtcvad`, or rerun with --vad-penalty 0.0."
        ) from exc
    import librosa

    if sr != VAD_SAMPLE_RATE:
        y16 = librosa.resample(y, orig_sr=sr, target_sr=VAD_SAMPLE_RATE)
    else:
        y16 = y
    y16_int = (np.clip(y16, -1.0, 1.0) * 32767.0).astype(np.int16)

    frame_samples = int(VAD_SAMPLE_RATE * VAD_FRAME_MS / 1000)  # 480 @ 16k/30ms
    n_frames = len(y16_int) // frame_samples
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    frame_flags = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        s = i * frame_samples
        frame_bytes = y16_int[s:s + frame_samples].tobytes()
        try:
            frame_flags[i] = vad.is_speech(frame_bytes, VAD_SAMPLE_RATE)
        except Exception:  # noqa: BLE001 -- webrtcvad raises on odd inputs
            frame_flags[i] = False

    has_vocals = np.zeros(len(times), dtype=bool)
    for w, t in enumerate(times):
        f0 = int(t * 1000 // VAD_FRAME_MS)
        f1 = int((t + window_sec) * 1000 // VAD_FRAME_MS)
        f0 = max(0, min(f0, n_frames))
        f1 = max(f0, min(f1, n_frames))
        if f1 > f0:
            has_vocals[w] = frame_flags[f0:f1].sum() > (f1 - f0) / 2
    return has_vocals


def _vocal_section_mask(template: dict,
                        sections: list[tuple[str, str, np.ndarray]]) -> np.ndarray:
    """True for sections that have at least one lyric line in the template."""
    lines_count = {
        s["section_id"]: len(s.get("lines", []) or [])
        for s in template.get("structure", [])
    }
    return np.array(
        [lines_count.get(sid, 0) > 0 for sid, _stype, _frames in sections],
        dtype=bool,
    )


def _apply_vad_penalty(log_emissions: np.ndarray,
                       has_vocals: np.ndarray,
                       is_vocal_section: np.ndarray,
                       penalty: float) -> np.ndarray:
    """Subtract `penalty` from log_emissions[w, s] when the window's vocal
    state disagrees with the section's vocal-vs-instrumental classification.
    """
    if penalty <= 0:
        return log_emissions
    mismatch = has_vocals[:, None] != is_vocal_section[None, :]
    return log_emissions - penalty * mismatch.astype(log_emissions.dtype)


# ---------------------------------------------------------------------------
# Time prior
# ---------------------------------------------------------------------------

def _section_time_ranges(template: dict,
                         sections: list[tuple[str, str, np.ndarray]]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends) in seconds, parallel to `sections`.

    Reads the explicit `start_time` / `end_time` fields build_song stamps on
    every section (including pure-instrumental ones). Falls back to the
    legacy fields and to line bounds for backward compatibility with
    templates built before those fields existed.
    """
    section_data = {
        s["section_id"]: s for s in template.get("structure", [])
    }
    starts = np.full(len(sections), np.nan, dtype=np.float64)
    ends = np.full(len(sections), np.nan, dtype=np.float64)
    for i, (sid, _stype, _frames) in enumerate(sections):
        s = section_data.get(sid)
        if s is None:
            continue
        if s.get("start_time") is not None and s.get("end_time") is not None:
            starts[i] = float(s["start_time"])
            ends[i] = float(s["end_time"])
            continue
        # Backward-compat fallback for templates predating start_time/end_time.
        lines = s.get("lines") or []
        line_starts = [l.get("start_sec") for l in lines
                       if l.get("start_sec") is not None]
        line_ends = [l.get("end_sec") for l in lines
                     if l.get("end_sec") is not None]
        if line_starts and line_ends:
            starts[i] = float(min(line_starts))
            ends[i] = float(max(line_ends))
            continue
        if s.get("start_sec") is not None:
            starts[i] = float(s["start_sec"])
        if s.get("end_sec") is not None:
            ends[i] = float(s["end_sec"])
    return starts, ends


def _expected_section_at(t: float,
                         starts: np.ndarray,
                         ends: np.ndarray) -> int:
    """Index of the section whose half-open [start_time, end_time) range
    contains `t`. If no range contains it (gap or out-of-range), fall back
    to the section whose midpoint is closest to `t`.
    """
    n = len(starts)
    if n == 0:
        return 0
    valid_pair = ~(np.isnan(starts) | np.isnan(ends))
    if valid_pair.any():
        contains = np.where(valid_pair & (starts <= t) & (t < ends))[0]
        if len(contains) > 0:
            return int(contains[0])
        mids = (starts + ends) / 2.0
        diffs = np.abs(mids - t)
        diffs[~valid_pair] = np.inf
        return int(np.argmin(diffs))
    # Degenerate: no section has a usable range. Default to first.
    return 0


def _expected_indices(times: list[float],
                      starts: np.ndarray,
                      ends: np.ndarray,
                      window_sec: float,
                      song_offset: float = 0.0) -> np.ndarray:
    """Per-window expected-section index, computed at the window's centre.

    The window centre minus `song_offset` becomes the effective time used
    against the template's section ranges. Windows whose effective time is
    negative (pre-song content -- band tuning, chatter, etc.) get index -1
    so the time prior contributes nothing for them.
    """
    n = len(times)
    out = np.full(n, -1, dtype=np.int32)
    for w, t in enumerate(times):
        eff = t + window_sec / 2.0 - song_offset
        if eff < 0:
            continue
        out[w] = _expected_section_at(eff, starts, ends)
    return out


def _apply_time_prior(log_emissions: np.ndarray,
                      expected_idx: np.ndarray,
                      weight: float) -> np.ndarray:
    """Add a bonus to the expected section at each window time.

    Only the expected section receives the bonus -- spreading half to the
    neighbours nullified the prior at section transitions (previous state +
    expected both got bonuses, so the smoother couldn't tell them apart).
    Windows with expected_idx == -1 (pre-song, before `song_offset`) get no
    bonus on any section.
    """
    if weight <= 0 or len(expected_idx) == 0:
        return log_emissions
    out = log_emissions.copy()
    for w, idx in enumerate(expected_idx):
        if idx >= 0:
            out[w, int(idx)] += weight
    return out


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> list[tuple[str, float, float]]:
    """Return [(section_id, start, end), ...] from a GT JSON file.

    The schema is the same one build_song.py reads:
        {"sections": [{"section_id": str, "start": num, "end": num, ...}, ...]}
    Section IDs starting with `_` (e.g. `_silence`) are kept in the list and
    later excluded from accuracy stats by the caller.
    """
    data = json.loads(path.read_text())
    out: list[tuple[str, float, float]] = []
    for s in data.get("sections", []):
        if "section_id" in s and "start" in s and "end" in s:
            out.append((s["section_id"], float(s["start"]), float(s["end"])))
    return out


def gt_section_at(t: float,
                  gt: list[tuple[str, float, float]]) -> Optional[str]:
    """Return the GT section_id whose half-open [start, end) contains t.

    Returns None when no interval matches (e.g. t past the labelled audio).
    """
    for sid, gs, ge in gt:
        if gs <= t < ge:
            return sid
    return None


def _is_excluded(section_id: Optional[str]) -> bool:
    return section_id is None or section_id.startswith("_")


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "time_sec", "predicted_section_id", "predicted_section_type",
            "confidence", "margin", "top3_sections_and_scores",
        ])
        for r in rows:
            top3 = ";".join(f"{sid}:{score:.3f}" for sid, _stype, score in r["top3"])
            w.writerow([
                f"{r['time']:.2f}", r["pred_id"], r["pred_type"],
                f"{r['conf']:.4f}", f"{r['margin']:.4f}", top3,
            ])


# ---------------------------------------------------------------------------
# Output: Markdown
# ---------------------------------------------------------------------------

def _fmt_mmss(t: float) -> str:
    m = int(t) // 60
    s = int(t) % 60
    return f"{m:02d}:{s:02d}"


def _accuracy_md_block(rows: list[dict]) -> list[str]:
    """Compute and format the 'Accuracy vs ground truth' section."""
    # Windows with a GT section (and not an excluded `_*` section).
    scored = [r for r in rows if not _is_excluded(r.get("gt_section_id"))]
    excluded = len(rows) - len(scored)
    n = len(scored)

    def _accuracy(field: str) -> float:
        if not scored:
            return 0.0
        return 100.0 * sum(1 for r in scored if r[field] == r["gt_section_id"]) / n

    raw_acc = _accuracy("pred_id")
    smooth_acc = _accuracy("smoothed_id")
    tp_acc = _accuracy("time_prior_id")

    # Per-section accuracy.
    per_section: dict[str, dict[str, int]] = {}
    for r in scored:
        sid = r["gt_section_id"]
        bucket = per_section.setdefault(sid, {"total": 0, "raw": 0, "smoothed": 0, "tp": 0})
        bucket["total"] += 1
        if r["pred_id"] == sid:
            bucket["raw"] += 1
        if r["smoothed_id"] == sid:
            bucket["smoothed"] += 1
        if r["time_prior_id"] == sid:
            bucket["tp"] += 1

    # Top confusions for raw and smoothed (true_gt != predicted).
    raw_conf: Counter = Counter()
    smooth_conf: Counter = Counter()
    for r in scored:
        gt = r["gt_section_id"]
        if r["pred_id"] != gt:
            raw_conf[(gt, r["pred_id"])] += 1
        if r["smoothed_id"] != gt:
            smooth_conf[(gt, r["smoothed_id"])] += 1

    out: list[str] = [
        "## Accuracy vs ground truth",
        "",
        f"- Raw (MERT argmax): {raw_acc:.1f}%",
        f"- Smoothed (Viterbi fusion): {smooth_acc:.1f}%",
        f"- Time-prior-only baseline: {tp_acc:.1f}%",
        f"- Excluded windows: {excluded}",
        "",
        "### Per-section accuracy",
        "",
        "| section_id | raw % | smoothed % | time-prior % |",
        "| --- | ---: | ---: | ---: |",
    ]
    # Sort by first time the section appears in GT for stable display.
    section_order: list[str] = []
    for r in scored:
        if r["gt_section_id"] not in section_order:
            section_order.append(r["gt_section_id"])
    for sid in section_order:
        b = per_section[sid]
        tot = b["total"]
        out.append(
            f"| `{sid}` | {100*b['raw']/tot:.1f}% | "
            f"{100*b['smoothed']/tot:.1f}% | "
            f"{100*b['tp']/tot:.1f}% |"
        )

    out += ["", "### Top confusions", ""]
    out.append("**Raw** (true → predicted: N windows)")
    if raw_conf:
        for (gt, pred), c in raw_conf.most_common(5):
            out.append(f"- `{gt}` → `{pred}`: {c} windows")
    else:
        out.append("- _none_")
    out.append("")
    out.append("**Smoothed** (true → predicted: N windows)")
    if smooth_conf:
        for (gt, pred), c in smooth_conf.most_common(5):
            out.append(f"- `{gt}` → `{pred}`: {c} windows")
    else:
        out.append("- _none_")
    return out


def write_md(rows: list[dict], path: Path, template: dict, audio_path: Path,
             smooth_enabled: bool = False,
             vad_enabled: bool = False,
             time_prior_enabled: bool = False,
             time_prior_weight: float = 0.0,
             song_offset: float = 0.0,
             gt_enabled: bool = False) -> None:
    n = len(rows)
    by_section = Counter(r["pred_id"] for r in rows)
    overall_conf = sum(r["conf"] for r in rows) / max(1, n)
    vocal_pct = (100.0 * sum(1 for r in rows if r["has_vocals"]) / n) if n else 0.0

    pair_counts: Counter = Counter()
    for r in rows:
        if len(r["top3"]) >= 2:
            a, b = r["top3"][0][0], r["top3"][1][0]
            pair_counts[tuple(sorted([a, b]))] += 1

    by_section_conf: dict[str, float] = {}
    for sid in by_section:
        confs = [r["conf"] for r in rows if r["pred_id"] == sid]
        by_section_conf[sid] = sum(confs) / len(confs)

    out: list[str] = [
        "# Position validation report",
        "",
        f"- Template: `{template.get('song_id', '?')}` "
        f"({len(template.get('references') or [])} reference"
        f"{'' if len(template.get('references') or []) == 1 else 's'}, "
        f"model `{template.get('embedding_model', '?')}`)",
        f"- Audio: `{audio_path.name}`",
        f"- Window: {WINDOW_SEC}s, hop: {HOP_SEC}s",
        f"- Total windows: {n}",
        f"- Mean confidence: {overall_conf:.3f}",
    ]
    if vad_enabled:
        out.append(f"- VAD: {vocal_pct:.1f}% of windows have vocals")
    if time_prior_enabled:
        out.append(f"- Time prior weight: {time_prior_weight:.1f}")
    if song_offset:
        out.append(
            f"- Song offset: {song_offset:.1f}s "
            f"(pre-song audio before time prior anchor)"
        )

    if gt_enabled:
        out += [""] + _accuracy_md_block(rows)

    out += [
        "",
        "## Predicted section distribution",
        "",
    ]
    if n == 0:
        out.append("_No windows scored._")
    else:
        for sid, c in by_section.most_common():
            pct = 100 * c / n
            out.append(
                f"- `{sid}`: {c} windows ({pct:.1f}%) "
                f"-- mean confidence {by_section_conf[sid]:.3f}"
            )

    out += ["", "## Most-confused section pairs", ""]
    top_pairs = pair_counts.most_common(5)
    if top_pairs:
        for (a, b), c in top_pairs:
            out.append(f"- `{a}` ↔ `{b}`: {c} windows")
    else:
        out.append("_None._")

    if smooth_enabled:
        raw_seq = [r["pred_id"] for r in rows]
        smooth_seq = [r["smoothed_id"] for r in rows]
        raw_changes = _count_changes(raw_seq)
        smooth_changes = _count_changes(smooth_seq)
        raw_run = _mean_run_length_sec(raw_seq, HOP_SEC)
        smooth_run = _mean_run_length_sec(smooth_seq, HOP_SEC)
        out += [
            "",
            "## Smoothing summary",
            "",
            f"- Raw: {raw_changes} section changes, "
            f"mean run length {raw_run:.1f}s",
            f"- Smoothed: {smooth_changes} section changes, "
            f"mean run length {smooth_run:.1f}s",
            f"- Predictions changed by smoothing: "
            f"{_changed_pct(raw_seq, smooth_seq):.1f}%",
        ]

    out += ["", "## Timeline (every 5s)", ""]
    last_t = -10.0
    for r in rows:
        if r["time"] - last_t >= 5.0:
            out.append(
                f"- `{_fmt_mmss(r['time'])}` → **{r['pred_id']}** "
                f"(conf={r['conf']:.2f}, margin={r['margin']:.2f})"
            )
            last_t = r["time"]

    if smooth_enabled:
        out += ["", "## Smoothed timeline (every 5s)", ""]
        last_t = -10.0
        for r in rows:
            if r["time"] - last_t >= 5.0:
                out.append(
                    f"- `{_fmt_mmss(r['time'])}` → **{r['smoothed_id']}**"
                )
                last_t = r["time"]

    if time_prior_enabled:
        out += ["", "## Time-prior-only baseline (every 5s)", ""]
        last_t = -10.0
        for r in rows:
            if r["time"] - last_t >= 5.0:
                tp = r["time_prior_id"]
                label = f"**{tp}**" if tp is not None else "_(pre-song)_"
                out.append(
                    f"- `{_fmt_mmss(r['time'])}` → {label}"
                )
                last_t = r["time"]

    path.write_text("\n".join(out) + "\n")


# ---------------------------------------------------------------------------
# Output: HTML
# ---------------------------------------------------------------------------

def _section_color(i: int, n: int) -> str:
    # Evenly spaced hues with HSL -- reproducible without any palette table.
    hue = (i * 360 / max(n, 1)) % 360
    return f"hsl({hue:.0f}, 70%, 50%)"


VAD_VOCAL_COLOR = "#2d2d2d"
VAD_QUIET_COLOR = "#dddddd"


def write_html(rows: list[dict], path: Path, template: dict,
               audio_path: Path, sections: list[tuple[str, str, np.ndarray]],
               smooth_enabled: bool = False,
               vad_enabled: bool = False,
               time_prior_enabled: bool = False,
               gt_enabled: bool = False) -> None:
    section_ids = [sid for sid, _stype, _frames in sections]
    color_map = {sid: _section_color(i, len(section_ids))
                 for i, sid in enumerate(section_ids)}

    if not rows:
        path.write_text(
            "<!doctype html><html><body><p>No windows scored.</p></body></html>"
        )
        return

    px_per_sec = 8
    total_time = rows[-1]["time"] + WINDOW_SEC
    chart_width = int(total_time * px_per_sec) + 20
    block_w = HOP_SEC * px_per_sec

    # Stack one strip per enabled output. Single- and two-strip layouts match
    # the pre-VAD output byte-for-byte so --vad-penalty 0.0 (and additionally
    # --transition-penalty 0.0) preserves earlier behaviour.
    strips: list[str] = []
    if gt_enabled:
        strips.append("ground_truth")
    strips.append("raw")
    if time_prior_enabled:
        strips.append("time_prior")
    if smooth_enabled:
        strips.append("smoothed")
    if vad_enabled:
        strips.append("vad")

    if len(strips) == 1:
        # Legacy single-row layout, byte-preserved.
        strip_h = 48
        strip_y = [6]
        label_y: list[Optional[int]] = [None]
        axis_top = 60
        tick_text_offset = 18
        svg_height = 90
    elif strips == ["raw", "smoothed"]:
        # Viterbi-only two-row layout from the previous PR, byte-preserved.
        strip_h = 32
        strip_y = [20, 80]
        label_y = [14, 72]
        axis_top = 116
        tick_text_offset = 14
        svg_height = 140
    else:
        # General multi-strip layout (raw+vad, raw+smoothed+vad).
        strip_h = 32
        row_gap = 28
        strip_y = [20]
        label_y = [14]
        for _ in strips[1:]:
            ny = strip_y[-1] + strip_h + row_gap
            strip_y.append(ny)
            label_y.append(ny - 8)
        axis_top = strip_y[-1] + strip_h + 4
        tick_text_offset = 14
        svg_height = axis_top + 24

    template_name = template.get("song_id") or "?"
    parts: list[str] = [
        '<!doctype html><html><head><meta charset="utf-8">',
        '<title>Predicted position over time</title>',
        '<style>',
        'body{font-family:system-ui,sans-serif;margin:24px;color:#222;}',
        'h1{font-size:20px;}',
        '.meta{color:#555;margin:4px 0 16px;font-size:13px;}',
        '.chart{overflow-x:auto;border:1px solid #ddd;padding:8px;background:#fafafa;}',
        '.legend{margin-top:16px;line-height:2;}',
        '.legend span{display:inline-block;margin:2px 6px 2px 0;padding:2px 8px;'
        'color:#fff;border-radius:3px;font-size:12px;}',
        '</style></head><body>',
        f'<h1>Predicted position over time '
        f'(band recording vs {html_lib.escape(audio_path.name)})</h1>',
        f'<p class="meta">Template <code>{html_lib.escape(template_name)}</code> '
        f'&middot; Model <code>{html_lib.escape(template.get("embedding_model") or "?")}</code> '
        f'&middot; {len(rows)} windows '
        f'&middot; {WINDOW_SEC}s window, {HOP_SEC}s hop</p>',
        '<div class="chart">',
        f'<svg width="{chart_width}" height="{svg_height}" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]

    # Time axis ticks every 30 seconds -- span both rows when smoothing is on.
    for t in range(0, int(total_time) + 1, 30):
        x = t * px_per_sec + 10
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{axis_top}" stroke="#ccc"/>'
        )
        parts.append(
            f'<text x="{x}" y="{axis_top + tick_text_offset}" font-size="10" fill="#555">'
            f'{_fmt_mmss(t)}</text>'
        )

    strip_labels = {
        "raw": "Raw", "smoothed": "Smoothed",
        "vad": "VAD", "time_prior": "Time prior",
        "ground_truth": "Ground truth",
    }
    for kind, sy, ly in zip(strips, strip_y, label_y):
        if ly is not None:
            parts.append(
                f'<text x="10" y="{ly}" font-size="11" fill="#444" '
                f'font-weight="600">{strip_labels[kind]}</text>'
            )
        for r in rows:
            x = r["time"] * px_per_sec + 10
            if kind == "raw":
                color = color_map.get(r["pred_id"], "#999")
                opacity = f'{max(0.15, min(1.0, r["conf"])):.2f}'
            elif kind == "smoothed":
                color = color_map.get(r["smoothed_id"], "#999")
                opacity = "1.00"
            elif kind == "time_prior":
                tp = r["time_prior_id"]
                if tp is None:
                    color = "#eeeeee"  # pre-song window (before song_offset)
                else:
                    color = color_map.get(tp, "#999")
                opacity = "1.00"
            elif kind == "ground_truth":
                gt = r.get("gt_section_id")
                if gt is None or gt.startswith("_"):
                    color = "#eeeeee"  # gap or excluded section
                else:
                    color = color_map.get(gt, "#999")
                opacity = "1.00"
            else:  # vad
                color = VAD_VOCAL_COLOR if r["has_vocals"] else VAD_QUIET_COLOR
                opacity = "1.00"
            parts.append(
                f'<rect x="{x:.1f}" y="{sy}" width="{block_w:.1f}" '
                f'height="{strip_h}" fill="{color}" opacity="{opacity}"/>'
            )

    parts.append('</svg></div>')

    parts.append('<div class="legend"><strong>Sections:</strong> ')
    for sid in section_ids:
        parts.append(
            f'<span style="background:{color_map[sid]}">'
            f'{html_lib.escape(sid)}</span>'
        )
    parts.append('</div></body></html>')

    path.write_text("\n".join(parts))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--template", type=Path, required=True,
                   help="Path to a built *_aligned.json template")
    p.add_argument("--audio", type=Path, required=True,
                   help="Audio file to validate against (mp3/wav/m4a/...)")
    p.add_argument("--out", required=True,
                   help="Output path prefix; writes <out>.csv, .md, .html")
    p.add_argument("--transition-penalty", type=float, default=3.0,
                   help="Viterbi log-penalty for jumps that are not "
                        "stay-in-section or forward-to-next-section. 0.0 "
                        "disables smoothing entirely (default 3.0).")
    p.add_argument("--vad-penalty", type=float, default=2.0,
                   help="Log-emission penalty when a window's vocal/no-vocal "
                        "state (from webrtcvad) disagrees with a section's "
                        "vocal-vs-instrumental classification. 0.0 disables "
                        "VAD entirely (default 2.0).")
    p.add_argument("--time-prior-weight", type=float, default=2.0,
                   help="Bonus added to the log-emission of the section the "
                        "template's reference timeline says should be playing "
                        "at this elapsed time. 0.0 disables the time prior "
                        "(default 2.0).")
    p.add_argument("--song-offset", type=float, default=0.0,
                   help="Seconds of pre-song content in the audio. Subtracted "
                        "from each window's centre timestamp before the time "
                        "prior looks up the expected section. MERT scoring, "
                        "ground-truth lookup, and report timestamps are "
                        "unaffected. Windows whose effective time is "
                        "negative (before song start) get no time-prior bonus "
                        "(default 0.0).")
    p.add_argument("--ground-truth", type=Path, default=None,
                   help="Optional path to a tooling/ground_truth/<song>.json file. "
                        "When set, the report includes accuracy numbers against "
                        "these labels and the HTML chart adds a ground-truth row "
                        "at the top.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    template = json.loads(args.template.read_text())
    sections = _collect_section_frames(template)
    if not sections:
        log.error("Template has no sections with audio_embedding_sequence -- "
                  "nothing to score against. Rebuild the template first.")
        return 1
    log.info("Template: %d scorable sections (%d sections total).",
             len(sections), len(template.get("structure", [])))

    log.info("Loading audio %s ...", args.audio)
    y = _load_audio(args.audio)
    duration = len(y) / EMBED_SR
    log.info("Audio: %.1fs at %d Hz", duration, EMBED_SR)

    if duration < WINDOW_SEC:
        log.error("Audio is shorter than the %.1fs window.", WINDOW_SEC)
        return 1

    extractor = bs.EmbeddingExtractor(template.get("embedding_model"))
    extractor.load()
    log.info("Using embedding model: %s", extractor.model_name)

    expected = max(0, int((duration - WINDOW_SEC) / HOP_SEC) + 1)
    log.info("Scoring %d windows ...", expected)

    # Collect per-window similarities into a (W, S) matrix so we can run
    # Viterbi over them after the loop. Discarding the matrix and keeping
    # only the argmax was the bug that prevented smoothing.
    times: list[float] = []
    sims_rows: list[np.ndarray] = []
    t = 0.0
    scored = 0
    while t + WINDOW_SEC <= duration:
        s0 = int(t * EMBED_SR)
        s1 = int((t + WINDOW_SEC) * EMBED_SR)
        chunk = y[s0:s1]
        emb = extractor.embed(chunk)
        if emb is None:
            t += HOP_SEC
            continue
        q = _l2_norm(emb)
        sims_rows.append(_score_window(q, sections))
        times.append(t)
        scored += 1
        if args.verbose and scored % 100 == 0:
            log.debug("  scored %d windows ...", scored)
        t += HOP_SEC

    log.info("Scored %d windows.", scored)

    smooth_enabled = args.transition_penalty > 0.0
    vad_enabled = args.vad_penalty > 0.0
    time_prior_enabled = args.time_prior_weight > 0.0
    gt_enabled = args.ground_truth is not None

    gt_intervals: list[tuple[str, float, float]] = []
    if gt_enabled:
        gt_intervals = load_ground_truth(args.ground_truth)
        log.info("Loaded %d ground-truth intervals from %s",
                 len(gt_intervals), args.ground_truth)

    has_vocals = np.zeros(scored, dtype=bool)
    if vad_enabled and scored > 0:
        log.info("Running VAD ...")
        has_vocals = _compute_vad(y, EMBED_SR, times, WINDOW_SEC)
        log.info("VAD: %d/%d windows have vocals (%.1f%%)",
                 int(has_vocals.sum()), len(has_vocals),
                 100.0 * has_vocals.mean() if len(has_vocals) else 0.0)

    # The time prior is computed from the template alone -- no audio
    # required -- so its expected-section sequence doubles as the
    # zero-audio baseline the report compares against.
    starts, ends = _section_time_ranges(template, sections)

    if scored == 0:
        sims_matrix = np.zeros((0, len(sections)), dtype=np.float32)
        raw_idx = np.zeros(0, dtype=np.int32)
        smoothed_idx = np.zeros(0, dtype=np.int32)
        expected_idx = np.zeros(0, dtype=np.int32)
    else:
        sims_matrix = np.stack(sims_rows, axis=0)
        raw_idx = sims_matrix.argmax(axis=1).astype(np.int32)
        # Always compute expected_idx so the time-prior-only baseline can be
        # reported even when --time-prior-weight 0.0 (the prior just doesn't
        # influence Viterbi in that case). Pre-song windows (effective time
        # before song start under --song-offset) get -1 and contribute nothing.
        expected_idx = _expected_indices(
            times, starts, ends, WINDOW_SEC, song_offset=args.song_offset,
        )
        if smooth_enabled:
            log_em = _log_emissions(sims_matrix)
            if time_prior_enabled:
                log_em = _apply_time_prior(
                    log_em, expected_idx, args.time_prior_weight,
                )
            if vad_enabled:
                is_vocal_section = _vocal_section_mask(template, sections)
                log_em = _apply_vad_penalty(
                    log_em, has_vocals, is_vocal_section, args.vad_penalty,
                )
            log_tr = _transition_log_probs(len(sections), args.transition_penalty)
            smoothed_idx = _viterbi(log_em, log_tr)
            extras = []
            if vad_enabled:
                extras.append(f"vad_penalty={args.vad_penalty:.2f}")
            if time_prior_enabled:
                extras.append(f"time_prior_weight={args.time_prior_weight:.2f}")
            log.info("Viterbi smoothing on (transition_penalty=%.2f%s).",
                     args.transition_penalty,
                     f", {', '.join(extras)}" if extras else "")
        else:
            smoothed_idx = raw_idx.copy()

    rows: list[dict] = []
    for w in range(scored):
        sims = sims_matrix[w]
        order = np.argsort(-sims)
        top3 = [(sections[int(i)][0], sections[int(i)][1], float(sims[int(i)]))
                for i in order[:3]]
        raw_i = int(raw_idx[w])
        sm_i = int(smoothed_idx[w])
        confidence = float(sims[raw_i])
        second = float(sims[int(order[1])]) if len(order) > 1 else 0.0
        tp_i = int(expected_idx[w])
        if tp_i >= 0:
            tp_id = sections[tp_i][0]
            tp_type = sections[tp_i][1]
        else:
            # Pre-song window (effective time before song_offset).
            tp_id = None
            tp_type = None
        gt_sid = None
        if gt_enabled:
            gt_sid = gt_section_at(times[w] + WINDOW_SEC / 2, gt_intervals)
        rows.append({
            "time": times[w],
            "pred_id": sections[raw_i][0],
            "pred_type": sections[raw_i][1],
            "conf": confidence,
            "margin": confidence - second,
            "top3": top3,
            "smoothed_id": sections[sm_i][0],
            "smoothed_type": sections[sm_i][1],
            "has_vocals": bool(has_vocals[w]) if w < len(has_vocals) else False,
            "time_prior_id": tp_id,
            "time_prior_type": tp_type,
            "gt_section_id": gt_sid,
        })

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    html_path = out_prefix.with_suffix(".html")

    write_csv(rows, csv_path)
    write_md(rows, md_path, template, args.audio,
             smooth_enabled=smooth_enabled,
             vad_enabled=vad_enabled,
             time_prior_enabled=time_prior_enabled,
             time_prior_weight=args.time_prior_weight,
             song_offset=args.song_offset,
             gt_enabled=gt_enabled)
    write_html(rows, html_path, template, args.audio, sections,
               smooth_enabled=smooth_enabled,
               vad_enabled=vad_enabled,
               time_prior_enabled=time_prior_enabled,
               gt_enabled=gt_enabled)

    print(f"\nWrote {csv_path}, {md_path.name}, {html_path.name} ({len(rows)} windows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
