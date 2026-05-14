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
prediction, whisper fusion, HMM smoothing.

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
                  sections: list[tuple[str, str, np.ndarray]]
                  ) -> list[tuple[str, str, float]]:
    """Rank every section by its max-cosine-sim against the query window."""
    ranked: list[tuple[str, str, float]] = []
    for sid, stype, frames in sections:
        sims = frames @ query_vec  # both sides L2-normed -> dot == cos
        ranked.append((sid, stype, float(sims.max())))
    ranked.sort(key=lambda x: -x[2])
    return ranked


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


def write_md(rows: list[dict], path: Path, template: dict, audio_path: Path) -> None:
    n = len(rows)
    by_section = Counter(r["pred_id"] for r in rows)
    overall_conf = sum(r["conf"] for r in rows) / max(1, n)

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

    out += ["", "## Timeline (every 5s)", ""]
    last_t = -10.0
    for r in rows:
        if r["time"] - last_t >= 5.0:
            out.append(
                f"- `{_fmt_mmss(r['time'])}` → **{r['pred_id']}** "
                f"(conf={r['conf']:.2f}, margin={r['margin']:.2f})"
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


def write_html(rows: list[dict], path: Path, template: dict,
               audio_path: Path, sections: list[tuple[str, str, np.ndarray]]) -> None:
    section_ids = [sid for sid, _stype, _frames in sections]
    color_map = {sid: _section_color(i, len(section_ids))
                 for i, sid in enumerate(section_ids)}

    if not rows:
        path.write_text(
            "<!doctype html><html><body><p>No windows scored.</p></body></html>"
        )
        return

    px_per_sec = 8
    chart_height = 60
    total_time = rows[-1]["time"] + WINDOW_SEC
    chart_width = int(total_time * px_per_sec) + 20

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
        f'<svg width="{chart_width}" height="{chart_height + 30}" '
        'xmlns="http://www.w3.org/2000/svg">',
    ]

    # Time axis ticks every 30 seconds.
    for t in range(0, int(total_time) + 1, 30):
        x = t * px_per_sec + 10
        parts.append(
            f'<line x1="{x}" y1="0" x2="{x}" y2="{chart_height}" stroke="#ccc"/>'
        )
        parts.append(
            f'<text x="{x}" y="{chart_height+18}" font-size="10" fill="#555">'
            f'{_fmt_mmss(t)}</text>'
        )

    # Window blocks: width = one hop so consecutive predictions tile cleanly.
    block_w = HOP_SEC * px_per_sec
    for r in rows:
        x = r["time"] * px_per_sec + 10
        color = color_map.get(r["pred_id"], "#999")
        opacity = max(0.15, min(1.0, r["conf"]))
        parts.append(
            f'<rect x="{x:.1f}" y="6" width="{block_w:.1f}" '
            f'height="{chart_height-12}" fill="{color}" '
            f'opacity="{opacity:.2f}"/>'
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

    rows: list[dict] = []
    t = 0.0
    while t + WINDOW_SEC <= duration:
        s0 = int(t * EMBED_SR)
        s1 = int((t + WINDOW_SEC) * EMBED_SR)
        chunk = y[s0:s1]
        emb = extractor.embed(chunk)
        if emb is None:
            t += HOP_SEC
            continue
        q = _l2_norm(emb)
        ranked = _score_window(q, sections)
        top_score = ranked[0][2]
        second_score = ranked[1][2] if len(ranked) > 1 else 0.0
        rows.append({
            "time": t,
            "pred_id": ranked[0][0],
            "pred_type": ranked[0][1],
            "conf": top_score,
            "margin": top_score - second_score,
            "top3": ranked[:3],
        })
        if args.verbose and len(rows) % 100 == 0:
            log.debug("  scored %d windows ...", len(rows))
        t += HOP_SEC

    log.info("Scored %d windows.", len(rows))

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    html_path = out_prefix.with_suffix(".html")

    write_csv(rows, csv_path)
    write_md(rows, md_path, template, args.audio)
    write_html(rows, html_path, template, args.audio, sections)

    print(f"\nWrote {csv_path}, {md_path.name}, {html_path.name} ({len(rows)} windows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
