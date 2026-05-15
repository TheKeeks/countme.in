"""
build_song.py
-------------
Phase-1-and-2 orchestrator: builds a single song template by blending
two-to-five reference recordings into one richer JSON template.

What it does, end-to-end:

  1. Parse the structured .lyrics file (via template_builder.parse_lyrics).
  2. Extract global audio_features (tempo, key, beats) from the first
     reference that loads.
  3. For each reference recording:
        a. Run faster-whisper + fuzzy-align (alignment.transcribe,
           alignment.align, ...) to get per-line start/end timestamps.
        b. Extract a MERT (or wav2vec2 fallback) embedding for every
           lyric line.
        c. Extract embeddings every 2 seconds within each section.
  4. Blend per-reference embeddings:
        - audio_embedding_per_reference[<filename>] on each line
        - audio_embedding_blended = element-wise mean, L2-normalized
        - section.audio_embedding_sequence = [list of per-slice blended
          embeddings across the section]
  5. Write the final JSON template.

Robustness: if one reference fails to align or embed, log a warning and
continue with the remaining ones. The pipeline succeeds as long as at
least one reference contributes useful data.

CLI:

    python tooling/build_song.py --song peggy-o \\
        --references "tooling/references/peggy-o/*.mp3" \\
        --lyrics tooling/songs/peggy_o.lyrics \\
        --out web/templates/peggy_o_aligned.json
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
import sys
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

# template_builder lives next to this file. alignment is imported lazily
# because it pulls in faster_whisper at module scope -- callers that only
# touch the blending helpers (tests, etc.) shouldn't need that installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import template_builder as tb  # noqa: E402


def _alignment_module():
    import alignment as al  # noqa: PLC0415
    return al


import inspect_template  # noqa: E402 -- shared report builder

log = logging.getLogger("build_song")

PRIMARY_MODEL = "m-a-p/MERT-v1-95M"
FALLBACK_MODEL = "facebook/wav2vec2-base"
EMBED_SAMPLE_RATE = 24000  # MERT-v1-95M expects 24kHz; wav2vec2-base wants 16kHz (we resample below)
SECTION_SLICE_SEC = 2.0
EMBED_DECIMALS = 4  # round floats in the JSON to keep file size sane

# Reference is discarded if fewer than this fraction of lyric lines aligned
# via whisper-matched words (interpolated lines don't count).
MIN_MATCH_FRAC = 0.25

GROUND_TRUTH_DIR = Path(__file__).resolve().parent / "ground_truth"


def load_ground_truth(song_id: str) -> Optional[dict[str, tuple[float, float]]]:
    """Return {section_id: (start_sec, end_sec)} from tooling/ground_truth/<song_id>.json,
    or None if no file exists.

    Section IDs beginning with `_` (e.g. `_silence`) are kept here so the
    validator can later mask them out of accuracy reporting; build_song
    itself just ignores any GT section that isn't in the template.
    """
    path = GROUND_TRUTH_DIR / f"{song_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    out: dict[str, tuple[float, float]] = {}
    for s in data.get("sections", []):
        sid = s.get("section_id")
        if sid is None or "start" not in s or "end" not in s:
            continue
        out[sid] = (float(s["start"]), float(s["end"]))
    log.info("Using ground truth from %s (%d sections).", path, len(out))
    return out


# ---------------------------------------------------------------------------
# Embedding model loader
# ---------------------------------------------------------------------------

class EmbeddingExtractor:
    """Lazy-loaded wrapper around a HuggingFace music encoder.

    Tries m-a-p/MERT-v1-95M (music-pretrained, 95M params) first; falls
    back to facebook/wav2vec2-base if MERT can't be loaded.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name
        self._loaded = False
        self._processor = None
        self._model = None
        self._sample_rate = EMBED_SAMPLE_RATE

    def load(self) -> None:
        if self._loaded:
            return

        import torch
        from transformers import AutoModel, AutoFeatureExtractor

        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))

        candidates = []
        if self.model_name:
            candidates.append(self.model_name)
        else:
            candidates.extend([PRIMARY_MODEL, FALLBACK_MODEL])

        last_err: Optional[Exception] = None
        for name in candidates:
            try:
                log.info("Loading embedding model %s ...", name)
                processor = AutoFeatureExtractor.from_pretrained(
                    name, trust_remote_code=True
                )
                model = AutoModel.from_pretrained(name, trust_remote_code=True)
                model.eval()
                self._processor = processor
                self._model = model
                self.model_name = name
                self._sample_rate = getattr(processor, "sampling_rate", EMBED_SAMPLE_RATE)
                self._loaded = True
                log.info("  using %s (sample rate %d Hz)", name, self._sample_rate)
                return
            except Exception as exc:  # noqa: BLE001 -- any model load error
                last_err = exc
                log.warning("  failed to load %s: %s", name, exc)
        raise RuntimeError(f"Could not load any embedding model: {last_err}")

    @property
    def sample_rate(self) -> int:
        return self._sample_rate if self._loaded else EMBED_SAMPLE_RATE

    def embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Mean-pooled hidden state for one audio slice. Returns None if too short."""
        if audio is None or len(audio) < int(self.sample_rate * 0.2):
            return None  # need at least 200ms to be meaningful
        import torch
        self.load()
        inputs = self._processor(
            audio, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        hidden = outputs.last_hidden_state  # [1, T, D]
        pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy()
        return pooled


# ---------------------------------------------------------------------------
# Audio + alignment per reference
# ---------------------------------------------------------------------------

def load_audio_for_embedding(path: Path, target_sr: int) -> np.ndarray:
    """Load audio mono at the target sample rate."""
    import librosa
    y, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return y


def slice_audio(y: np.ndarray, sr: int, start_sec: float, end_sec: float) -> np.ndarray:
    s0 = int(max(0.0, start_sec) * sr)
    s1 = int(min(len(y) / sr, end_sec) * sr)
    if s1 <= s0:
        return np.zeros(0, dtype=np.float32)
    return y[s0:s1]


def align_reference(audio_path: Path, template: dict, model_size: str,
                    transcript_path: Optional[Path] = None,
                    vad_filter: bool = False) -> dict:
    """Run whisper + fuzzy alignment for one reference. Returns the spans dict."""
    al = _alignment_module()
    expected = al.flatten_lyrics(template)
    if transcript_path is not None:
        recognized = json.loads(transcript_path.read_text())
        recognized = [
            {"text": al.normalize(r["text"]), "start": float(r["start"]),
             "end": float(r["end"])}
            for r in recognized if al.normalize(r["text"])
        ]
        log.info("  loaded %d transcript words from %s", len(recognized), transcript_path.name)
    else:
        recognized = al.transcribe(audio_path, model_size, vad_filter=vad_filter)
    matches = al.align(recognized, expected)
    matched_n = sum(1 for m in matches if m is not None)
    log.info("  matched %d / %d expected words (%.0f%%)",
             matched_n, len(expected), 100 * matched_n / max(1, len(expected)))
    spans = al.collect_line_spans(expected, matches)
    spans = al.interpolate_missing(template, spans)
    return spans


# ---------------------------------------------------------------------------
# Per-reference processing
# ---------------------------------------------------------------------------

def process_reference(audio_path: Path, template: dict, extractor: EmbeddingExtractor,
                      model_size: str, vad_filter: bool = False,
                      ground_truth: Optional[dict[str, tuple[float, float]]] = None) -> dict:
    """Returns {
        "filename": "...",
        "spans": {(section_id, line_idx): {start_sec, end_sec, ...}, ...},
        "line_embeddings": {(section_id, line_idx): np.ndarray | None},
        "section_sequences": {section_id: [np.ndarray, ...]},
        "section_bounds": {section_id: (start_sec, end_sec)},
        "bounds_source": {section_id: "ground_truth" | "whisper"},
    }

    When `ground_truth` is provided, its bounds override the Whisper-derived
    bounds for any section it covers. Line spans are clipped to their parent
    section's GT bounds; lines whose Whisper alignment falls entirely outside
    the GT range are dropped with a warning.
    """
    log.info("[%s] aligning ...", audio_path.name)
    spans = align_reference(audio_path, template, model_size, vad_filter=vad_filter)

    log.info("[%s] extracting embeddings ...", audio_path.name)
    extractor.load()
    y_embed = load_audio_for_embedding(audio_path, extractor.sample_rate)
    audio_duration = len(y_embed) / extractor.sample_rate if extractor.sample_rate else 0.0
    gt = ground_truth or {}

    # Line-level embeddings, clipped to GT section bounds when available.
    line_embeddings: dict = {}
    for section in template["structure"]:
        sid = section["section_id"]
        gt_bounds = gt.get(sid)
        for line in section["lines"]:
            key = (sid, line["line_index"])
            if key not in spans:
                line_embeddings[key] = None
                continue
            sp = spans[key]
            s0, s1 = sp["start_sec"], sp["end_sec"]
            if gt_bounds is not None:
                gs, ge = gt_bounds
                clipped_s = max(s0, gs)
                clipped_e = min(s1, ge)
                if clipped_e <= clipped_s:
                    log.warning(
                        "[%s] line %s/%d Whisper span [%.2f, %.2f] is outside "
                        "GT bounds [%.2f, %.2f]; dropping.",
                        audio_path.name, sid, line["line_index"],
                        s0, s1, gs, ge,
                    )
                    line_embeddings[key] = None
                    continue
                s0, s1 = clipped_s, clipped_e
                sp["start_sec"], sp["end_sec"] = s0, s1
            audio_slice = slice_audio(y_embed, extractor.sample_rate, s0, s1)
            line_embeddings[key] = extractor.embed(audio_slice)

    # Section-level embedding sequences. When ground truth is present every
    # section in GT uses its bounds directly; everything else follows the
    # three-pass fallback (line-derived, then neighbour-derived) we used
    # before.
    sec_ids = [s["section_id"] for s in template["structure"]]
    bounds: dict[str, tuple[float, float]] = {}
    bounds_source: dict[str, str] = {}

    if gt:
        for sid in sec_ids:
            if sid in gt:
                bounds[sid] = gt[sid]
                bounds_source[sid] = "ground_truth"

    # Pass 1: bounds from the line spans we got from alignment (already
    # clipped above when GT is present). Only fills sections still missing.
    for section in template["structure"]:
        sid = section["section_id"]
        if sid in bounds:
            continue
        starts = [spans[(sid, l["line_index"])]["start_sec"]
                  for l in section["lines"]
                  if (sid, l["line_index"]) in spans]
        ends = [spans[(sid, l["line_index"])]["end_sec"]
                for l in section["lines"]
                if (sid, l["line_index"]) in spans]
        if starts and ends:
            bounds[sid] = (max(0.0, min(starts) - 0.5), max(ends) + 0.5)
            bounds_source[sid] = "whisper"

    # Pass 2: fill remaining sections from neighbours in song order.
    for i, sid in enumerate(sec_ids):
        if sid in bounds:
            continue
        prev_end: Optional[float] = None
        for j in range(i - 1, -1, -1):
            if sec_ids[j] in bounds:
                prev_end = bounds[sec_ids[j]][1]
                break
        next_start: Optional[float] = None
        for j in range(i + 1, len(sec_ids)):
            if sec_ids[j] in bounds:
                next_start = bounds[sec_ids[j]][0]
                break
        sec_start = prev_end if prev_end is not None else 0.0
        sec_end = next_start if next_start is not None else audio_duration
        bounds[sid] = (sec_start, sec_end)
        bounds_source[sid] = "whisper"

    # Pass 3: embed every section at SECTION_SLICE_SEC intervals.
    section_sequences: dict = {}
    for section in template["structure"]:
        sid = section["section_id"]
        sec_start, sec_end = bounds[sid]
        seq = []
        t = sec_start
        while t < sec_end:
            slice_end = min(t + SECTION_SLICE_SEC, sec_end)
            chunk = slice_audio(y_embed, extractor.sample_rate, t, slice_end)
            emb = extractor.embed(chunk)
            if emb is not None:
                seq.append(emb)
            t += SECTION_SLICE_SEC
        section_sequences[sid] = seq

    return {
        "filename": audio_path.name,
        "spans": spans,
        "line_embeddings": line_embeddings,
        "section_sequences": section_sequences,
        # Per-section (start_sec, end_sec) covering both lyric-derived bounds
        # (vocal sections) and neighbour-derived bounds (intro / jam / outro).
        # Used by _populate_sections to stamp explicit start_time / end_time
        # fields on every section -- including the instrumental ones the
        # time-prior validator needs.
        "section_bounds": dict(bounds),
        # Records which source produced each section's bounds so inspect_template
        # (and downstream debugging) can verify ground truth was actually used.
        "bounds_source": dict(bounds_source),
    }


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------

def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-9:
        return v
    return v / n


def blend_vectors(vectors: list[np.ndarray]) -> Optional[np.ndarray]:
    valid = [v for v in vectors if v is not None]
    if not valid:
        return None
    stacked = np.stack(valid, axis=0)
    mean = stacked.mean(axis=0)
    return _l2_norm(mean)


def blend_sequences(sequences: list[list[np.ndarray]]) -> list[np.ndarray]:
    """Element-wise mean across sequences, truncated to the shortest."""
    valid = [seq for seq in sequences if seq]
    if not valid:
        return []
    min_len = min(len(s) for s in valid)
    blended = []
    for i in range(min_len):
        vectors = [s[i] for s in valid]
        avg = np.stack(vectors, axis=0).mean(axis=0)
        blended.append(_l2_norm(avg))
    return blended


def _round_vec(v: Optional[np.ndarray]) -> Optional[list]:
    if v is None:
        return None
    return [round(float(x), EMBED_DECIMALS) for x in v]


def _round_seq(seq: list[np.ndarray]) -> list[list]:
    return [_round_vec(v) for v in seq if v is not None]


# ---------------------------------------------------------------------------
# Template assembly
# ---------------------------------------------------------------------------

def _expand_references(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches and Path(pat).exists():
            matches = [pat]
        for m in matches:
            if m in seen:
                continue
            seen.add(m)
            paths.append(Path(m))
    return paths


def assemble_template(lyrics_path: Path, song_id: str, title: str,
                      version_notes: str,
                      references: list[Path], extractor: EmbeddingExtractor,
                      model_size: str, vad_filter: bool = False) -> dict:
    sections = tb.parse_lyrics(lyrics_path)
    template = asdict(tb.SongTemplate(
        song_id=song_id,
        title=title,
        version_notes=version_notes,
        audio_features=None,
        structure=sections,
    ))

    ground_truth = load_ground_truth(song_id)
    if ground_truth is None:
        log.info("Using Whisper-aligned bounds (no ground truth found for %s).",
                 song_id)
    template["ground_truth_used"] = ground_truth is not None

    n_lines_total = sum(len(s["lines"]) for s in template["structure"])

    # Global audio features from the first reference that loads successfully.
    audio_features = None
    for ref in references:
        try:
            audio_features = asdict(tb.extract_audio_features(ref))
            template["reference_for_audio_features"] = ref.name
            break
        except Exception as exc:  # noqa: BLE001
            log.warning("audio feature extraction failed for %s: %s", ref.name, exc)
    template["audio_features"] = audio_features

    # Process each reference; gate on alignment quality.
    per_ref_results: list[dict] = []
    for ref in references:
        try:
            result = process_reference(ref, template, extractor, model_size,
                                       vad_filter=vad_filter,
                                       ground_truth=ground_truth)
        except Exception as exc:  # noqa: BLE001
            log.warning("reference %s failed (%s); continuing with the rest",
                        ref.name, exc)
            continue

        matched_lines = sum(
            1 for sp in result["spans"].values()
            if not sp.get("interpolated", True) and sp.get("matched_word_count", 0) > 0
        )
        if n_lines_total > 0:
            frac = matched_lines / n_lines_total
            if frac < MIN_MATCH_FRAC:
                log.warning(
                    "[%s] alignment quality too low: %d/%d lines matched (%.0f%%). "
                    "Discarding this reference.",
                    ref.name, matched_lines, n_lines_total, 100 * frac,
                )
                continue
        per_ref_results.append(result)

    if not per_ref_results:
        raise RuntimeError(
            "All references failed alignment quality check. Try: a larger whisper "
            "model (--whisper-model small or medium), --vad-filter off, a different "
            "reference, or supply --transcript with a pre-computed transcript."
        )

    template["references"] = [r["filename"] for r in per_ref_results]
    template["embedding_model"] = extractor.model_name

    # Stamp the line-level embeddings and timing back onto the template.
    _populate_lines(template, per_ref_results)
    _populate_sections(template, per_ref_results)

    # Diagnostic report -- same shape as inspect_template prints on a built file.
    for line in inspect_template.report(template, n_attempted_refs=len(references)).splitlines():
        log.info(line)

    return template


def _populate_lines(template: dict, per_ref: list[dict]) -> None:
    for section in template["structure"]:
        for line in section["lines"]:
            key = (section["section_id"], line["line_index"])
            per_ref_embeddings: dict = {}
            per_ref_spans: dict = {}
            for r in per_ref:
                emb = r["line_embeddings"].get(key)
                if emb is not None:
                    per_ref_embeddings[r["filename"]] = _round_vec(emb)
                sp = r["spans"].get(key)
                if sp is not None:
                    per_ref_spans[r["filename"]] = {
                        "start_sec": round(float(sp["start_sec"]), 3),
                        "end_sec": round(float(sp["end_sec"]), 3),
                        "matched_word_count": int(sp["matched_word_count"]),
                        "interpolated": bool(sp["interpolated"]),
                    }

            line["audio_embedding_per_reference"] = per_ref_embeddings
            blended = blend_vectors([r["line_embeddings"].get(key) for r in per_ref])
            line["audio_embedding_blended"] = _round_vec(blended)
            line["timing_per_reference"] = per_ref_spans

            # Blended (median) start/end across references, for the runtime to
            # have a single canonical line span.
            starts = [sp["start_sec"] for sp in per_ref_spans.values()]
            ends = [sp["end_sec"] for sp in per_ref_spans.values()]
            line["start_sec"] = round(float(np.median(starts)), 3) if starts else None
            line["end_sec"] = round(float(np.median(ends)), 3) if ends else None
            # The legacy chroma_signature field is no longer populated.
            line.pop("chroma_signature", None)


def _populate_sections(template: dict, per_ref: list[dict]) -> None:
    for section in template["structure"]:
        sid = section["section_id"]
        sequences = [r["section_sequences"].get(sid, []) for r in per_ref]
        blended_seq = blend_sequences(sequences)
        section["audio_embedding_sequence"] = _round_seq(blended_seq)

        # Roll section bounds from blended line bounds (legacy field).
        starts = [l["start_sec"] for l in section["lines"]
                  if l.get("start_sec") is not None]
        ends = [l["end_sec"] for l in section["lines"]
                if l.get("end_sec") is not None]
        if starts:
            section["start_sec"] = round(max(0.0, min(starts) - 0.5), 3)
        if ends:
            section["end_sec"] = round(max(ends) + 0.5, 3)

        # If any reference reported this section's bounds came from ground
        # truth, all of them did (it's a per-song decision) -- promote the
        # GT bounds directly and tag the source. Otherwise mark as whisper.
        sources_seen = {r.get("bounds_source", {}).get(sid) for r in per_ref}
        sources_seen.discard(None)
        if "ground_truth" in sources_seen:
            gt_bounds = [r["section_bounds"][sid] for r in per_ref
                         if r.get("bounds_source", {}).get(sid) == "ground_truth"
                         and sid in r.get("section_bounds", {})]
            if gt_bounds:
                section["start_time"] = round(float(gt_bounds[0][0]), 3)
                section["end_time"] = round(float(gt_bounds[0][1]), 3)
                section["bounds_source"] = "ground_truth"
                continue
        section["bounds_source"] = "whisper"

        # Explicit start_time / end_time on every section -- this is what the
        # time-prior validator reads. Vocal sections use first-line-start /
        # last-line-end. Instrumental sections (no lyric lines) fall back to
        # the per-reference (start_sec, end_sec) covering the audio chunks
        # we extracted embeddings from, blended across references via median.
        if section["lines"]:
            line_starts = [l["start_sec"] for l in section["lines"]
                           if l.get("start_sec") is not None]
            line_ends = [l["end_sec"] for l in section["lines"]
                         if l.get("end_sec") is not None]
            if line_starts and line_ends:
                section["start_time"] = round(float(min(line_starts)), 3)
                section["end_time"] = round(float(max(line_ends)), 3)
                continue
        per_ref_bounds = [r.get("section_bounds", {}).get(sid) for r in per_ref]
        per_ref_starts = [b[0] for b in per_ref_bounds if b is not None]
        per_ref_ends = [b[1] for b in per_ref_bounds if b is not None]
        if per_ref_starts and per_ref_ends:
            section["start_time"] = round(float(np.median(per_ref_starts)), 3)
            section["end_time"] = round(float(np.median(per_ref_ends)), 3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a song template from multiple reference recordings.")
    p.add_argument("--song", required=True, help="Song slug (used as song_id, e.g. peggy-o)")
    p.add_argument("--references", required=True, action="append",
                   help="Reference audio file or glob (repeatable). Quote globs in your shell.")
    p.add_argument("--lyrics", type=Path, required=True, help=".lyrics file path")
    p.add_argument("--out", type=Path, required=True, help="Output JSON template path")
    p.add_argument("--title", default=None, help="Display title (default: derived from --song)")
    p.add_argument("--version-notes", default="",
                   help="Free-form notes about which arrangement this represents")
    p.add_argument("--whisper-model", default="base",
                   help="faster-whisper size: tiny | base | small | medium | large-v3")
    p.add_argument("--embedding-model", default=None,
                   help=f"HF model id for the encoder (default: {PRIMARY_MODEL}, "
                        f"falls back to {FALLBACK_MODEL})")
    p.add_argument("--vad", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable Silero VAD pre-filtering in whisper. Off by default "
                        "because VAD drops sung vocals buried in band noise. "
                        "Use --vad to re-enable, --no-vad to be explicit.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    references = _expand_references(args.references)
    if not references:
        log.error("No reference files matched %s", args.references)
        return 1
    if len(references) > 5:
        log.warning("Got %d references; using the first 5 (templates start to get unwieldy).",
                    len(references))
        references = references[:5]

    log.info("Building %s from %d reference(s):", args.song, len(references))
    for r in references:
        log.info("  - %s", r)

    title = args.title or args.song.replace("_", " ").replace("-", " ").title()
    extractor = EmbeddingExtractor(args.embedding_model)

    try:
        template = assemble_template(
            lyrics_path=args.lyrics,
            song_id=args.song,
            title=title,
            version_notes=args.version_notes,
            references=references,
            extractor=extractor,
            model_size=args.whisper_model,
            vad_filter=args.vad,
        )
    except RuntimeError as exc:
        log.error("%s", exc)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(template, indent=2))

    n_sections = len(template["structure"])
    n_lines = sum(len(s["lines"]) for s in template["structure"])
    n_lines_with_emb = sum(
        1 for s in template["structure"] for l in s["lines"]
        if l.get("audio_embedding_blended")
    )
    print(f"\nWrote {args.out}")
    print(f"  references:        {len(references)}")
    print(f"  embedding model:   {template.get('embedding_model')}")
    print(f"  sections:          {n_sections}")
    print(f"  lines:             {n_lines}")
    print(f"  lines w/ blended:  {n_lines_with_emb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
