"""
detect_first_verse.py
---------------------
Probe: can Whisper recognise the first verse start in phone-stand band
audio?

MERT-only position tracking landed at 21% raw / 55% smoothed on the
band recording (the smoothed number is essentially the time prior
talking), so we need a complementary signal. This script measures
Whisper's ability to detect a distinctive opening lyric ("as we rode
out" for Peggy O) in arbitrary audio and report when it heard it.

If this works on the real band audio we'll plumb Whisper into the
validator as a separate emission term. If it doesn't, we need a
different plan. This module is intentionally standalone -- not
integrated -- to keep the probe small.

CLI:

    python tooling/detect_first_verse.py \\
        --audio path/to/band_recording.mp3 \\
        --search-text "as we rode out" \\
        --tap-time 30.0 \\
        --ground-truth-time 53.0 \\
        --output detect/report.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# template_io lives next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from template_io import get_song_lyrics  # noqa: E402


# ---------------------------------------------------------------------------
# Text normalisation + fuzzy search
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip punctuation (keeping apostrophes), collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _fuzzy_search(words: list[dict], search_text: str,
                  min_score: float = 70.0) -> list[dict]:
    """Slide a window over `words` and find all matches above `min_score`.

    `words` is a list of `{"word": str, "start": float, "end": float}`
    dicts (Whisper's output, normalised below). Each returned candidate
    has `score`, `matched_text`, and `audio_t` (start time of the first
    word in the matched window).
    """
    from rapidfuzz import fuzz

    norm_search = _normalize(search_text)
    search_tokens = norm_search.split()
    n = len(search_tokens)
    if n == 0 or len(words) < n:
        return []

    candidates: list[dict] = []
    for i in range(len(words) - n + 1):
        window = words[i : i + n]
        window_text = " ".join(_normalize(w["word"]) for w in window).strip()
        if not window_text:
            continue
        score = float(fuzz.ratio(window_text, norm_search))
        if score >= min_score:
            candidates.append({
                "score": score,
                "matched_text": " ".join(w["word"].strip() for w in window).strip(),
                "audio_t": float(window[0]["start"]),
            })
    return candidates


# ---------------------------------------------------------------------------
# Audio + Whisper
# ---------------------------------------------------------------------------

def _ensure_wav(audio_path: Path, target_sr: int = 16000) -> Path:
    """Pre-convert any input to a clean mono WAV at `target_sr` via ffmpeg.

    Sidesteps the librosa+audioread MP3 quirks the validator hit on
    rehearsal recordings. Caller is responsible for unlinking the result.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = Path(tf.name)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(audio_path),
         "-ar", str(target_sr), "-ac", "1", "-f", "wav", str(wav_path)],
        check=True,
    )
    return wav_path


def _vocal_stem_rms_db(vocals_path: Path) -> float:
    """RMS energy of the vocal stem in dBFS. A very low value (< ~-50 dB)
    suggests the source separator found ~no vocal content even though it
    produced an output file.
    """
    import numpy as np
    import soundfile as sf

    y, _sr = sf.read(str(vocals_path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    return 20.0 * float(np.log10(max(rms, 1e-12)))


def _separate_vocals(audio_path: Path) -> tuple[Path, Path, float]:
    """Run Demucs htdemucs on `audio_path` and return (vocals_wav,
    cleanup_dir, rms_db).

    Caller is responsible for `shutil.rmtree(cleanup_dir)` once it's
    done with the vocals stem. We use --two-stems vocals so Demucs only
    writes the vocal + no_vocal split instead of all four stems.
    """
    out_dir = Path(tempfile.mkdtemp(prefix="demucs_"))
    print("  running Demucs (htdemucs, two-stem vocals) ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "demucs.separate",
         "-n", "htdemucs",
         "--two-stems", "vocals",
         "-o", str(out_dir),
         str(audio_path)],
        check=True,
    )
    candidates = list(out_dir.rglob("vocals.wav"))
    if not candidates:
        raise RuntimeError(
            f"Demucs ran but produced no vocals.wav under {out_dir}"
        )
    vocals_path = candidates[0]
    rms_db = _vocal_stem_rms_db(vocals_path)
    return vocals_path, out_dir, rms_db


WHISPER_LANGUAGE = "en"
WHISPER_TEMPERATURE = 0.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6


def _transcribe(audio_path: Path, model_size: str,
                initial_prompt: Optional[str] = None) -> tuple[str, list[dict]]:
    """Return (full_text, [{"word", "start", "end"}, ...]).

    Tries faster-whisper first (much faster than openai-whisper, smaller
    int8 footprint). Falls back to openai-whisper if faster-whisper isn't
    available.

    Decoding is pinned to language=en, temperature=0.0 (deterministic; no
    fallback resampling), no_speech_threshold=0.6. `initial_prompt` is
    passed through to bias the decoder toward expected vocabulary when
    set -- the main lever against the "Be out now, maybe I can't do
    anything to leave you behind" hallucination loop we saw on band audio.
    """
    # faster-whisper
    try:
        from faster_whisper import WhisperModel  # noqa: PLC0415
    except ImportError:
        WhisperModel = None  # type: ignore[assignment]

    if WhisperModel is not None:
        print(f"  loading faster-whisper '{model_size}' ...", flush=True)
        model = WhisperModel(model_size, compute_type="int8")
        print(f"  transcribing with faster-whisper ...", flush=True)
        segments, _info = model.transcribe(
            str(audio_path),
            word_timestamps=True,
            language=WHISPER_LANGUAGE,
            temperature=WHISPER_TEMPERATURE,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            initial_prompt=initial_prompt,
        )
        words: list[dict] = []
        text_parts: list[str] = []
        for seg in segments:
            text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    words.append({
                        "word": w.word.strip(),
                        "start": float(w.start),
                        "end": float(w.end),
                    })
        return " ".join(text_parts).strip(), words

    # openai-whisper fallback
    try:
        import whisper  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "Neither faster-whisper nor openai-whisper is installed. "
            "Install one: `pip install faster-whisper` or `pip install -U openai-whisper`."
        ) from exc

    print(f"  loading openai-whisper '{model_size}' ...", flush=True)
    model = whisper.load_model(model_size)
    print(f"  transcribing with openai-whisper ...", flush=True)
    result = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=WHISPER_LANGUAGE,
        temperature=WHISPER_TEMPERATURE,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        initial_prompt=initial_prompt,
    )
    words = []
    for seg in result.get("segments") or []:
        for w in seg.get("words") or []:
            words.append({
                "word": (w.get("word") or "").strip(),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
            })
    return (result.get("text") or "").strip(), words


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _format_mmss(t: float) -> str:
    m = int(t) // 60
    s = t - 60 * m
    return f"{m:02d}:{s:05.2f}"


def _best_match(candidates: list[dict]) -> Optional[dict]:
    """Highest score, ties broken by earliest audio_t."""
    if not candidates:
        return None
    return max(candidates, key=lambda c: (c["score"], -c["audio_t"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--search-text", required=True,
                   help='Lyric phrase to search for (e.g. "as we rode out").')
    p.add_argument("--tap-time", type=float, default=0.0,
                   help="Earliest audio time after which to consider matches "
                        "(default 0.0).")
    p.add_argument("--ground-truth-time", type=float, default=None,
                   help="Optional known verse_1 start time for error reporting.")
    p.add_argument("--model", default="base",
                   help="Whisper model size: tiny / base / small (default base).")
    p.add_argument("--output", type=Path, default=None,
                   help="Optional path to write a detailed JSON report.")
    p.add_argument("--separate-vocals", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Run Demucs htdemucs to isolate the vocal stem before "
                        "transcribing. Off by default; recommended on band-stage "
                        "audio where vocals are buried under instruments.")
    p.add_argument("--initial-prompt", default=None,
                   help="Optional text to seed Whisper's decoder. Biases the "
                        "model toward expected vocabulary -- the main lever "
                        "against hallucinated loops on weak audio.")
    p.add_argument("--template", type=Path, default=None,
                   help="Optional path to a built song template JSON. When "
                        "set, the initial prompt is built automatically from "
                        "the template's lyrics. An explicit --initial-prompt "
                        "takes precedence over this.")
    args = p.parse_args(argv)

    # Resolve the initial prompt: --initial-prompt wins over --template.
    initial_prompt: Optional[str] = None
    initial_prompt_source: Optional[str] = None
    if args.initial_prompt:
        initial_prompt = args.initial_prompt
        initial_prompt_source = "explicit"
        snippet = initial_prompt[:120]
        ellipsis = "..." if len(initial_prompt) > 120 else ""
        print(
            f"Initial prompt from --initial-prompt (explicit): {snippet}{ellipsis}",
            flush=True,
        )
    elif args.template is not None:
        template_dict = json.loads(args.template.read_text())
        derived = get_song_lyrics(template_dict)
        if derived:
            initial_prompt = derived
            initial_prompt_source = "template"
            snippet = derived[:120]
            ellipsis = "..." if len(derived) > 120 else ""
            print(
                f"Initial prompt from --template ({args.template}): "
                f"{snippet}{ellipsis}",
                flush=True,
            )
        else:
            print(
                f"--template ({args.template}) had no lyric lines; "
                f"no initial prompt",
                flush=True,
            )
    else:
        print("No initial prompt", flush=True)

    cleanup_dirs: list[Path] = []
    vocal_rms_db: Optional[float] = None
    audio_for_whisper: Path = args.audio

    if args.separate_vocals:
        print("Running Demucs vocal isolation...", flush=True)
        vocals_path, demucs_dir, vocal_rms_db = _separate_vocals(args.audio)
        cleanup_dirs.append(demucs_dir)
        print(
            f"Vocal stem isolated, RMS energy: {vocal_rms_db:.1f} dB",
            flush=True,
        )
        audio_for_whisper = vocals_path

    wav_path = _ensure_wav(audio_for_whisper)
    try:
        full_text, words = _transcribe(
            wav_path, args.model, initial_prompt=initial_prompt,
        )
    finally:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        for d in cleanup_dirs:
            try:
                import shutil  # noqa: PLC0415
                shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    # Restrict matching to audio_t >= tap_time. Use word.start so a long
    # word straddling the tap doesn't slip through.
    filtered = [w for w in words if w["start"] >= args.tap_time]

    # Excerpt of transcript covering [tap, tap+90s] just for the stdout
    # preview -- doesn't affect matching.
    excerpt_end = args.tap_time + 90.0
    excerpt_words = [w for w in filtered if w["start"] < excerpt_end]
    excerpt_text = " ".join(w["word"] for w in excerpt_words).strip()

    candidates = _fuzzy_search(filtered, args.search_text)
    candidates.sort(key=lambda c: c["audio_t"])

    first_match = candidates[0] if candidates else None
    best_match = _best_match(candidates)

    print()
    print("Whisper transcript (first 90s after tap):")
    print(f"  {excerpt_text}" if excerpt_text else "  (no transcribed words in window)")

    print()
    print("Candidate matches (score >= 70%, chronological):")
    if candidates:
        for c in candidates:
            print(f"  [{c['score']:5.1f}%] {c['matched_text']!r} at "
                  f"audio_t={c['audio_t']:.2f}s ({_format_mmss(c['audio_t'])})")
    else:
        print("  (none)")

    print()
    if first_match is None:
        print("First match: (none above threshold)")
        print("Best match:  (none above threshold)")
        rc = 1
    else:
        print(
            f"First match (earliest after tap, this is the deployment anchor): "
            f"{first_match['matched_text']!r} at audio_t={first_match['audio_t']:.2f}s "
            f"(score={first_match['score']:.1f})"
        )
        assert best_match is not None
        print(
            f"Best match (highest score): "
            f"{best_match['matched_text']!r} at audio_t={best_match['audio_t']:.2f}s "
            f"(score={best_match['score']:.1f})"
        )
        rc = 0

    first_err: Optional[float] = None
    best_err: Optional[float] = None
    if args.ground_truth_time is not None:
        print()
        print(f"Ground truth verse_1 start: {args.ground_truth_time:.2f}s")
        if first_match is not None:
            first_err = first_match["audio_t"] - args.ground_truth_time
            print(f"First-match error: {first_err:+.2f}s")
            assert best_match is not None
            best_err = best_match["audio_t"] - args.ground_truth_time
            print(f"Best-match error:  {best_err:+.2f}s")

    if args.output is not None:
        report = {
            "audio": str(args.audio),
            "search_text": args.search_text,
            "tap_time": args.tap_time,
            "ground_truth_time": args.ground_truth_time,
            "model": args.model,
            "vocal_separation_used": bool(args.separate_vocals),
            "initial_prompt_used": initial_prompt is not None,
            "initial_prompt_text": initial_prompt,
            "initial_prompt_source": initial_prompt_source,
            "whisper_language": WHISPER_LANGUAGE,
            "whisper_temperature": WHISPER_TEMPERATURE,
            "transcript_full_text": full_text,
            "transcript_words": words,
            "candidates": candidates,
            "first_match": first_match,
            "best_match": best_match,
        }
        if vocal_rms_db is not None:
            report["vocal_stem_rms_db"] = vocal_rms_db
        if first_err is not None:
            report["first_match_error_sec"] = first_err
            report["best_match_error_sec"] = best_err
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print()
        print(f"Wrote JSON report to {args.output}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
