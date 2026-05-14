"""
alignment.py
------------
Phase 2: populate the line-level timestamps in a song template by
force-aligning the reference audio against the structured lyrics.

Pipeline:
1. Transcribe the audio with faster-whisper, getting per-word timestamps.
2. Sequence-match the recognized word stream against the expected lyric
   word stream (difflib SequenceMatcher).
3. For each expected lyric line, derive start_sec / end_sec from the
   matched words that fell inside it.
4. Lines with no direct matches are interpolated linearly between
   surrounding anchors.
5. For each line with a usable timestamp, sample the audio's chroma at
   that window and store a 12-d "chroma signature" the live runtime
   uses as an acoustic fingerprint.

Usage:
    python alignment.py \
        --audio peggy_o.m4a \
        --template peggy_o.json \
        --out peggy_o_aligned.json \
        [--model base]   # tiny | base | small | medium | large-v3
"""

import argparse
import json
import re
from pathlib import Path
from difflib import SequenceMatcher

import librosa
from faster_whisper import WhisperModel


WORD_RE = re.compile(r"[a-z0-9']+")


def normalize(word: str) -> str:
    matches = WORD_RE.findall(word.lower())
    return matches[0] if matches else ""


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: Path, model_size: str = "base") -> list:
    """Run faster-whisper. Return [{text, start, end}, ...] (normalized words)."""
    print(f"  loading whisper model '{model_size}'...", flush=True)
    model = WhisperModel(model_size, compute_type="int8")
    print(f"  transcribing {audio_path.name}...", flush=True)
    segments, _ = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        vad_filter=True,
        language="en",
    )
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            t = normalize(w.word)
            if t:
                words.append({"text": t, "start": float(w.start), "end": float(w.end)})
    print(f"  recognized {len(words)} words")
    return words


# ---------------------------------------------------------------------------
# Lyric flattening + alignment
# ---------------------------------------------------------------------------

def flatten_lyrics(template: dict) -> list:
    """Flatten the template into a word stream with line/section context."""
    stream = []
    for section in template["structure"]:
        for line in section["lines"]:
            for w in line["text"].split():
                n = normalize(w)
                if n:
                    stream.append({
                        "section_id": section["section_id"],
                        "line_idx": line["line_index"],
                        "word": n,
                    })
    return stream


def align(recognized: list, expected: list) -> list:
    """
    Sequence-align recognized vs expected word streams.
    Returns a list parallel to `expected`; each entry is either
    None (unmatched) or {start, end} from the matched recognized word.
    """
    rec = [r["text"] for r in recognized]
    exp = [e["word"] for e in expected]
    sm = SequenceMatcher(a=exp, b=rec, autojunk=False)

    matched = [None] * len(expected)
    for block in sm.get_matching_blocks():
        for i in range(block.size):
            ei, ri = block.a + i, block.b + i
            matched[ei] = {
                "start": recognized[ri]["start"],
                "end": recognized[ri]["end"],
            }
    return matched


def collect_line_spans(expected: list, matches: list) -> dict:
    """Group matched word times by line, return {(section_id, line_idx): span}."""
    by_line: dict = {}
    for i, exp in enumerate(expected):
        if matches[i] is not None:
            key = (exp["section_id"], exp["line_idx"])
            by_line.setdefault(key, []).append(matches[i])

    spans = {}
    for key, times in by_line.items():
        spans[key] = {
            "start_sec": min(t["start"] for t in times),
            "end_sec": max(t["end"] for t in times),
            "matched_word_count": len(times),
            "interpolated": False,
        }
    return spans


def interpolate_missing(template: dict, spans: dict) -> dict:
    """Fill in lines without direct matches by linear interpolation."""
    ordered = []
    for section in template["structure"]:
        for line in section["lines"]:
            ordered.append((section["section_id"], line["line_index"]))

    known = [(i, spans[k]["start_sec"], spans[k]["end_sec"])
             for i, k in enumerate(ordered) if k in spans]

    if not known:
        return spans

    for i, k in enumerate(ordered):
        if k in spans:
            continue
        prev_a = next((kn for kn in reversed(known) if kn[0] < i), None)
        next_a = next((kn for kn in known if kn[0] > i), None)
        if prev_a and next_a:
            frac = (i - prev_a[0]) / (next_a[0] - prev_a[0])
            est_start = prev_a[2] + frac * (next_a[1] - prev_a[2])
            spans[k] = {
                "start_sec": est_start,
                "end_sec": est_start + 3.0,
                "matched_word_count": 0,
                "interpolated": True,
            }
        elif prev_a:
            spans[k] = {
                "start_sec": prev_a[2] + 1.0,
                "end_sec": prev_a[2] + 4.0,
                "matched_word_count": 0,
                "interpolated": True,
            }
        elif next_a:
            spans[k] = {
                "start_sec": max(0, next_a[1] - 3.0),
                "end_sec": next_a[1],
                "matched_word_count": 0,
                "interpolated": True,
            }
    return spans


# ---------------------------------------------------------------------------
# Chroma signature per line
# ---------------------------------------------------------------------------

def compute_chroma_signature(y, sr, start_sec, end_sec):
    s0 = int(max(0, start_sec) * sr)
    s1 = int(min(len(y) / sr, end_sec) * sr)
    if s1 - s0 < sr // 10:  # need at least 100ms
        return None
    chroma = librosa.feature.chroma_cqt(y=y[s0:s1], sr=sr)
    return [float(c) for c in chroma.mean(axis=1)]


# ---------------------------------------------------------------------------
# Apply to template
# ---------------------------------------------------------------------------

def apply(template: dict, spans: dict, audio_path: Path):
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)

    matched, interp = 0, 0
    for section in template["structure"]:
        for line in section["lines"]:
            key = (section["section_id"], line["line_index"])
            if key not in spans:
                continue
            sp = spans[key]
            line["start_sec"] = round(sp["start_sec"], 3)
            line["end_sec"] = round(sp["end_sec"], 3)
            line["matched_word_count"] = sp["matched_word_count"]
            line["interpolated"] = sp["interpolated"]
            line["chroma_signature"] = compute_chroma_signature(
                y, sr, sp["start_sec"], sp["end_sec"]
            )
            if sp["interpolated"]:
                interp += 1
            else:
                matched += 1

    # Roll section bounds up from line bounds
    for section in template["structure"]:
        starts = [l["start_sec"] for l in section["lines"]
                  if l.get("start_sec") is not None]
        ends = [l["end_sec"] for l in section["lines"]
                if l.get("end_sec") is not None]
        if starts:
            section["start_sec"] = round(max(0, min(starts) - 0.5), 3)
        if ends:
            section["end_sec"] = round(max(ends) + 0.5, 3)

    print(f"  lines directly matched: {matched}")
    print(f"  lines interpolated:     {interp}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default="base",
                   help="faster-whisper size: tiny | base | small | medium | large-v3")
    p.add_argument("--transcript", type=Path, default=None,
                   help="Optional pre-computed transcript JSON [{text, start, end}, ...] "
                        "to skip whisper (useful for re-runs or restricted environments)")
    args = p.parse_args()

    template = json.loads(args.template.read_text())
    expected = flatten_lyrics(template)
    print(f"expected words: {len(expected)}")

    if args.transcript is not None:
        recognized = json.loads(args.transcript.read_text())
        # normalize in case the input wasn't pre-normalized
        recognized = [{"text": normalize(r["text"]), "start": float(r["start"]),
                       "end": float(r["end"])} for r in recognized if normalize(r["text"])]
        print(f"  loaded {len(recognized)} words from {args.transcript}")
    else:
        recognized = transcribe(args.audio, args.model)
    matches = align(recognized, expected)
    matched_n = sum(1 for m in matches if m is not None)
    print(f"matched expected words: {matched_n} / {len(expected)} "
          f"({100*matched_n/max(1,len(expected)):.0f}%)")

    spans = collect_line_spans(expected, matches)
    spans = interpolate_missing(template, spans)
    apply(template, spans, args.audio)

    args.out.write_text(json.dumps(template, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
