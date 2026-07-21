"""
word_anchor_probe.py
--------------------
Probe: does Whisper on RAW band audio produce usable lyric PHRASE
ANCHORS -- occasional correct "I just heard 'will you marry me'" hits
that pin the tracker to an exact line?

Rung 4 of the live-tracking ladder. The Phase-3 tracker (chroma +
vocal presence) cannot tell verse N from verse N+1 -- verses are
musically identical by construction; only the WORDS differ. A single
correct phrase match collapses that ambiguity. The tracker doesn't
need transcription, it needs sparse anchors, so the bar is much lower
than the old first-verse Whisper experiments: some correct hits per
take, and near-zero FALSE hits -- especially inside the jam, where
nobody is singing and any lyric match is a hallucination by
definition.

Method: faster-whisper word timestamps on the raw take (no Demucs, no
separation), slide a window over the transcript per template lyric
line, fuzzy-match (rapidfuzz). Every match >= --min-score becomes an
anchor {time, line, score} and is classified against ground truth:

  correct        match time inside the GT span of the line's verse
                 (block-level GT: the line's verse block)
  wrong_verse    inside a different verse/block -- the dangerous kind
  instrumental   inside intro/jam/outro -- pure hallucination

NO initial_prompt lyric biasing: biasing Whisper with the exact lyrics
we then match against would manufacture false anchors from
hallucinations. The decoder must earn its hits.

CLI:

    python tooling/word_anchor_probe.py \\
        --audio take.wav \\
        --template tooling/songs/peggy_o_aligned.json \\
        --ground-truth tooling/ground_truth/peggy_o_band.json \\
        --model tiny.en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect_first_verse import _ensure_wav, _normalize  # noqa: E402
from raw_onset_probe import _load_sections  # noqa: E402

MIN_WINDOW_WORDS = 3   # phrases shorter than this are too ambiguous
DEDUPE_SEC = 5.0       # keep the best match per (line, ~5s neighborhood)


# ---------------------------------------------------------------------------
# Template lines + verse blocks
# ---------------------------------------------------------------------------

def _template_lines(template: dict) -> list[dict]:
    out = []
    for s in template.get("structure", []):
        for line in s.get("lines", []):
            text = _normalize(line.get("text") or "")
            if not text:
                continue
            out.append({
                "section_id": s["section_id"],
                "line_index": line["line_index"],
                "text": text,
                "n_words": len(text.split()),
            })
    return out


def _blocks(gt_sections: list[dict]) -> dict[str, set[str]]:
    """GT section -> set of template verse ids it covers. Band GT files
    may use block sections (verse_1 spanning verses 1-4)."""
    verse_ids = [f"verse_{i}" for i in range(1, 9)]
    gt_ids = {s["section_id"] for s in gt_sections}
    blocks: dict[str, set[str]] = {}
    if "verse_2" in gt_ids:  # per-verse GT (take 1)
        for v in verse_ids:
            blocks[v] = {v}
        blocks["verse_8"] = {"verse_8", "outro_1"}
    else:  # block GT: verse_1 = verses 1-4, verse_5 = verses 5-8
        blocks["verse_1"] = {"verse_1", "verse_2", "verse_3", "verse_4"}
        blocks["verse_5"] = {"verse_5", "verse_6", "verse_7", "verse_8",
                             "outro_1"}
    return blocks


# ---------------------------------------------------------------------------
# Transcription + matching
# ---------------------------------------------------------------------------

def _transcribe(audio: Path, model_size: str) -> list[dict]:
    from faster_whisper import WhisperModel

    print(f"  transcribing with faster-whisper {model_size} ...", flush=True)
    model = WhisperModel(model_size, compute_type="int8")
    segments, _info = model.transcribe(
        str(audio), word_timestamps=True, language="en",
        temperature=0.0, no_speech_threshold=0.6,
    )
    words = []
    for seg in segments:
        for w in seg.words or []:
            token = _normalize(w.word)
            if token:
                words.append({"word": token, "start": float(w.start),
                              "end": float(w.end)})
    return words


def _find_anchors(words: list[dict], lines: list[dict],
                  min_score: float) -> list[dict]:
    from rapidfuzz import fuzz

    anchors: list[dict] = []
    for line in lines:
        n = max(MIN_WINDOW_WORDS, line["n_words"])
        if len(words) < n:
            continue
        best_in_window: Optional[dict] = None
        for i in range(len(words) - n + 1):
            window = words[i:i + n]
            text = " ".join(w["word"] for w in window)
            score = float(fuzz.ratio(text, line["text"]))
            if score < min_score:
                continue
            t = window[0]["start"]
            if best_in_window and t - best_in_window["time"] > DEDUPE_SEC:
                anchors.append(best_in_window)
                best_in_window = None
            if best_in_window is None or score > best_in_window["score"]:
                best_in_window = {
                    "time": t, "score": score,
                    "section_id": line["section_id"],
                    "line_index": line["line_index"],
                    "matched_text": text, "line_text": line["text"],
                }
                best_in_window["time"] = t
        if best_in_window:
            anchors.append(best_in_window)
    anchors.sort(key=lambda a: a["time"])
    return anchors


# ---------------------------------------------------------------------------
# Classification vs ground truth
# ---------------------------------------------------------------------------

def _classify(anchors: list[dict], gt_sections: list[dict],
              blocks: dict[str, set[str]], offset: float) -> None:
    spans = [(s["section_id"], float(s["start"]), float(s["end"]))
             for s in gt_sections if not s["section_id"].startswith("_")]
    for a in anchors:
        t_abs = a["time"] + offset
        gt_sid = next((sid for sid, s0, s1 in spans if s0 <= t_abs < s1),
                      None)
        a["t_abs"] = round(t_abs, 1)
        a["gt_section"] = gt_sid
        if gt_sid is None:
            a["verdict"] = "excluded"
        elif gt_sid.startswith("verse"):
            covered = blocks.get(gt_sid, {gt_sid})
            a["verdict"] = ("correct" if a["section_id"] in covered
                            else "wrong_verse")
        else:
            a["verdict"] = "instrumental"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--song-offset", type=float, default=None,
                   help="Defaults to the ground-truth file's "
                        "song_offset_sec. Audio is assumed to start at "
                        "recording t=0 (NOT pre-sliced).")
    p.add_argument("--model", default="tiny.en",
                   help="faster-whisper model size (default tiny.en -- "
                        "the size a browser deployment would run).")
    p.add_argument("--min-score", type=float, default=80.0)
    p.add_argument("--output-md", type=Path, default=Path("word_anchor.md"))
    p.add_argument("--output-json", type=Path,
                   default=Path("word_anchor.json"))
    args = p.parse_args(argv)

    template = json.loads(args.template.read_text())
    lines = _template_lines(template)
    gt = json.loads(args.ground_truth.read_text())
    offset = (args.song_offset if args.song_offset is not None
              else float(gt.get("song_offset_sec", 0.0)))
    gt_sections = _load_sections(args.ground_truth)
    blocks = _blocks(gt_sections)

    print(f"Audio: {args.audio} | {len(lines)} template lines | "
          f"offset {offset:.1f}s", flush=True)
    wav = _ensure_wav(args.audio, target_sr=16000)
    try:
        words = _transcribe(wav, args.model)
    finally:
        import os
        try:
            os.unlink(wav)
        except OSError:
            pass
    print(f"  {len(words)} words transcribed", flush=True)

    anchors = _find_anchors(words, lines, args.min_score)
    # NOTE: word times are relative to the full recording (we transcribe
    # from t=0 so pre-song talk is visible too); classification handles
    # offset via absolute GT spans. Here offset shifts nothing because
    # audio starts at recording t=0 -- pass offset=0 to _classify's
    # addition. We keep the parameter for pre-sliced audio inputs.
    _classify(anchors, gt_sections, blocks, 0.0)

    counts = {"correct": 0, "wrong_verse": 0, "instrumental": 0,
              "excluded": 0}
    for a in anchors:
        counts[a["verdict"]] += 1
    verse_cov: dict[str, int] = {}
    for a in anchors:
        if a["verdict"] == "correct":
            verse_cov[a["gt_section"]] = verse_cov.get(a["gt_section"], 0) + 1

    print(f"Anchors: {len(anchors)} | correct {counts['correct']} | "
          f"wrong verse {counts['wrong_verse']} | instrumental (jam etc.) "
          f"{counts['instrumental']}", flush=True)
    for a in anchors:
        mark = {"correct": "OK ", "wrong_verse": "BAD",
                "instrumental": "HAL", "excluded": "-- "}[a["verdict"]]
        print(f"  [{mark}] t={a['t_abs']:6.1f}s  {a['section_id']}/"
              f"{a['line_index']} ({a['score']:.0f}) "
              f"{a['matched_text']!r}", flush=True)

    md = [
        "# Word-anchor probe (rung 4)",
        "",
        f"- Audio: `{args.audio}` | model: {args.model} | min score "
        f"{args.min_score:.0f} | no initial_prompt (no lyric biasing)",
        f"- Anchors: {len(anchors)} -- correct {counts['correct']}, "
        f"wrong verse {counts['wrong_verse']}, instrumental "
        f"{counts['instrumental']}",
        f"- Verses with >=1 correct anchor: "
        f"{sorted(verse_cov)} ({len(verse_cov)})",
        "",
        "| verdict | t | template line | score | heard |",
        "|---|---|---|---|---|",
    ]
    for a in anchors:
        md.append(f"| {a['verdict']} | {a['t_abs']:.1f}s | "
                  f"{a['section_id']}/{a['line_index']} | "
                  f"{a['score']:.0f} | {a['matched_text']} |")
    md.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))

    report = {
        "audio": str(args.audio), "model": args.model,
        "min_score": args.min_score, "n_words": len(words),
        "counts": counts, "verse_coverage": verse_cov,
        "anchors": anchors,
        "transcript_words": words,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1))
    print(f"Wrote {args.output_md} and {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
