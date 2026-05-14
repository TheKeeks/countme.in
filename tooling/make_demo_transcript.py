"""
make_demo_transcript.py
-----------------------
Generate a synthetic ASR transcript that simulates faster-whisper output
on the real Peggy-O recording. Lets us test alignment.py in this sandbox
without downloading a whisper model.

Produces a JSON file: [{text, start, end}, ...]

NOT NEEDED for production -- this just lets us demo the alignment logic
end-to-end here. On your machine, run alignment.py without --transcript
and it'll call whisper for real.
"""
import json
import random
from pathlib import Path

# Section layout (rough estimates for a 7:30 Peggy-O performance).
# Numbers are best-guess durations from listening to typical Dead arrangements.
# When alignment.py runs against real whisper, these get replaced with reality.
SECTION_TIMING = [
    ("intro_1", 0.0, 18.0, []),
    ("verse_1", 18.0, 53.0, [
        "as we rode out to fennario",
        "as we rode out to fennario",
        "our captain fell in love with a lady like a dove",
        "and he called her by name pretty peggy o",
    ]),
    ("verse_2", 53.0, 88.0, [
        "will you marry me pretty peggy o",
        "will you marry me pretty peggy o",
        "if you will marry me i'll set your cities free",
        "and free all the ladies in the area o",
    ]),
    ("verse_3", 88.0, 123.0, [
        "i would marry you sweet william o",
        "i would marry you sweet william o",
        "i would marry you but your guineas are too few",
        "and i fear my mama would be angry o",
    ]),
    ("verse_4", 123.0, 158.0, [
        "what would your mama think pretty peggy o",
        "what would your mama think pretty peggy o",
        "what would your mama think if she heard my guineas clink",
        "saw me marching at the head of my soldiers o",
    ]),
    ("jam_1", 158.0, 295.0, []),
    ("verse_5", 295.0, 330.0, [
        "if ever i return pretty peggy o",
        "if ever i return pretty peggy o",
        "if ever i return your cities i will burn",
        "destroy all the ladies in the area o",
    ]),
    ("verse_6", 330.0, 365.0, [
        "come steppin down the stairs pretty peggy o",
        "come steppin down the stairs pretty peggy o",
        "come steppin down the stairs combin back your yellow hair",
        "bid a last farewell to your william o",
    ]),
    ("verse_7", 365.0, 400.0, [
        "sweet william he is dead pretty peggy o",
        "sweet william he is dead pretty peggy o",
        "sweet william he is dead and he died for a maid",
        "and he's buried in the louisiana country o",
    ]),
    ("verse_8", 400.0, 435.0, [
        "as we rode out to fennario",
        "as we rode out to fennario",
        "our captain fell in love with a lady like a dove",
        "and he called her by name pretty peggy o",
    ]),
    ("outro_1", 435.0, 450.0, []),
]

# A few simulated ASR mishearings (sung vocals are hard).
MISHEARS = {
    "fennario": ["finario", "ferrario", "fennarrio"],
    "peggy": ["piggy", "paggy"],
    "guineas": ["guineas", "kennys", "guineys"],
    "fennario": ["fennario", "fennaro"],
    "william": ["william", "willem"],
}


def jitter(t: float, amount: float = 0.06) -> float:
    return max(0.0, t + random.uniform(-amount, amount))


def main():
    random.seed(7)  # reproducible
    words_out = []

    for section_id, sec_start, sec_end, lines in SECTION_TIMING:
        if not lines:
            continue  # instrumental section, no sung words

        n_lines = len(lines)
        line_dur = (sec_end - sec_start) / n_lines

        for line_i, line_text in enumerate(lines):
            line_start = sec_start + line_i * line_dur
            line_end = line_start + line_dur * 0.85  # leave gap between lines

            words = line_text.split()
            n_words = len(words)
            if n_words == 0:
                continue
            per_word = (line_end - line_start) / n_words

            for wi, w in enumerate(words):
                # 8% chance whisper "drops" a word entirely
                if random.random() < 0.08:
                    continue
                # 12% chance whisper mishears (per known confusable)
                token = w
                if w in MISHEARS and random.random() < 0.12:
                    token = random.choice(MISHEARS[w])

                ws = line_start + wi * per_word
                we = ws + per_word * 0.9
                words_out.append({
                    "text": token,
                    "start": round(jitter(ws), 3),
                    "end": round(jitter(we), 3),
                })

    out_path = Path("demo_transcript.json")
    out_path.write_text(json.dumps(words_out, indent=2))
    print(f"wrote {out_path} ({len(words_out)} words)")


if __name__ == "__main__":
    main()
