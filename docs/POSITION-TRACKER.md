# Position tracker (Phase 3): how the live engine works

`web/js/position-tracker.js` replaced the timer stub with a real
engine: a joint **(position × tempo-rate) HMM** over the template's
reference timeline, updated twice a second from the raw mic stream.
No source separation, no neural nets at runtime — FFT features and a
banded Viterbi update, ~400× real time in node on one core.

## Signals

Two per-0.5s observations, both computed on the same 16 kHz stream:

1. **Chroma** (`web/js/chroma.js`): L2-normalized 12-bin pitch-class
   profile (4096-pt FFT, 65–2093 Hz, 1/f octave compensation), scored
   by cosine similarity (×κ=8) against the template's reference chroma
   (`web/templates/<song>_chroma.json`, exported at 2 fps by
   `tooling/export_web_chroma.py` from the template's
   `chroma_reference`).
2. **Vocal presence** (`web/js/vocal-onset.js`, see
   docs/VOCAL-ONSET-FUSION.md): the per-song vocal head's smoothed
   P(vocal), weighted by its self-confidence gate (×κ=6·gate). States
   in sections with lyric lines expect P(vocal) high; instrumental
   sections expect it low. This term is what holds the tracker at the
   end of the jam until the singer actually re-enters.

## Why rate is a hidden state

The first prototype (position-only HMM) hit a precise failure: verses
of the same song are chroma-identical repetitions, so nothing in the
observation distinguishes verse 2 from verse 3 and the tracker
aliased forward through the verse block at up to 2× (verse_4 predicted
53 s early on take 1), landing in the jam ~70 s before the band got
there. Tightening per-step speed limits didn't fix it — the drift was
sustained, not instantaneous.

The fix models what bands actually do: **tempo is consistent within a
performance**. Rate (7 values, 0.5×–1.75× of reference) is a hidden
state with a switching cost. In `low`-elasticity sections (verses) the
position must advance at the current rate each step; in `medium`/`high`
sections (intro, jam, outro) movement is free — a 53 s reference intro
can map to a 13 s band intro, and the jam can stretch arbitrarily.
This took take-1 strict per-verse accuracy from 47% → 95%.

## Measured results (real band takes, node harness on the shipped JS)

Scored against the hand-labeled ground truth; "block" treats any verse
of the correct verse-block as correct (takes 2/3 only have block-level
labels), "strict" requires the exact verse (only take 1 has per-verse
labels). Baseline: the MERT-only experiment ran 21% raw / 55% smoothed.

| take | mix | block | strict | verse_5 re-entry | state |
|---|---|---|---|---|---|
| 1 (Feb 26) | good, lossless | **98%** | **95%** | **+3.0 s** | locked |
| 2 | good, AAC | **96%** | n/a | **+3.0 s** | locked |
| 3 | hot, vocal buried | **86%** | n/a | +16.0 s | locked |

Take 3 runs with the vocal gate closed (the head is unreliable on that
mix and says so), i.e. the 86% row is the chroma-only fallback quality.
The Python prototype and the shipped JS agree within ±1 s / 1–2 points
(the residual difference is the causal one-window vocal pairing lag).

## Contract with the UI

Unchanged public API: `start()/stop()/snapTo()/consume(frame)`, with
`onPositionChange({sectionId, lineIndex, confidence})` fired on line
changes. During instrumental sections the emitted line is the next
upcoming one, so the singer sees what's coming. `state` moves between
`listening / locked / lost` from the posterior mass within ±10 s of the
argmax. `snapTo` (the emergency overlay) re-concentrates the posterior
at the tapped line — it plays nicely with the filter rather than
fighting it. Songs without a `<song>_chroma.json` resource fall back to
the legacy timer stub.

## Adding a song

1. Build the template as usual (`build_song.py` + `chroma_template.py`).
2. `python tooling/export_web_chroma.py --template tooling/songs/<id>_aligned.json --output web/templates/<id>_chroma.json`
3. Optional but recommended: label a rehearsal take
   (`tooling/ground_truth/`) and train the vocal head
   (`tooling/train_vocal_head.py` → `web/templates/<id>_vocal_head.json`).

## Known limitations

- Per-verse identity within a repeated-verse block rests on the rate
  model, not on the observations; a band that drastically changes
  tempo mid-verse-block will smear it. The natural disambiguator is
  lyric recognition, but rung 4 (tooling/word_anchor_probe.py)
  falsified it on the current recordings: Whisper on the raw band mix
  produces pure hallucination loops at tiny.en/base.en (zero real
  words) and at small.en catches one real phrase ("will you marry me")
  then loops it across the whole recording -- anchors from it would be
  wrong-verse hits everywhere. The blocker is the same one rungs 1-2
  hit: the vocal sits very low in these room recordings. The
  "train on the lyrics instead of parsing them" variant (rung 4b,
  query-by-example: match live audio against exemplar snippets of the
  band singing each verse's opening line, taken from labeled takes)
  was also falsified: cross-take, the true verse ranks at chance in
  both mel and MERT-layer-9 feature spaces (ranks 5/3/3/3 and 3/1/7/3
  of 8) and jam windows score in the same range as real verses -- the
  verse-identity information is simply not present in these
  recordings at any feature level. Words cannot be recovered by any
  model, trained or parsed, if the audio doesn't carry them. And the
  recordings ARE deployment audio -- band-confirmed as captured from
  the mic-stand position, the best available -- so word-level anchors
  are infeasible for this deployment, not merely untested. The only
  route to lyric anchors would be a direct feed (vocal channel from
  the mixer into the device); with mic-stand air capture, the
  vocal-presence head is the strongest voice signal extractable and
  exact-verse identity rests on the rate model.
- Single song validated so far (three takes). The pipeline is
  per-song by construction; new songs need the two export steps above.
- The verse-5 hold depends on the vocal head's gate opening; on
  hot mixes the tracker degrades to chroma-only (take 3 row).
