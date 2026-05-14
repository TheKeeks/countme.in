# How to add a new song

End-to-end: from "I have a recording" to "the song shows up on stage."

## 1. Gather inputs

You need two things:

- **A reference recording** of your band performing the song (board tape, rehearsal
  recording, or a well-known Dead live cut as a stand-in). WAV, MP3, M4A all fine.
- **Your lyric text** — what your singer actually sings, not what's on a lyrics
  website. Small variations matter for ASR matching.

## 2. Write a `.lyrics` file

Drop it in `tooling/songs/`. Use `peggy_o.lyrics` as the reference. Format:

```
[intro] elasticity=medium duration_range=10,40
(instrumental notes go in parentheses; they aren't sung)

[verse 1] elasticity=low
First line of verse 1
Second line of verse 1
...

[jam] elasticity=high duration_range=60,300
(can stretch wildly)

[verse 2] elasticity=low
...
```

Section types: `intro`, `verse`, `chorus`, `pre-chorus`, `bridge`, `jam`, `solo`,
`outro`, or anything else you want — the type is a label, not a constraint.

Section attributes:
- `elasticity` = `low` | `medium` | `high`. How much the section can stretch.
  Verses/choruses: `low`. Jams/solos: `high`. Intros/outros: `medium`.
- `duration_range` = MIN,MAX seconds. Optional. Used by the live tracker to
  constrain its belief.

## 3. Build the template (Phase 1)

```bash
cd tooling
python template_builder.py \
    --lyrics songs/your_song.lyrics \
    --audio /path/to/reference.m4a \
    --out songs/your_song.json \
    --version-notes "Reference: Show date or recording info"
```

This produces `your_song.json` with structure + global audio features.

## 4. Align line timestamps (Phase 2)

```bash
python alignment.py \
    --audio /path/to/reference.m4a \
    --template songs/your_song.json \
    --out songs/your_song_aligned.json \
    --model base    # or 'small'/'medium' for better sung-vocal accuracy
```

This calls Whisper, gets word timestamps, fuzzy-matches against your lyric text,
and populates per-line `start_sec` / `end_sec` / `chroma_signature`.

**Check the output**: look at `matched_word_count` per line. If most lines have 0
or 1 matches, either the audio is too noisy or the lyric text doesn't match what's
sung. Re-run with a larger model (`--model small` or `medium`) or edit the lyrics
file to match the actual delivery.

## 5. Ship it to the web app

```bash
cp songs/your_song_aligned.json ../web/templates/
```

Then edit `web/js/template-loader.js`:

```js
const TEMPLATE_INDEX = [
  'peggy_o_aligned.json',
  'your_song_aligned.json',   // ← add this
];
```

## 6. Test locally

```bash
cd ../web
python3 -m http.server 8000
# open http://localhost:8000
```

Your song should appear on the home screen.

## 7. Deploy

```bash
git add tooling/songs/your_song.lyrics tooling/songs/your_song_aligned.json \
        web/templates/your_song_aligned.json web/js/template-loader.js
git commit -m "Add Your Song"
git push
```

GitHub Actions deploys automatically. New version is live in ~1 minute.
