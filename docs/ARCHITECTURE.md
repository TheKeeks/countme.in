# Architecture

## Goals

1. **Track a singer's position in a song in real time**, line-level accurate.
2. **Handle jam-band tempo elasticity**: a song can be 4 minutes or 14 minutes;
   the system can't be locked to a fixed tempo.
3. **Run on iPhone/iPad in a browser tab**, offline-capable. No app store, no
   install, no venue Wi-Fi required.
4. **Fail gracefully**: if the tracker loses confidence, the singer has a one-tap
   emergency control to resync, and nothing about the system silently melts down.

## Two-system design

### Offline: per-song template builder (Python)

Each song gets a JSON "knowledge base" built once from a reference recording:

- **Structure**: section sequence (intro/verse/chorus/jam/outro), each with an
  elasticity flag (`low`/`medium`/`high`) and optional expected duration range.
  Verses are rigid, jams can stretch 10x.
- **Lyric text**: per-line, indexed and ordered.
- **Audio fingerprints**: global tempo, beat track, key estimate, song-wide chroma
  profile, and per-line chroma signatures (12-d vectors) from the reference audio.
- **Per-line timestamps**: when each line is sung in the reference recording, with
  word-level granularity from forced alignment.

This replaces a naive "train a model on the Dead's entire catalog" approach. The
template is the prior; we're not learning the band's style, we're encoding the
song's known structure.

### Online: live position tracker (JS, in-browser)

Three concurrent signals, fused:

1. **Chroma-DTW**: streaming chroma vectors from the mic, compared to the
   reference recording's chroma sequence and per-line chroma signatures via
   online dynamic time warping. Tolerates tempo elasticity by design.
2. **Whisper ASR**: whisper-tiny.en running in transformers.js, transcribing
   short rolling windows. Each recognized word is fuzzy-matched against the
   expected upcoming lyric text. Strong matches snap the tracker's position.
3. **Section elasticity**: marked-elastic sections (jams) allow the position
   estimate to "hold" without losing belief; non-elastic sections expect tight
   movement.

These feed a posterior over `(section_id, line_index)` maintained as a
particle filter or simplified HMM. When the most-likely line changes, the UI
updates.

## Why not pure end-to-end ML?

It would work, but the failure modes are worse:

- Out-of-distribution audio (loud band, monitor bleed, weird arrangements like
  China → Rider transitions) breaks black-box models in ways you can't debug
  mid-show.
- The DTW + template approach gives interpretable state: "we're 73% sure we're
  in verse 2, line 3." That's debuggable; a vector embedding isn't.
- Templates encode structural knowledge that's hard to learn — like the fact that
  "Drums → Space" can extend 20 minutes before "The Other One" returns.

We can still use embeddings (HuBERT/wav2vec2) as features in the DTW step
later if we want; the architecture doesn't preclude it.

## Browser-only constraints

- **No backend**. Hosted on GitHub Pages. All inference in the tab.
- **Service worker caches** everything for offline use: app shell, song templates,
  and the Whisper model (~40MB for tiny.en quantized).
- **Wake lock** keeps the screen on during a set.
- **Web Audio API** captures mic in real time. Echo cancellation / AGC / noise
  suppression are explicitly *disabled* — we want raw vocal, not VoIP-processed.

## Failure handling

- If the tracker's confidence drops below a threshold for >5s, status changes to
  "lost" (visible in the control panel).
- The singer can tap the top of the screen to reveal controls, then "I'm lost"
  to bring up a tappable line picker. One tap resyncs.
- If the mic fails to start (permission denied, hardware issue), a visible error
  surfaces rather than silently doing nothing.

## What gets shipped vs cached

| Source | Size | Where it lives | Cached for offline |
|--------|------|----------------|--------------------|
| App shell (HTML/CSS/JS) | ~50KB | GitHub Pages | Yes, service worker shell cache |
| Song templates (per song) | ~50-100KB each | GitHub Pages | Yes, stale-while-revalidate |
| Whisper-tiny.en model | ~40MB | Hugging Face CDN | Yes, cache-first |
| Reference audio | Local-only, not deployed | (`.gitignore`d) | N/A |

After one first-run with Wi-Fi, the device has everything it needs to run a full
gig offline.
