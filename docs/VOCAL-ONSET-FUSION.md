# Vocal-onset fusion: the live re-anchor signal

How the tracker knows the singer came back in after the jam — without
Demucs, without a neural VAD, on raw phone-mic audio.

## Why this design (the probe ladder)

Four probes on real band recordings of Peggy-O drove every decision here:

| rung | approach | result |
|---|---|---|
| 1 | energy / bandpass envelopes (`raw_onset_probe.py`) | ✗ vocal-minus-jam contrast ~0 dB on the raw mix |
| 2 | Silero VAD (`vad_probe.py`) | ✗ P(speech) ≈ 0.00 on singing over a band; fires only on actual talking |
| 3 | logistic head on log-mel windows (`svd_probe.py`) | ✓ AUC 0.96, verse_5 at +0.5s (within-recording) |
| 3b | same head across recordings (`xrec_probe.py`) | ✓ verse_1 −0.5s, verse_5 +4.0s on a never-seen take; **but** unreliable on a hot mix (take 3: AUC ~0.72) |

Take 3 (fastest, hottest mix, vocal low in the blend) is the permanent
hard case: anchors can land within ±1s yet the jam leaks false onsets.
Conclusion baked into this design: **the vocal-onset signal is a
confidence-gated evidence term fused into the tracker's posterior —
never a standalone re-anchor trigger.**

## Signal chain (browser, `web/js/vocal-onset.js`)

```
mic 16 kHz mono (AudioEngine, raw: no AGC/NS/EC)
  → 1024-pt Hann frames, hop 512 (~32 ms), streaming (center=false)
  → power spectrum → 64-band Slaney mel (0–8 kHz)
  → dB, running-max reference (causal ref=np.max), floor −80 dB
  → pooled per 0.5 s window: mean + std (128 dims)
  → z-score (train-set stats) → logistic → P(vocal)
  → trailing 2 s moving average
  → sustained crossing: threshold 0.5 held ≥ 1.5 s → ONSET event
  → gate: q90−q10 spread of smoothed P over an expanding window
```

Every constant lives in the weights JSON (`feature_config`, `gate`), not
in code. Measured cost: **673× real time** in node on one core; per-day
battery cost on a phone is noise.

### Why these choices

- **16 kHz / streaming frames / causal everything**: the features are
  computed *exactly* as the browser hears audio. The trainer
  (`tooling/train_vocal_head.py`) mirrors this — `center=False`,
  running-max dB reference instead of librosa's whole-file `ref=np.max`,
  trailing (not centered) smoothing — so shipped weights mean what they
  say. Retraining at this config was validated: cross-take AUC 0.956
  (train take 1 → take 2), anchors −2.0s / −1.0s, zero jam false alarms.
- **Onset = start of the run**, fired once the run survives 1.5 s.
  Detection latency is therefore ~1.5–2.5 s after the true entry
  (sustain + smoothing lag), comfortably inside the ±5 s anchor
  tolerance the validator works with.
- **Gate over an expanding window** (not trailing): during a long jam a
  trailing window fills with lows, the spread collapses, and the gate
  would shut exactly when the re-entry matters. The expanding window
  keeps the verses' highs in view. `min_sec: 60` keeps the gate closed
  until enough of the song has been heard (it also suppresses pre-song
  noodling onsets: on take 2, the 0:24/0:29 noodling onsets fired with
  gate 0.00).

## Calibration & shipping

`tooling/train_vocal_head.py` trains on any number of labeled takes
(the `tooling/ground_truth/*_band*.json` files are the labels) and
writes one JSON per song, e.g. `web/templates/peggy_o_vocal_head.json`
(~12 KB): feature config, scaler, coefficients, gate constants,
provenance (`trained_on`), eval results, and a **parity vector** — the
expected features of a deterministic synthetic signal.
`VocalOnsetDetector.parityCheck()` recomputes it at load time and flags
any drift between the JS DSP and the Python trainer (measured:
0.0005 dB max difference; live curves match Python to 5 decimals on a
full take).

More calibration takes help: adding take 2 to training moved take-3
anchors from one-hit to both-within-±1s in the probe study. Record a
take, confirm two timestamps, add a ground-truth file, retrain — the
head is song-specific and takes seconds to train.

The current shipped head for peggy_o is trained on takes 1+2. Serving
and offline caching ride the existing `/templates/`
stale-while-revalidate path; `vocal-onset.js` is in the service-worker
shell cache.

## Fusion semantics (Phase-3 tracker contract)

The detector exposes two callbacks; the tracker consumes both:

- `onWindow({time, prob, smoothed, gate})` — 2 Hz. Continuous weak
  evidence: P(vocal) low across recent windows supports "we are in the
  jam"; high supports "someone is singing". Weight by `gate`.
- `onOnset({time, gate, smoothed})` — sparse. Evidence spike for
  transitions into vocal-flagged sections (the template knows which
  section starts are vocal entries after instrumental sections —
  verse_1 after intro, verse_5 after jam). The tracker should multiply
  its transition prior for those boundaries, scaled by `gate`, **never
  hard-snap**. Chroma-DTW remains the base posterior; a gated-off
  detector (gate 0) degrades to exactly today's behavior.

The verse_5 walk-through, as measured end-to-end in the JS module on
take 2 (head trained on take 1 only — a true cross-recording run):
verses fire onsets with gate ~1.0, the 122 s jam produces **zero
onsets**, and the first post-jam onset lands at 5:02.0 with gate 0.93
against a band-confirmed re-entry of 5:03 (−1.0 s). On take 3 (hot
mix), the same module fires 44 onsets — **every one carrying gate
0.00**: the tracker ignores them all and never re-anchors on bad
evidence.

## Failure modes & mitigations

| mode | evidence | mitigation |
|---|---|---|
| vocal buried in a hot mix | take 3: AUC 0.72, jam false onsets | gate reads spread 0.67 → weight 0; chroma-DTW only |
| muffled/soft re-entry | take 2: +4.0 s (probe), −1.0 s (browser config) | within tolerance; onset is evidence, not a snap |
| shouted interjection mid-jam ("damn") | take 3 at 3:12/3:19 | brief bursts don't survive the 1.5 s sustain; excluded from training labels (`_damn` section) |
| browser ignores 16 kHz hint | iOS variance | detector linear-resamples any input rate |
| DSP drift between JS and Python | — | load-time parity check against the shipped vector |
| tempo variation | takes are 7.5 / 6.8 / 5.9 min versions of the same song | features are 0.5 s spectral stats; no tempo dependence observed |

## Acceptance criteria for going live

1. Parity check passes on device (`parityCheck().ok`).
2. On a good-mix rehearsal take: re-entry onset within ±5 s, zero
   gated-in false onsets during the jam. (Currently: −1.0 s, zero.)
3. On the hot-mix take: no onset with gate > 0. (Currently: 44/44 at 0.)
4. Runtime: detector CPU ≪ real time on target devices. (673× in node.)
5. Tracker integration (Phase 3): with the gate forced to 0 the tracker
   behaves identically to chroma-DTW-only — the term can only add.

## Files

- `web/js/vocal-onset.js` — runtime detector (this spec's signal chain)
- `web/templates/peggy_o_vocal_head.json` — shipped head, takes 1+2
- `tooling/train_vocal_head.py` — trainer / evaluator / exporter
- `tooling/ground_truth/peggy_o_band*.json` — labels (3 takes)
- Probe history: `tooling/raw_onset_probe.py`, `vad_probe.py`,
  `svd_probe.py`, `xrec_probe.py`
