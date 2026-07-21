/**
 * position-tracker.js — online position tracking against a song template.
 *
 * This is the BRAIN: a joint (position × tempo-rate) HMM over the
 * template's reference timeline, updated twice a second from two live
 * signals, both computed on the raw mic stream (no source separation):
 *
 *   1. CHROMA: an L2-normalized pitch-class profile per 0.5 s step
 *      (chroma.js), scored by cosine similarity against the template's
 *      reference chroma (web/templates/<song>_chroma.json).
 *   2. VOCAL PRESENCE: the per-song vocal head (vocal-onset.js),
 *      weighted by its self-confidence gate. Verse states expect
 *      P(vocal) high; intro/jam/outro states expect it low. This is
 *      what re-anchors the verse-5 re-entry after the jam.
 *
 * The tempo RATE is a hidden state (7 values, 0.5×–1.75× of the
 * reference): in low-elasticity sections (verses) the position must
 * advance at the current rate, which stops the tracker drifting
 * through chroma-identical repeated verses; in medium/high sections
 * (intro/jam) movement is free, letting a 53 s reference intro map to
 * a 13 s band intro and a jam stretch arbitrarily. Rate persists with
 * a switching cost — bands hold tempo within a performance.
 *
 * Validated offline against three real band takes (see
 * docs/POSITION-TRACKER.md): 98% / 97% / 86% section accuracy,
 * verse-5 re-entry within +2–3 s on the good-mix takes. The old
 * timer-based stub survives as the fallback when a song has no chroma
 * resource yet.
 */

import { ChromaExtractor } from './chroma.js';
import { VocalOnsetDetector } from './vocal-onset.js';

const STEP_SEC = 0.5;
const RATES = [0.5, 0.65, 0.8, 1.0, 1.2, 1.45, 1.75];
const MAX_D = 8;
const RATE_SWITCH_COST = 2.5;
const KAPPA = 8.0;        // chroma observation sharpness
const VOCAL_KAPPA = 6.0;  // vocal-consistency weight (scaled by gate)
const LOCK_CONFIDENCE = 0.6;
const LOST_CONFIDENCE = 0.2;
const CONFIDENCE_WINDOW_SEC = 10.0;
// A prompter should essentially never scroll backwards on its own: verses
// are chroma-identical, so the posterior's argmax can flip between "still
// in verse N" and "into verse N+1" hypotheses (field-tested failure: the
// display snapped from verse 2 back to verse 1 line 1 mid-song). Forward
// moves are emitted immediately; backward moves require BOTH sustained
// backward evidence AND the backward mode to decisively dominate the
// posterior mass near the currently displayed position. The user's
// snapTo bypasses this.
const BACKWARD_HOLD_STEPS = 6;      // 3 s at 2 Hz
// The raw argmax races ahead within a verse (its lines are musically
// near-identical) and then corrects back — the display would wobble
// forward/backward by a few lines. Emitting the MEDIAN of the recent
// argmax positions filters the race-and-correct spikes; sustained moves
// pass through as one clean change ~1 s later.
const EMIT_MEDIAN_STEPS = 5;
const BACKWARD_SUPPRESS_LINES = 3; // never show backward hops this small

const MEDIUM_COST = [-2.0, -1.0, -0.5, -1.0, -1.2, -1.5, -1.8, -2.0, -2.2];
const HIGH_COST = [-0.7, -0.9, -1.1, -1.3, -1.5, -1.7, -1.9, -1e9, -1e9];

export class PositionTracker {
  /**
   * @param template  the *_aligned.json song template
   * @param resources optional {chroma, vocalHead}: parsed
   *   <song>_chroma.json and <song>_vocal_head.json. Without chroma the
   *   tracker degrades to the legacy timer stub.
   */
  constructor(template, resources = {}) {
    this.template = template;
    this.state = 'idle'; // idle | listening | locked | lost
    this.onPositionChange = null;
    this._flatLines = this._flattenLines(template);
    this._referenceDuration = template.audio_features?.duration_sec ?? 60;
    this._timer = null;
    this._currentIndex = -1;

    this._live = null;
    if (resources.chroma && resources.chroma.data) {
      this._initEngine(resources);
    } else {
      console.warn('position-tracker: no chroma resource; using timer stub');
    }
  }

  _initEngine(resources) {
    const ref = resources.chroma.data.map(row => Float64Array.from(row));
    const n = ref.length;
    // Per-state metadata from the template structure (reference timeline).
    const elastic = new Uint8Array(n).fill(1); // 0 low, 1 medium, 2 high
    const vocal = new Uint8Array(n);
    const sectionOf = new Array(n).fill(null);
    for (const s of this.template.structure) {
      const a = s.start_time, b = s.end_time;
      if (a == null || b == null) continue;
      const e = { low: 0, medium: 1, high: 2 }[s.elasticity] ?? 1;
      const isVocal = (s.lines && s.lines.length > 0) ? 1 : 0;
      for (let i = Math.floor(a / STEP_SEC);
           i < Math.min(n, Math.floor(b / STEP_SEC)); i++) {
        elastic[i] = e;
        vocal[i] = isVocal;
        sectionOf[i] = s.section_id;
      }
    }
    // Advance-cost table per (elasticity, rate, d).
    const cost = [];
    for (let e = 0; e < 3; e++) {
      const perRate = [];
      for (let ri = 0; ri < RATES.length; ri++) {
        const row = new Float64Array(MAX_D + 1);
        for (let d = 0; d <= MAX_D; d++) {
          if (e === 0) {
            const x = d - 2 * RATES[ri];
            row[d] = -(x * x) / (2 * 0.7 * 0.7);
          } else {
            row[d] = (e === 1 ? MEDIUM_COST : HIGH_COST)[d];
          }
        }
        perRate.push(row);
      }
      cost.push(perRate);
    }

    this._live = {
      ref, n, elastic, vocal, sectionOf, cost,
      chromaExtractor: new ChromaExtractor(),
      vocalDetector: resources.vocalHead
        ? new VocalOnsetDetector(resources.vocalHead) : null,
      logp: null,           // Float64Array(nr * n)
      scratch: null,
      lastVocal: { smoothed: 0.5, gate: 0 },
      stepCount: 0,
      recentConfidence: [],
    };
    const lv = this._live;
    lv.chromaExtractor.onChroma = (e) => this._step(e.vector);
    if (lv.vocalDetector) {
      lv.vocalDetector.onWindow = (w) => {
        lv.lastVocal = { smoothed: w.smoothed, gate: w.gate };
      };
      const parity = lv.vocalDetector.parityCheck();
      if (!parity.ok) {
        console.warn(
          `vocal head parity check FAILED (${parity.maxDiffDb.toFixed(3)} dB); `
          + 'disabling vocal term');
        lv.vocalDetector = null;
      }
    }
  }

  _flattenLines(template) {
    const out = [];
    for (const section of template.structure) {
      for (const line of section.lines) {
        out.push({
          sectionId: section.section_id,
          lineIndex: line.line_index,
          startSec: line.start_sec,
          endSec: line.end_sec,
          text: line.text,
        });
      }
    }
    return out;
  }

  start() {
    this.state = 'listening';
    this._currentIndex = -1;
    this._backSteps = 0;
    this._recentBest = [];
    if (!this._live) {
      // Legacy stub: timer-based advancement.
      this._startTime = performance.now() / 1000;
      this._tickStub();
      return;
    }
    const lv = this._live;
    lv.chromaExtractor.reset();
    if (lv.vocalDetector) lv.vocalDetector.reset();
    lv.stepCount = 0;
    lv.recentConfidence = [];
    const nr = RATES.length;
    lv.logp = new Float64Array(nr * lv.n);
    lv.scratch = new Float64Array(nr * lv.n);
    // Start prior: the user taps at song start — mass near reference 0.
    for (let ri = 0; ri < nr; ri++) {
      for (let i = 0; i < lv.n; i++) {
        const dt = (i * STEP_SEC) / 10.0;
        lv.logp[ri * lv.n + i] = -0.5 * dt * dt;
      }
    }
  }

  stop() {
    this.state = 'idle';
    if (this._timer) { clearTimeout(this._timer); this._timer = null; }
  }

  /** Called when the user taps a line in the emergency overlay. */
  snapTo(sectionId, lineIndex) {
    const idx = this._flatLines.findIndex(
      l => l.sectionId === sectionId && l.lineIndex === lineIndex
    );
    if (idx < 0) return;
    this._currentIndex = idx;
    this._backSteps = 0;
    const line = this._flatLines[idx];
    if (this._live && this._live.logp) {
      // Concentrate posterior mass at the tapped line (±2 s), all rates.
      const lv = this._live;
      const target = Math.floor((line.startSec || 0) / STEP_SEC);
      for (let ri = 0; ri < RATES.length; ri++) {
        for (let i = 0; i < lv.n; i++) {
          const dt = ((i - target) * STEP_SEC) / 2.0;
          lv.logp[ri * lv.n + i] = -0.5 * dt * dt;
        }
      }
    } else {
      this._startTime = (performance.now() / 1000) - (line.startSec || 0);
    }
    if (this.onPositionChange) {
      this.onPositionChange({ sectionId, lineIndex, confidence: 1.0 });
    }
    this.state = 'locked';
  }

  /** Called by the AudioEngine for each frame. */
  consume(audioFrame) {
    if (this.state === 'idle' || !this._live) return;
    this._live.chromaExtractor.consume(audioFrame);
    if (this._live.vocalDetector) this._live.vocalDetector.consume(audioFrame);
  }

  // -- HMM step (runs at 2 Hz, driven by the chroma extractor) ----------

  _step(chromaVec) {
    const lv = this._live;
    if (!lv.logp) return;
    const n = lv.n, nr = RATES.length;
    const logp = lv.logp, next = lv.scratch;
    next.fill(-1e9);

    // Transition: banded advance, cost by SOURCE state's elasticity+rate.
    for (let ri = 0; ri < nr; ri++) {
      const src = ri * n, dst = ri * n;
      for (let i = 0; i < n; i++) {
        const base = logp[src + i];
        if (base < -1e8) continue;
        const costRow = lv.cost[lv.elastic[i]][ri];
        const dMax = Math.min(MAX_D, n - 1 - i);
        for (let d = 0; d <= dMax; d++) {
          const cand = base + costRow[d];
          if (cand > next[dst + i + d]) next[dst + i + d] = cand;
        }
      }
      // hold at the end state
      const endIdx = dst + n - 1;
      if (logp[endIdx] - 0.1 > next[endIdx]) next[endIdx] = logp[endIdx] - 0.1;
    }
    // Rate switching (position preserved).
    for (let ri = 0; ri < nr; ri++) {
      for (let i = 0; i < n; i++) {
        const here = ri * n + i;
        if (ri > 0) {
          const c = next[(ri - 1) * n + i] - RATE_SWITCH_COST;
          if (c > next[here]) next[here] = c;
        }
        if (ri < nr - 1) {
          const c = next[(ri + 1) * n + i] - RATE_SWITCH_COST;
          if (c > next[here]) next[here] = c;
        }
      }
    }

    // Observation: chroma cosine + gated vocal consistency.
    const { smoothed: p, gate: g } = lv.lastVocal;
    let best = -Infinity;
    for (let i = 0; i < n; i++) {
      const r = lv.ref[i];
      let sim = 0;
      for (let k = 0; k < 12; k++) sim += r[k] * chromaVec[k];
      const match = lv.vocal[i] ? p : 1 - p;
      const obs = KAPPA * sim + g * VOCAL_KAPPA * (match - 0.5);
      for (let ri = 0; ri < nr; ri++) {
        const v = next[ri * n + i] + obs;
        logp[ri * n + i] = v;
        if (v > best) best = v;
      }
    }
    for (let i = 0; i < logp.length; i++) logp[i] -= best;
    lv.stepCount += 1;

    this._emitPosition();
  }

  _emitPosition() {
    const lv = this._live;
    const n = lv.n, nr = RATES.length;
    // Max over rates per position; softmax mass near the argmax.
    let bestI = 0, bestV = -Infinity;
    const posMax = new Float64Array(n).fill(-Infinity);
    for (let ri = 0; ri < nr; ri++) {
      for (let i = 0; i < n; i++) {
        const v = lv.logp[ri * n + i];
        if (v > posMax[i]) posMax[i] = v;
      }
    }
    for (let i = 0; i < n; i++) {
      if (posMax[i] > bestV) { bestV = posMax[i]; bestI = i; }
    }
    let massNear = 0, massTotal = 0;
    const win = Math.round(CONFIDENCE_WINDOW_SEC / STEP_SEC);
    for (let i = 0; i < n; i++) {
      const w = Math.exp(posMax[i]);
      massTotal += w;
      if (Math.abs(i - bestI) <= win) massNear += w;
    }
    const confidence = massTotal > 0 ? massNear / massTotal : 0;
    this.state = confidence >= LOCK_CONFIDENCE ? 'locked'
      : confidence <= LOST_CONFIDENCE ? 'lost' : 'listening';

    // Median-of-recent argmax for a stable display position.
    if (!this._recentBest) this._recentBest = [];
    this._recentBest.push(bestI);
    if (this._recentBest.length > EMIT_MEDIAN_STEPS) this._recentBest.shift();
    const sorted = [...this._recentBest].sort((a, b) => a - b);
    const medianI = sorted[Math.floor(sorted.length / 2)];
    const refTime = medianI * STEP_SEC;
    // Current line: containing line, else the next upcoming one (so the
    // display shows what's coming during instrumental sections).
    let idx = this._flatLines.findIndex(
      l => l.startSec != null && refTime >= l.startSec && refTime < l.endSec);
    if (idx < 0) {
      idx = this._flatLines.findIndex(
        l => l.startSec != null && l.startSec >= refTime);
      if (idx < 0) idx = this._flatLines.length - 1;
    }
    if (idx < this._currentIndex) {
      // Small backward corrections (a few lines, race-and-correct within
      // a verse) are never shown: the display renders lookahead anyway,
      // so briefly being a line ahead is harmless while a backward hop is
      // visibly disruptive. Forward motion resumes on its own. Large
      // backward moves mean genuine relocalization and pass after
      // sustained evidence.
      if (this._currentIndex - idx <= BACKWARD_SUPPRESS_LINES) return;
      this._backSteps = (this._backSteps || 0) + 1;
      if (this._backSteps < BACKWARD_HOLD_STEPS) return;
    } else {
      this._backSteps = 0;
    }
    if (idx !== this._currentIndex) {
      this._backSteps = 0;
      this._currentIndex = idx;
      const l = this._flatLines[idx];
      if (l && this.onPositionChange) {
        this.onPositionChange({
          sectionId: l.sectionId,
          lineIndex: l.lineIndex,
          confidence,
        });
      }
    }
  }

  // -- Legacy timer stub (songs without a chroma resource) ---------------

  _tickStub() {
    if (this.state !== 'listening' && this.state !== 'locked') return;
    const elapsed = (performance.now() / 1000) - this._startTime;
    let newIdx = this._currentIndex;
    for (let i = 0; i < this._flatLines.length; i++) {
      const l = this._flatLines[i];
      if (l.startSec == null) continue;
      if (elapsed >= l.startSec && (i === this._flatLines.length - 1
          || elapsed < this._flatLines[i + 1].startSec)) {
        newIdx = i;
        break;
      }
    }
    if (newIdx !== this._currentIndex) {
      this._currentIndex = newIdx;
      const l = this._flatLines[newIdx];
      if (l && this.onPositionChange) {
        this.onPositionChange({
          sectionId: l.sectionId, lineIndex: l.lineIndex, confidence: 0.5,
        });
      }
    }
    if (elapsed >= this._referenceDuration) { this.stop(); return; }
    this._timer = setTimeout(() => this._tickStub(), 250);
  }
}
