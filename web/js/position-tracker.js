/**
 * position-tracker.js — online position tracking against a song template.
 *
 * This is the BRAIN. It maintains a probability distribution over
 * "where in the song are we right now" and emits a `onPositionChange`
 * callback whenever the most likely position crosses a line boundary.
 *
 * STATUS: SCAFFOLD ONLY. The current implementation is a deliberately
 * simple time-based playback (advance one line per N seconds) so the
 * UI works end-to-end without the ML pieces wired up yet. Phase 3
 * replaces this with the real engine:
 *
 *   1. Chroma-based online DTW against template.audio_features.chroma_*
 *      and per-line chroma_signature vectors.
 *   2. Whisper-tiny.en (running in transformers.js) emitting word-level
 *      ASR results that fuzzy-match against expected upcoming lyric text.
 *   3. HMM/particle filter fusing (1) + (2) + section elasticity flags
 *      into a posterior over (section, line_index).
 *
 * For now: a stub that advances through lines based on each line's
 * relative duration in the template, scaled into wall-clock time.
 * This lets you visually verify the UI scrolls correctly before any
 * ML is hooked up.
 */

export class PositionTracker {
  constructor(template) {
    this.template = template;
    this.state = 'idle';           // idle | listening | locked | lost
    this.onPositionChange = null;
    this._timer = null;
    this._flatLines = this._flattenLines(template);
    this._currentIndex = -1;
    // Total reference audio duration -> used to scale stub playback.
    this._referenceDuration = template.audio_features?.duration_sec ?? 60;
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
    this._startTime = performance.now() / 1000;
    this._currentIndex = -1;
    this._tick();
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
    const elapsedAtThisLine = this._flatLines[idx].startSec || 0;
    this._startTime = (performance.now() / 1000) - elapsedAtThisLine;
    if (this.onPositionChange) {
      this.onPositionChange({
        sectionId, lineIndex, confidence: 1.0,
      });
    }
    this.state = 'locked';
  }

  /** Called by the AudioEngine each time a new audio frame is available.
   *  STUB: ignores the audio for now -- timer-based advancement only.
   *  Phase 3 wiring: feature extraction + DTW step here. */
  // eslint-disable-next-line no-unused-vars
  consume(audioFrame) { /* phase 3 */ }

  _tick() {
    if (this.state !== 'listening' && this.state !== 'locked') return;

    const elapsed = (performance.now() / 1000) - this._startTime;

    // Find which template line we *should* be on, by reference timestamps.
    let newIdx = this._currentIndex;
    for (let i = 0; i < this._flatLines.length; i++) {
      const l = this._flatLines[i];
      if (l.startSec == null) continue;
      if (elapsed >= l.startSec && (i === this._flatLines.length - 1 || elapsed < this._flatLines[i + 1].startSec)) {
        newIdx = i;
        break;
      }
    }

    if (newIdx !== this._currentIndex) {
      this._currentIndex = newIdx;
      const l = this._flatLines[newIdx];
      if (l && this.onPositionChange) {
        this.onPositionChange({
          sectionId: l.sectionId,
          lineIndex: l.lineIndex,
          confidence: 0.5, // stub: not based on anything yet
        });
      }
    }

    // Stop at end of reference duration
    if (elapsed >= this._referenceDuration) {
      this.stop();
      return;
    }
    this._timer = setTimeout(() => this._tick(), 250);
  }
}
