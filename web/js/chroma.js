/**
 * chroma.js — streaming pitch-class-profile extraction from raw mic
 * audio, for the position tracker's chroma-vs-template observation.
 *
 * One L2-normalized 12-bin chroma vector per 0.5 s step: 4096-sample
 * Hann frame (~0.26 s of the 16 kHz stream) at the top of each step,
 * FFT power folded onto pitch classes over 65–2093 Hz with 1/f octave
 * compensation. Formula-identical to the Python prototype the tracker
 * was validated with (pcp_chroma in the Phase-3 eval; see
 * docs/POSITION-TRACKER.md).
 *
 * Dependency-free; reuses the FFT from vocal-onset.js.
 */

import { VocalOnsetDetector } from './vocal-onset.js';

const SR = 16000;
const FFT_N = 4096;
const HOP = 8000; // 0.5 s -> 2 fps, aligned with the vocal-head windows
const FMIN = 65.0;
const FMAX = 2093.0;

export class ChromaExtractor {
  constructor() {
    this.onChroma = null; // ({time, vector: Float64Array(12)})
    this._hann = VocalOnsetDetector._hannPeriodic(FFT_N);
    // Precompute per-bin pitch class + weight for the valid band.
    const nBins = FFT_N / 2 + 1;
    this._pc = new Int8Array(nBins).fill(-1);
    this._w = new Float64Array(nBins);
    for (let k = 0; k < nBins; k++) {
      const f = (k * SR) / FFT_N;
      if (f < FMIN || f > FMAX) continue;
      const midi = 69 + 12 * Math.log2(f / 440.0);
      this._pc[k] = ((Math.round(midi) % 12) + 12) % 12;
      this._w[k] = 1.0 / Math.max(f, FMIN);
    }
    this._re = new Float64Array(FFT_N);
    this._im = new Float64Array(FFT_N);
    this.reset();
  }

  reset() {
    this._buf = new Float32Array(0);
    this._pos = 0;       // absolute sample index of next frame start
    this._consumed = 0;
    this._resamplePhase = 0;
    this._resampleLast = 0;
  }

  /** Feed an AudioEngine frame: {samples, sampleRate}. */
  consume(frame) {
    let samples = frame.samples;
    if (frame.sampleRate !== SR) {
      samples = this._resampleLinear(samples, frame.sampleRate);
    }
    const merged = new Float32Array(this._buf.length + samples.length);
    merged.set(this._buf);
    merged.set(samples, this._buf.length);
    this._buf = merged;

    while (this._pos + FFT_N <= this._consumed + this._buf.length) {
      const off = this._pos - this._consumed;
      this._emit(this._buf.subarray(off, off + FFT_N));
      this._pos += HOP;
    }
    // Trim consumed samples. NOTE: hop > frame here, so _pos can point up
    // to (hop - frame) samples past the data received so far — cap the
    // trim at what the buffer actually holds.
    const drop = Math.min(this._pos - this._consumed, this._buf.length);
    if (drop > 4 * FFT_N) {
      this._buf = this._buf.slice(drop);
      this._consumed += drop;
    }
  }

  _emit(frame) {
    const re = this._re, im = this._im;
    for (let i = 0; i < FFT_N; i++) { re[i] = frame[i] * this._hann[i]; im[i] = 0; }
    VocalOnsetDetector._fft(re, im);
    const v = new Float64Array(12);
    const nBins = FFT_N / 2 + 1;
    for (let k = 0; k < nBins; k++) {
      const pc = this._pc[k];
      if (pc < 0) continue;
      v[pc] += (re[k] * re[k] + im[k] * im[k]) * this._w[k];
    }
    let norm = 0;
    for (let i = 0; i < 12; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    if (norm > 0) for (let i = 0; i < 12; i++) v[i] /= norm;
    if (this.onChroma) {
      this.onChroma({ time: this._pos / SR, vector: v });
    }
  }

  _resampleLinear(samples, fromRate) {
    const ratio = fromRate / SR;
    const out = [];
    let pos = this._resamplePhase;
    while (pos < samples.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = samples[Math.max(0, i)];
      const b = samples[Math.min(samples.length - 1, i + 1)];
      out.push(a * (1 - frac) + b * frac);
      pos += ratio;
    }
    this._resamplePhase = pos - samples.length;
    this._resampleLast = samples[samples.length - 1];
    return Float32Array.from(out);
  }
}
