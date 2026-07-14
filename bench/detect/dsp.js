/**
 * Pure DSP + detection helpers for the in-browser separation+onset page.
 *
 * Kept free of any DOM/onnxruntime references so the exact same code can
 * be parity-tested in Node against the offline pipeline:
 *  - STFT must EXACTLY match what the exported umxhq core was trained on
 *    (openunmix Separator): torch.stft with n_fft=4096, hop=1024,
 *    periodic Hann window, center=True, pad_mode="reflect",
 *    normalized=False, onesided -> magnitude (ComplexNorm power 1).
 *  - Envelope + onset detection mirror tooling/raw_onset_probe.py /
 *    detect_first_verse.py: 100ms power frames -> dB -> 1.0s moving
 *    average (edge-trimmed), baseline median of the first 8s after the
 *    song offset, threshold = baseline + 20 dB, sustained rising-edge
 *    crossings (2s window median within 5 dB of threshold).
 *    One deliberate deviation: the offline [-50,-10] dBFS threshold clamp
 *    is skipped — spectrogram-band dB has an arbitrary scale offset, so
 *    absolute dBFS bounds don't transfer (everything else is
 *    baseline-relative and transfers as-is).
 */

export const N_FFT = 4096;
export const HOP = 1024;
export const NB_BINS = N_FFT / 2 + 1; // 2049

// ---------------------------------------------------------------------------
// FFT: iterative radix-2, precomputed tables, complex in-place.
// ---------------------------------------------------------------------------

export class FFT {
  constructor(n) {
    if ((n & (n - 1)) !== 0) throw new Error(`FFT size ${n} not a power of 2`);
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r = (r << 1) | ((i >> b) & 1);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
  }

  /** In-place forward FFT on interleaved-free (re[], im[]) arrays. */
  transform(re, im) {
    const { n, rev, cos, sin } = this;
    for (let i = 0; i < n; i++) {
      const r = rev[i];
      if (r > i) {
        let t = re[i]; re[i] = re[r]; re[r] = t;
        t = im[i]; im[i] = im[r]; im[r] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const step = n / len;
      for (let i = 0; i < n; i += len) {
        for (let j = 0; j < half; j++) {
          const k = j * step;
          const c = cos[k], s = sin[k];
          const a = i + j, b = a + half;
          const tr = re[b] * c - im[b] * s;
          const ti = re[b] * s + im[b] * c;
          re[b] = re[a] - tr;
          im[b] = im[a] - ti;
          re[a] += tr;
          im[a] += ti;
        }
      }
    }
  }
}

/** Periodic Hann, identical to torch.hann_window(n) (periodic=True). */
export function hannWindow(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / n));
  return w;
}

/** Reflect-padded sample lookup (torch pad_mode="reflect": no edge repeat). */
function reflectAt(arr, i, n) {
  if (i < 0) i = -i;
  else if (i >= n) i = 2 * n - 2 - i;
  return arr[i];
}

/** Number of STFT frames torch.stft(center=True) produces. */
export function numFrames(nSamples) {
  return Math.floor(nSamples / HOP) + 1;
}

/**
 * Magnitude STFT of ONE frame for TWO real channels at once, using the
 * classic two-reals-in-one-complex-FFT packing: F(L + iR) splits by
 * conjugate symmetry into the two spectra. Writes |L| and |R| for bins
 * 0..NB_BINS-1 into magL/magR.
 *
 * Frame f covers samples [f*HOP - N_FFT/2, f*HOP + N_FFT/2) of the
 * reflect-padded signal — exactly torch.stft with center=True.
 */
export function stftFrameMag(fft, window, left, right, frameIdx, magL, magR,
                             scratchRe, scratchIm) {
  const n = fft.n;
  const nSamples = left.length;
  const start = frameIdx * HOP - n / 2;
  for (let i = 0; i < n; i++) {
    const idx = start + i;
    const w = window[i];
    scratchRe[i] = reflectAt(left, idx, nSamples) * w;
    scratchIm[i] = reflectAt(right, idx, nSamples) * w;
  }
  fft.transform(scratchRe, scratchIm);
  // Unpack: L[k] = (Z[k] + conj(Z[n-k]))/2 ; R[k] = (Z[k] - conj(Z[n-k]))/2i
  for (let k = 0; k < NB_BINS; k++) {
    const k2 = k === 0 ? 0 : n - k;
    const zr = scratchRe[k], zi = scratchIm[k];
    const wr = scratchRe[k2], wi = scratchIm[k2];
    const lr = 0.5 * (zr + wr), li = 0.5 * (zi - wi);
    const rr = 0.5 * (zi + wi), ri = 0.5 * (wr - zr);
    magL[k] = Math.hypot(lr, li);
    magR[k] = Math.hypot(rr, ri);
  }
}

// ---------------------------------------------------------------------------
// Envelope (100ms power frames -> dB -> 1.0s smoothing, edge-trimmed)
// ---------------------------------------------------------------------------

export const ENV_FRAME_SEC = 0.1;
export const SMOOTH_SEC = 1.0;

/**
 * Aggregate per-STFT-frame band power (frame f is centered at f*HOP/sr)
 * into ENV_FRAME_SEC bins, convert to dB. 10*log10(power) is the same dB
 * value as the offline detector's 20*log10(rms), so the baseline-relative
 * +20 dB threshold behaves identically.
 */
export function powerToEnvelopeDb(framePower, sampleRate) {
  const frameDt = HOP / sampleRate;
  const nEnv = Math.max(1, Math.floor((framePower.length * frameDt) / ENV_FRAME_SEC));
  const sums = new Float64Array(nEnv);
  const counts = new Float64Array(nEnv);
  for (let f = 0; f < framePower.length; f++) {
    const k = Math.min(nEnv - 1, Math.floor((f * frameDt) / ENV_FRAME_SEC));
    sums[k] += framePower[f];
    counts[k] += 1;
  }
  const times = new Float64Array(nEnv);
  const db = new Float64Array(nEnv);
  for (let k = 0; k < nEnv; k++) {
    const p = counts[k] > 0 ? sums[k] / counts[k] : 0;
    times[k] = k * ENV_FRAME_SEC;
    db[k] = 10 * Math.log10(Math.max(p, 1e-24));
  }
  return { times, db };
}

/** Moving average (same kernel as detect_first_verse._moving_average). */
export function movingAverage(arr, window) {
  if (window <= 1 || arr.length === 0) return Float64Array.from(arr);
  const out = new Float64Array(arr.length);
  // 'same'-mode convolution with a box kernel, zero-padded edges:
  // np.convolve(arr, ones(w)/w, 'same')[i] sums arr[i-floor(w/2) .. i+ceil(w/2)-1].
  const half = Math.floor(window / 2);
  for (let i = 0; i < arr.length; i++) {
    let s = 0;
    for (let j = 0; j < window; j++) {
      const idx = i + j - half;
      if (idx >= 0 && idx < arr.length) s += arr[idx];
    }
    out[i] = s / window;
  }
  return out;
}

/** Smooth + trim the zero-pad edge frames (raw_onset_probe's fix: dB is
 * negative, zero-padding drags edges toward 0 and fakes loud blips). */
export function smoothEnvelope(times, db) {
  const window = Math.max(1, Math.round(SMOOTH_SEC / ENV_FRAME_SEC));
  let sm = movingAverage(db, window);
  let t = times;
  const half = Math.floor(window / 2);
  if (half > 0 && db.length > 2 * half) {
    sm = sm.subarray(half, sm.length - half);
    t = t.subarray(half, t.length - half);
  }
  return { times: t, db: sm };
}

// ---------------------------------------------------------------------------
// Onset detection (mirror of raw_onset_probe._detect, minus dBFS clamp)
// ---------------------------------------------------------------------------

export const SONG_OFFSET = 20.0;
export const BASELINE_WINDOW_SEC = 8.0;
export const AUTO_OFFSET_DB = 20.0;
// Second gate: vocal-to-mix RATIO envelope offset. Tuned on the real band
// recording (parameter sweep, 2026-07-14): energy-only detection at any
// threshold leaves 5-14 false onsets inside the jam (Open-Unmix leak tracks
// band loudness), but requiring BOTH the absolute vocal-band energy gate
// (+20 dB) AND the vocal/mix ratio gate (+5 dB) to fire within
// AND_MATCH_SEC gives 0 jam false positives with verse_1 at +0.4s and
// verse_5 at +4.0s (inside the ±5s re-anchor tolerance). Stable across
// ABS +20..24 dB; RATIO +4 loses verse_5, +6 readmits a jam FP.
export const RATIO_OFFSET_DB = 5.0;
export const AND_MATCH_SEC = 1.5;
export const SUSTAIN_WINDOW_SEC = 2.0;
export const SUSTAIN_TOLERANCE_DB = 5.0;
export const VERDICT_TOLERANCE_SEC = 5.0;

function median(values) {
  if (values.length === 0) return null;
  const s = Float64Array.from(values).sort();
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

export function detectOnsets(times, db, songOffset = SONG_OFFSET,
                             offsetDb = AUTO_OFFSET_DB) {
  const baseVals = [];
  for (let i = 0; i < times.length; i++) {
    if (times[i] >= songOffset && times[i] < songOffset + BASELINE_WINDOW_SEC) {
      baseVals.push(db[i]);
    }
  }
  const baselineDb = median(baseVals);
  const thresholdDb = baselineDb === null ? Infinity : baselineDb + offsetDb;

  const dt = times.length > 1 ? times[1] - times[0] : ENV_FRAME_SEC;
  const windowFrames = Math.max(1, Math.ceil(SUSTAIN_WINDOW_SEC / dt));
  const medianFloor = thresholdDb - SUSTAIN_TOLERANCE_DB;
  const n = times.length;

  const onsets = [];
  const intervals = [];
  let intervalStart = null;
  let prevBelow = true; // pre-offset counts as below (same as offline)
  for (let i = 0; i < n; i++) {
    if (times[i] < songOffset) continue;
    const above = db[i] >= thresholdDb;
    if (above && prevBelow) {
      intervalStart = times[i];
      const win = Array.from(db.subarray(i, Math.min(i + windowFrames, n)));
      const m = median(win);
      if (win.length > 0 && m !== null && m >= medianFloor) onsets.push(times[i]);
    } else if (!above && !prevBelow && intervalStart !== null) {
      intervals.push([intervalStart, times[i]]);
      intervalStart = null;
    }
    prevBelow = !above;
  }
  if (intervalStart !== null && n > 0) intervals.push([intervalStart, times[n - 1]]);

  return { baselineDb, thresholdDb, onsets, intervals };
}

/**
 * Two-feature AND gate: keep only the absolute-energy onsets that have a
 * ratio-envelope onset within `matchSec`. The gated onset keeps the
 * energy-onset timestamp (the sharper of the two edges).
 */
export function andCombineOnsets(absOnsets, ratioOnsets, matchSec = AND_MATCH_SEC) {
  return absOnsets.filter((a) =>
    ratioOnsets.some((r) => Math.abs(r - a) <= matchSec));
}

// ---------------------------------------------------------------------------
// Ground-truth scoring (mirror of raw_onset_probe)
// ---------------------------------------------------------------------------

export function loadSections(gt) {
  return (gt.sections || []).filter(
    (s) => !String(s.section_id || "").startsWith("_"));
}

export function scoreSections(sections, onsets) {
  return sections.map((s) => {
    const gtStart = Number(s.start);
    let nearest = null, error = null;
    if (onsets.length) {
      nearest = onsets.reduce((a, b) =>
        Math.abs(b - gtStart) < Math.abs(a - gtStart) ? b : a);
      error = nearest - gtStart;
    }
    return { sectionId: s.section_id, gtStart, nearest, error };
  });
}

export function verdict(rows, sectionId) {
  const row = rows.find((r) => r.sectionId === sectionId);
  if (!row || row.error === null) {
    return { sectionId, detected: false, nearest: null, error: null };
  }
  return {
    sectionId,
    detected: Math.abs(row.error) <= VERDICT_TOLERANCE_SEC,
    nearest: row.nearest,
    error: row.error,
  };
}
