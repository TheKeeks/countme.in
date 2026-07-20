/**
 * vocal-onset.js — live sung-vocal presence detection on raw mic audio.
 *
 * Runs the per-song vocal head trained by tooling/train_vocal_head.py:
 * log-mel features -> z-score -> logistic -> P(vocal), with causal
 * smoothing, sustained-crossing onset detection, and a self-confidence
 * gate. No source separation, no neural nets, no dependencies — the
 * whole pipeline is FFT + a 64-filter mel bank + a dot product, well
 * under real time on a phone.
 *
 * Evidence base (see docs/VOCAL-ONSET-FUSION.md): on real band takes
 * this detects the verse re-entry after the jam within ±2s when the
 * vocal is decently present in the mix, and the gate is designed to
 * withhold the signal on hot mixes where the head is unreliable.
 *
 * MUST stay formula-identical to the Python side
 * (tooling/train_vocal_head.py): same STFT (center=false), same Slaney
 * mel filterbank as librosa's default, same running-max dB reference,
 * same pooling. The weights JSON carries a parity vector — a synthetic
 * signal's expected features — and parityCheck() verifies this
 * implementation against it at load time.
 *
 * Usage:
 *   const head = await (await fetch('templates/peggy_o_vocal_head.json')).json();
 *   const det = new VocalOnsetDetector(head);
 *   det.onOnset = (e) => tracker.vocalOnsetEvidence(e); // e.gate in [0,1]
 *   audioEngine.onAudioFrame = (f) => { det.consume(f); tracker.consume(f); };
 */

export class VocalOnsetDetector {
  constructor(head) {
    this.head = head;
    const cfg = head.feature_config;
    this.sr = cfg.sample_rate;
    this.nFft = cfg.n_fft;
    this.hop = cfg.hop;
    this.nMels = cfg.n_mels;
    this.windowSamples = Math.round(cfg.window_sec * this.sr);
    this.smoothK = Math.max(1, Math.round(cfg.smoothing_sec / cfg.window_sec));
    this.sustainWindows = Math.max(1, Math.ceil(cfg.sustain_sec / cfg.window_sec));
    this.logFloorDb = cfg.log_floor_db;

    this.onWindow = null; // ({time, prob, smoothed, gate})
    this.onOnset = null;  // ({time, gate, smoothed})

    this._hann = VocalOnsetDetector._hannPeriodic(this.nFft);
    this._melBank = VocalOnsetDetector._slaneyMelBank(
      this.sr, this.nFft, this.nMels, cfg.fmin, cfg.fmax);
    this._fftRe = new Float64Array(this.nFft);
    this._fftIm = new Float64Array(this.nFft);

    this.reset();
  }

  reset() {
    this._buf = new Float32Array(0);
    this._framePos = 0;       // absolute sample index of next frame start
    this._consumed = 0;       // absolute samples consumed (buffer trim offset)
    this._runningRefDb = -Infinity;
    this._winIdx = 0;
    this._winSum = new Float64Array(this.nMels);
    this._winSumSq = new Float64Array(this.nMels);
    this._winCount = 0;
    this._probs = [];
    this._smoothed = [];
    this._sortedSmoothed = [];
    this._aboveRun = 0;
    this._runStartTime = 0;
    this._onsetFiredThisRun = false;
    this._resamplePhase = 0;
    this._resampleLast = 0;
  }

  /** Feed an AudioEngine frame: {samples: Float32Array, sampleRate}. */
  consume(frame) {
    let samples = frame.samples;
    if (frame.sampleRate !== this.sr) {
      samples = this._resample(samples, frame.sampleRate);
    }
    // Append to buffer.
    const merged = new Float32Array(this._buf.length + samples.length);
    merged.set(this._buf);
    merged.set(samples, this._buf.length);
    this._buf = merged;

    while (this._framePos + this.nFft <= this._consumed + this._buf.length) {
      const off = this._framePos - this._consumed;
      this._processFrame(this._buf.subarray(off, off + this.nFft));
      this._framePos += this.hop;
    }
    // Trim consumed samples, keeping the overlap the next frame needs.
    const keepFrom = this._framePos - this._consumed;
    if (keepFrom > 4 * this.nFft) {
      this._buf = this._buf.slice(keepFrom);
      this._consumed += keepFrom;
    }
  }

  // -- internals ----------------------------------------------------------

  _processFrame(frame) {
    // Power spectrum of the Hann-windowed frame.
    const n = this.nFft;
    const re = this._fftRe, im = this._fftIm;
    for (let i = 0; i < n; i++) { re[i] = frame[i] * this._hann[i]; im[i] = 0; }
    VocalOnsetDetector._fft(re, im);
    const nBins = n / 2 + 1;

    // Mel energies -> dB with running-max reference (causal ref=np.max).
    let frameMaxDb = -Infinity;
    const melDb = new Float64Array(this.nMels);
    const bank = this._melBank;
    for (let m = 0; m < this.nMels; m++) {
      const row = bank[m];
      let acc = 0;
      for (let k = row.lo; k < row.hi; k++) {
        const p = re[k] * re[k] + im[k] * im[k];
        acc += row.w[k - row.lo] * p;
      }
      const db = 10 * Math.log10(Math.max(acc, 1e-10));
      melDb[m] = db;
      if (db > frameMaxDb) frameMaxDb = db;
    }
    if (frameMaxDb > this._runningRefDb) this._runningRefDb = frameMaxDb;

    // Pool into the current 0.5s window (window index by frame START time,
    // matching the Python trainer's floor(frame_time / window_sec)).
    const frameStart = this._framePos; // samples
    const idx = Math.floor(frameStart / this.windowSamples);
    if (idx > this._winIdx && this._winCount > 0) {
      this._finalizeWindow();
      this._winIdx = idx;
    } else if (this._winCount === 0) {
      this._winIdx = idx;
    }
    for (let m = 0; m < this.nMels; m++) {
      const v = Math.max(melDb[m] - this._runningRefDb, this.logFloorDb);
      this._winSum[m] += v;
      this._winSumSq[m] += v * v;
    }
    this._winCount += 1;
  }

  _finalizeWindow() {
    const c = this._winCount;
    const nM = this.nMels;
    const feat = new Float64Array(2 * nM);
    for (let m = 0; m < nM; m++) {
      const mean = this._winSum[m] / c;
      const varr = Math.max(this._winSumSq[m] / c - mean * mean, 0);
      feat[m] = mean;
      feat[nM + m] = Math.sqrt(varr);
    }
    this._winSum.fill(0); this._winSumSq.fill(0); this._winCount = 0;

    // z-score + logistic.
    const h = this.head;
    let logit = h.intercept;
    for (let i = 0; i < feat.length; i++) {
      logit += h.coef[i] * ((feat[i] - h.scaler_mean[i]) / h.scaler_scale[i]);
    }
    const prob = 1 / (1 + Math.exp(-logit));
    const time = this._winIdx * this.head.feature_config.window_sec;

    this._probs.push(prob);
    const from = Math.max(0, this._probs.length - this.smoothK);
    let s = 0;
    for (let i = from; i < this._probs.length; i++) s += this._probs[i];
    const smoothed = s / (this._probs.length - from);
    this._smoothed.push(smoothed);
    this._insertSorted(smoothed);
    const gate = this._gate();

    // Sustained-crossing onset: fire once per above-threshold run, as soon
    // as the run reaches sustain length. Onset time = run start.
    const thr = this.head.feature_config.threshold;
    if (smoothed >= thr) {
      if (this._aboveRun === 0) this._runStartTime = time;
      this._aboveRun += 1;
      if (this._aboveRun >= this.sustainWindows && !this._onsetFiredThisRun) {
        this._onsetFiredThisRun = true;
        if (this.onOnset) {
          this.onOnset({ time: this._runStartTime, gate, smoothed });
        }
      }
    } else {
      this._aboveRun = 0;
      this._onsetFiredThisRun = false;
    }
    if (this.onWindow) this.onWindow({ time, prob, smoothed, gate });
  }

  /** Self-confidence gate in [0,1]: q90-q10 spread of the smoothed curve
   *  over an EXPANDING window since start. A crisply bimodal curve
   *  (vocals clearly separate from the band) spreads wide; a mushy hot
   *  mix stays narrow and the tracker should ignore vocal onsets. */
  _gate() {
    const g = this.head.gate;
    const elapsed = this._smoothed.length
      * this.head.feature_config.window_sec;
    if (elapsed < g.min_sec) return 0;
    const q = (arr, f) => arr[Math.min(arr.length - 1,
      Math.max(0, Math.floor(f * (arr.length - 1))))];
    const spread = q(this._sortedSmoothed, 0.9) - q(this._sortedSmoothed, 0.1);
    return Math.min(1, Math.max(0, (spread - g.low) / (g.high - g.low)));
  }

  _insertSorted(v) {
    const a = this._sortedSmoothed;
    let lo = 0, hi = a.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    a.splice(lo, 0, v);
  }

  _resample(samples, fromRate) {
    // Linear resampling with phase carried across chunks. AudioEngine
    // requests a 16kHz context, but some browsers ignore the hint.
    const ratio = fromRate / this.sr;
    const out = [];
    let pos = this._resamplePhase;
    while (pos < samples.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = i === 0 && this._resamplePhase < 1 && pos < 1
        ? this._resampleLast : samples[Math.max(0, i)];
      const b = samples[Math.min(samples.length - 1, i + 1)];
      out.push(a * (1 - frac) + b * frac);
      pos += ratio;
    }
    this._resamplePhase = pos - samples.length;
    this._resampleLast = samples[samples.length - 1];
    return Float32Array.from(out);
  }

  /** Verify this implementation against the Python-computed parity
   *  vector in the head file. Returns {ok, maxDiffDb}. Call at load. */
  parityCheck() {
    const cfg = this.head.feature_config;
    const sr = cfg.sample_rate;
    const y = new Float32Array(2 * sr);
    for (let i = 0; i < y.length; i++) {
      const t = i / sr;
      y[i] = 0.5 * Math.sin(2 * Math.PI * 440 * t)
        + 0.3 * Math.sin(2 * Math.PI * 1000 * t)
        + 0.2 * Math.sin(2 * Math.PI * 3000 * t);
    }
    // Absolute-dB feature path (no running ref / floor), isolating
    // FFT + filterbank parity — mirrors _parity_vector() in the trainer.
    const expected = this.head.parity.expected_windows;
    const nM = this.nMels;
    const sums = [], sumsqs = [], counts = [];
    for (let w = 0; w < expected.length; w++) {
      sums.push(new Float64Array(nM));
      sumsqs.push(new Float64Array(nM));
      counts.push(0);
    }
    const re = new Float64Array(this.nFft), im = new Float64Array(this.nFft);
    for (let fp = 0; fp + this.nFft <= y.length; fp += this.hop) {
      const w = Math.floor(fp / this.windowSamples);
      if (w >= expected.length) break;
      for (let i = 0; i < this.nFft; i++) {
        re[i] = y[fp + i] * this._hann[i]; im[i] = 0;
      }
      VocalOnsetDetector._fft(re, im);
      for (let m = 0; m < nM; m++) {
        const row = this._melBank[m];
        let acc = 0;
        for (let k = row.lo; k < row.hi; k++) {
          acc += row.w[k - row.lo] * (re[k] * re[k] + im[k] * im[k]);
        }
        const db = 10 * Math.log10(Math.max(acc, 1e-10));
        sums[w][m] += db;
        sumsqs[w][m] += db * db;
      }
      counts[w] += 1;
    }
    let maxDiff = 0;
    for (let w = 0; w < expected.length; w++) {
      for (let m = 0; m < nM; m++) {
        const mean = sums[w][m] / counts[w];
        const varr = Math.max(sumsqs[w][m] / counts[w] - mean * mean, 0);
        const std = Math.sqrt(varr);
        maxDiff = Math.max(maxDiff,
          Math.abs(mean - expected[w][m]),
          Math.abs(std - expected[w][nM + m]));
      }
    }
    return { ok: maxDiff <= this.head.parity.tolerance_db, maxDiffDb: maxDiff };
  }

  // -- static DSP helpers -------------------------------------------------

  static _hannPeriodic(n) {
    const w = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
    }
    return w;
  }

  /** Slaney-scale mel filterbank identical to librosa.filters.mel
   *  defaults (htk=false, norm='slaney'). Rows are stored sparse:
   *  {lo, hi, w} covering only the nonzero bin range. */
  static _slaneyMelBank(sr, nFft, nMels, fmin, fmax) {
    const fSp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fSp;
    const logStep = Math.log(6.4) / 27;
    const hzToMel = (f) => f < minLogHz
      ? f / fSp : minLogMel + Math.log(f / minLogHz) / logStep;
    const melToHz = (m) => m < minLogMel
      ? m * fSp : minLogHz * Math.exp(logStep * (m - minLogMel));

    const melMin = hzToMel(fmin), melMax = hzToMel(fmax);
    const melPts = [];
    for (let i = 0; i < nMels + 2; i++) {
      melPts.push(melToHz(melMin + ((melMax - melMin) * i) / (nMels + 1)));
    }
    const nBins = Math.floor(nFft / 2) + 1;
    const fftFreqs = [];
    for (let k = 0; k < nBins; k++) fftFreqs.push((k * sr) / nFft);

    const bank = [];
    for (let m = 0; m < nMels; m++) {
      const [fLo, fC, fHi] = [melPts[m], melPts[m + 1], melPts[m + 2]];
      const enorm = 2 / (fHi - fLo);
      let lo = nBins, hi = 0;
      const weights = [];
      for (let k = 0; k < nBins; k++) {
        const lower = (fftFreqs[k] - fLo) / (fC - fLo);
        const upper = (fHi - fftFreqs[k]) / (fHi - fC);
        const w = Math.max(0, Math.min(lower, upper)) * enorm;
        if (w > 0) {
          if (k < lo) lo = k;
          hi = k + 1;
          weights[k] = w;
        }
      }
      const row = { lo: Math.min(lo, hi), hi, w: new Float64Array(Math.max(0, hi - lo)) };
      for (let k = row.lo; k < row.hi; k++) row.w[k - row.lo] = weights[k] || 0;
      bank.push(row);
    }
    return bank;
  }

  /** In-place iterative radix-2 Cooley-Tukey FFT. */
  static _fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang), wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < len / 2; k++) {
          const uRe = re[i + k], uIm = im[i + k];
          const vRe = re[i + k + len / 2] * curRe - im[i + k + len / 2] * curIm;
          const vIm = re[i + k + len / 2] * curIm + im[i + k + len / 2] * curRe;
          re[i + k] = uRe + vRe; im[i + k] = uIm + vIm;
          re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm;
          const nRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nRe;
        }
      }
    }
  }
}
