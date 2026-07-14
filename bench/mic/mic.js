/**
 * Stage 1b: live-microphone separation + onset detection.
 *
 * Same chain as bench/detect/ — magnitude STFT (4096/1024 Hann, the exact
 * umxhq transform), the bench/umx/ ONNX vocals core in ~7 s chunks,
 * vocal-band 250–3000 Hz energy, baseline + offset sustained-onset
 * detector — but the audio comes from getUserMedia in a real room. The
 * real-world unknown this tests: does vocal energy survive band bleed on
 * the actual iOS mic.
 *
 * The DSP/detection module is imported from bench/detect/dsp.js (pure,
 * parity-tested against torch/numpy). Detection is the two-feature AND
 * gate tuned on the real band recording: absolute vocal-band energy AND
 * the vocal-to-mix ratio must both fire within ±1.5s — energy alone lets
 * jam leak through (9 false onsets on the reference recording); the gate
 * zeroes them. Both offsets are adjustable inputs here. Song offset is
 * 0 — times are relative to recording start, and the first
 * BASELINE_WINDOW_SEC of the recording are the baseline (start recording
 * during instrumental playing).
 *
 * Same self-reporting as the other bench pages: every number, status
 * change, and full error text renders on the page.
 */

import {
  FFT, N_FFT, HOP, NB_BINS, hannWindow, stftFrameMag, numFrames,
  powerToEnvelopeDb, smoothEnvelope, detectOnsets, andCombineOnsets,
  BASELINE_WINDOW_SEC, SUSTAIN_WINDOW_SEC, SUSTAIN_TOLERANCE_DB,
  RATIO_OFFSET_DB, AND_MATCH_SEC,
} from "../detect/dsp.js";

// Same onnxruntime-web version as the other bench pages.
const ORT_VERSION = "1.23.2";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_URL = `${ORT_DIST}ort.webgpu.min.mjs`;
const MODEL_URL = "../umx/umx_vocals.onnx"; // reused as-is from bench/umx/

const SR = 44100;               // model rate; mic is resampled to this
const CHUNK_FRAMES = 302;       // the ONNX export's fixed nb_frames
const INPUT_NAME = "mag_spec";
const BAND_LO_HZ = 250, BAND_HI_HZ = 3000;
const BIN_HZ = SR / N_FFT;
const BAND_LO = Math.round(BAND_LO_HZ / BIN_HZ);
const BAND_HI = Math.round(BAND_HI_HZ / BIN_HZ);
const MAX_RECORD_SEC = 45;
const SONG_OFFSET = 0; // recording starts at the user's gesture

const $ = (id) => document.getElementById(id);
const recordBtn = $("record");
const playBtn = $("play");
const reanalyzeBtn = $("reanalyze");

// ---------------------------------------------------------------------------
// On-page reporting (the only "console" an iPhone user has)
// ---------------------------------------------------------------------------

function logLine(msg) {
  const el = $("log");
  const t = (performance.now() / 1000).toFixed(1).padStart(6);
  el.textContent += `[${t}s] ${msg}\n`;
  el.scrollTop = el.scrollHeight;
}

function setStatus(msg) {
  $("status").textContent = msg;
  if (msg) logLine(`STATUS: ${msg}`);
}

function setStatusQuiet(msg) {
  $("status").textContent = msg;
}

function showError(context, err) {
  const box = $("errors");
  box.style.display = "block";
  const pre = document.createElement("pre");
  const detail =
    err instanceof Error
      ? `${err.name}: ${err.message}\n${err.stack || ""}`
      : String(err);
  pre.textContent = `[${context}]\n${detail}`;
  $("error-list").appendChild(pre);
  logLine(`ERROR in ${context}: ${detail.split("\n")[0]}`);
}

window.addEventListener("error", (e) =>
  showError("window.onerror", e.error || e.message));
window.addEventListener("unhandledrejection", (e) =>
  showError("unhandledrejection", e.reason));

const fmtSec = (s) => `${s.toFixed(2)} s`;
const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  ort: null,
  session: null,
  sessionPromise: null,
  ep: null,
  busy: false,
  recording: false,
  stopRequested: false,
  captured: null,   // { samples: Float32Array (mono, capture rate), rate }
  resampled: null,     // Float32Array @ 44100
  envelope: null,      // { times, db } — absolute vocal-band energy
  envelopeRatio: null, // { times, db } — vocal-to-mix ratio (second gate)
  timings: null,
  playbackCtx: null,
  playbackSource: null,
};

// ---------------------------------------------------------------------------
// Model session (kicked off in the background while recording runs)
// ---------------------------------------------------------------------------

async function loadOrt() {
  let ortMod = await import(ORT_URL);
  if (!ortMod.InferenceSession && ortMod.default) ortMod = ortMod.default;
  if (!ortMod.InferenceSession) {
    throw new Error("onnxruntime-web module loaded but InferenceSession missing");
  }
  ortMod.env.wasm.wasmPaths = ORT_DIST;
  logLine(`onnxruntime-web@${ORT_VERSION} loaded`);
  return ortMod;
}

async function ensureSession() {
  if (state.session) return;
  state.ort = await loadOrt();

  let adapter = null;
  if ("gpu" in navigator) {
    try { adapter = await navigator.gpu.requestAdapter(); } catch { /* no-op */ }
  }
  logLine(adapter ? "WebGPU adapter available" : "no WebGPU adapter; using WASM");

  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status} for ${MODEL_URL}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  logLine(`model downloaded: ${(bytes.byteLength / 1048576).toFixed(1)} MB`);

  const attempts = adapter ? ["webgpu", "wasm"] : ["wasm"];
  let lastErr = null;
  for (const ep of attempts) {
    try {
      state.session = await state.ort.InferenceSession.create(bytes.slice(), {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      state.ep = ep;
      logLine(`session created on ${ep}`);
      return;
    } catch (err) {
      lastErr = err;
      showError(`session creation on ${ep}`, err);
    }
  }
  throw new Error(`all execution providers failed; last error: ${lastErr}`);
}

// ---------------------------------------------------------------------------
// Mic capture (AudioWorklet preferred; ScriptProcessor fallback)
// ---------------------------------------------------------------------------

const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(4096);
    this.fill = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      let i = 0;
      while (i < ch.length) {
        const n = Math.min(ch.length - i, this.buf.length - this.fill);
        this.buf.set(ch.subarray(i, i + n), this.fill);
        this.fill += n; i += n;
        if (this.fill === this.buf.length) {
          this.port.postMessage(this.buf.slice());
          this.fill = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("capture", CaptureProcessor);
`;

function showMicSettings(track) {
  const s = track.getSettings ? track.getSettings() : {};
  const put = (id, key, wantOff) => {
    const el = $(id);
    const v = s[key];
    el.textContent = v === undefined ? "(not reported)" : String(v);
    // Flag processing left ON despite our "off" request — the iOS suspicion.
    el.className = "num" + (wantOff && v === true ? " warn" : "");
  };
  put("s-ec", "echoCancellation", true);
  put("s-ns", "noiseSuppression", true);
  put("s-agc", "autoGainControl", true);
  $("s-sr").textContent = s.sampleRate !== undefined ? `${s.sampleRate} Hz` : "(not reported)";
  $("s-ch").textContent = s.channelCount !== undefined ? String(s.channelCount) : "(not reported)";
  $("mic-settings").style.display = "table";
  logLine(`mic settings: EC=${s.echoCancellation} NS=${s.noiseSuppression} ` +
          `AGC=${s.autoGainControl} rate=${s.sampleRate} ch=${s.channelCount}`);
}

function updateMeter(peak) {
  const db = 20 * Math.log10(Math.max(peak, 1e-6));
  const pct = Math.min(100, Math.max(0, (db + 60) / 60 * 100));
  const meter = $("meter");
  meter.style.width = `${pct}%`;
  meter.className = peak > 0.89 ? "hot" : "";
  $("meter-db").textContent = `level: ${db.toFixed(1)} dBFS peak (last block)`;
}

async function recordFromMic() {
  setStatus("Requesting microphone…");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        channelCount: 1,
      },
    });
  } catch (err) {
    if (err && (err.name === "NotAllowedError" || err.name === "SecurityError")) {
      throw new Error(
        "Microphone permission denied. Allow mic access for this site " +
        "(iOS: AA menu → Website Settings → Microphone: Allow, or " +
        "Settings → Safari → Microphone) and tap Record again.");
    }
    throw err;
  }

  const track = stream.getAudioTracks()[0];
  showMicSettings(track);

  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx(); // native rate for glitch-free capture; resample later
  if (ctx.state === "suspended") await ctx.resume();
  const rate = ctx.sampleRate;
  const source = ctx.createMediaStreamSource(stream);
  const sink = ctx.createGain();
  sink.gain.value = 0; // keep the graph pulled without audible monitoring
  sink.connect(ctx.destination);

  const chunks = [];
  let total = 0;
  const maxSamples = MAX_RECORD_SEC * rate;
  let done;
  const finished = new Promise((r) => { done = r; });

  const onChunk = (data) => {
    if (!state.recording) return;
    chunks.push(data);
    total += data.length;
    let peak = 0;
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]);
      if (a > peak) peak = a;
    }
    updateMeter(peak);
    setStatusQuiet(`Recording… ${(total / rate).toFixed(1)}s / ${MAX_RECORD_SEC}s (tap Stop to end early)`);
    if (total >= maxSamples || state.stopRequested) {
      state.recording = false;
      done();
    }
  };

  let node = null, scriptNode = null, captureKind;
  if (ctx.audioWorklet) {
    const blobUrl = URL.createObjectURL(
      new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
    try {
      await ctx.audioWorklet.addModule(blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
    node = new AudioWorkletNode(ctx, "capture", {
      numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
    });
    node.port.onmessage = (e) => onChunk(e.data);
    source.connect(node);
    node.connect(sink);
    captureKind = "AudioWorklet";
  } else {
    scriptNode = ctx.createScriptProcessor(4096, 1, 1);
    scriptNode.onaudioprocess = (e) => onChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(scriptNode);
    scriptNode.connect(sink);
    captureKind = "ScriptProcessor (AudioWorklet unavailable)";
  }
  logLine(`recording via ${captureKind} @ ${rate} Hz`);

  state.recording = true;
  state.stopRequested = false;
  recordBtn.textContent = "■ Stop";
  recordBtn.classList.add("recording");
  setStatus("Recording… (first ~8s = baseline; keep it instrumental)");

  await finished;

  recordBtn.classList.remove("recording");
  recordBtn.textContent = "● Record";
  try { node && node.disconnect(); } catch { /* best effort */ }
  try { scriptNode && scriptNode.disconnect(); } catch { /* best effort */ }
  try { source.disconnect(); } catch { /* best effort */ }
  stream.getTracks().forEach((t) => t.stop());
  await ctx.close().catch(() => {});

  const samples = new Float32Array(total);
  let off = 0;
  for (const c of chunks) { samples.set(c, off); off += c.length; }
  logLine(`captured ${(total / rate).toFixed(2)}s @ ${rate} Hz (${captureKind})`);
  if (total < rate * 2) {
    throw new Error(`recording too short (${(total / rate).toFixed(2)}s) — ` +
                    `record at least a few seconds`);
  }
  return { samples, rate };
}

// ---------------------------------------------------------------------------
// Resample to the model rate (OfflineAudioContext; linear fallback)
// ---------------------------------------------------------------------------

async function resampleTo44100(samples, srcRate) {
  if (srcRate === SR) return samples;
  const outLen = Math.ceil((samples.length * SR) / srcRate);
  try {
    const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    const off = new Ctx(1, outLen, SR);
    const buf = off.createBuffer(1, samples.length, srcRate);
    buf.copyToChannel(samples, 0);
    const src = off.createBufferSource();
    src.buffer = buf;
    src.connect(off.destination);
    src.start();
    const rendered = await off.startRendering();
    logLine(`resampled ${srcRate} -> ${SR} Hz via OfflineAudioContext`);
    return new Float32Array(rendered.getChannelData(0));
  } catch (err) {
    logLine(`OfflineAudioContext resample failed (${err}); using linear fallback`);
    const out = new Float32Array(outLen);
    const ratio = srcRate / SR;
    for (let i = 0; i < outLen; i++) {
      const x = i * ratio;
      const j = Math.floor(x);
      const frac = x - j;
      const a = samples[Math.min(j, samples.length - 1)];
      const b = samples[Math.min(j + 1, samples.length - 1)];
      out[i] = a + (b - a) * frac;
    }
    return out;
  }
}

// ---------------------------------------------------------------------------
// Separation chain (same loop as bench/detect/, mono duplicated to 2ch)
// ---------------------------------------------------------------------------

async function separateToBandPower(mono) {
  const totalFrames = numFrames(mono.length);
  const nChunks = Math.ceil(totalFrames / CHUNK_FRAMES);
  logLine(`STFT: ${totalFrames} frames, ${nChunks} chunks of ${CHUNK_FRAMES}`);

  const fft = new FFT(N_FFT);
  const window = hannWindow(N_FFT);
  const magL = new Float64Array(NB_BINS);
  const magR = new Float64Array(NB_BINS);
  const sre = new Float64Array(N_FFT);
  const sim = new Float64Array(N_FFT);
  const plane = NB_BINS * CHUNK_FRAMES;
  const input = new Float32Array(2 * plane);
  const vocalPower = new Float64Array(totalFrames);
  const mixPower = new Float64Array(totalFrames); // for the ratio gate

  let stftSec = 0, inferSec = 0;
  for (let c = 0; c < nChunks; c++) {
    setStatusQuiet(`Analyzing… chunk ${c + 1}/${nChunks}`);
    let t0 = performance.now();
    input.fill(0);
    const framesInChunk = Math.min(CHUNK_FRAMES, totalFrames - c * CHUNK_FRAMES);
    for (let j = 0; j < framesInChunk; j++) {
      const f = c * CHUNK_FRAMES + j;
      // mono duplicated into both model channels
      stftFrameMag(fft, window, mono, mono, f, magL, magR, sre, sim);
      let mp = 0;
      for (let k = BAND_LO; k <= BAND_HI; k++) {
        mp += magL[k] * magL[k] + magR[k] * magR[k];
      }
      mixPower[f] = mp;
      for (let k = 0; k < NB_BINS; k++) {
        input[k * CHUNK_FRAMES + j] = magL[k];
        input[plane + k * CHUNK_FRAMES + j] = magR[k];
      }
    }
    stftSec += (performance.now() - t0) / 1000;

    t0 = performance.now();
    const tensor = new state.ort.Tensor("float32", input, [1, 2, NB_BINS, CHUNK_FRAMES]);
    const out = await state.session.run({ [INPUT_NAME]: tensor });
    const est = Object.values(out)[0].data;
    inferSec += (performance.now() - t0) / 1000;

    for (let j = 0; j < framesInChunk; j++) {
      let p = 0;
      for (let ch = 0; ch < 2; ch++) {
        const base = ch * plane;
        for (let k = BAND_LO; k <= BAND_HI; k++) {
          const v = est[base + k * CHUNK_FRAMES + j];
          p += v * v;
        }
      }
      vocalPower[c * CHUNK_FRAMES + j] = p;
    }
    await yieldToUI();
  }
  return { vocalPower, mixPower, stftSec, inferSec };
}

// ---------------------------------------------------------------------------
// Plot (envelope is the key diagnostic — drawn big)
// ---------------------------------------------------------------------------

function drawPlot(env, det, duration) {
  const canvas = $("plot");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 640, cssH = canvas.clientHeight || 240;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssW, H = cssH, padL = 30, padB = 16, padT = 6;

  const tMax = Math.max(duration, 1);
  let dbMin = Infinity, dbMax = -Infinity;
  for (const v of env.db) { if (v < dbMin) dbMin = v; if (v > dbMax) dbMax = v; }
  if (det.thresholdDb !== Infinity) dbMin = Math.max(dbMin, det.thresholdDb - 40);
  const span = Math.max(1, dbMax - dbMin);
  const x = (t) => padL + ((W - padL) * t) / tMax;
  const y = (db) => padT + (H - padT - padB) * (1 - (Math.max(db, dbMin) - dbMin) / span);

  ctx.clearRect(0, 0, W, H);
  ctx.font = "9px system-ui";

  // Baseline window shading.
  ctx.fillStyle = "rgba(120,120,120,0.15)";
  ctx.fillRect(x(SONG_OFFSET), padT,
               x(Math.min(SONG_OFFSET + BASELINE_WINDOW_SEC, tMax)) - x(SONG_OFFSET),
               H - padT - padB);
  ctx.fillStyle = "rgba(120,120,120,0.9)";
  ctx.fillText("baseline", x(SONG_OFFSET) + 2, padT + 8);

  // Threshold line.
  if (det.thresholdDb !== Infinity) {
    ctx.strokeStyle = "#9333ea";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(padL, y(det.thresholdDb));
    ctx.lineTo(W, y(det.thresholdDb));
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#9333ea";
    ctx.fillText(`threshold ${det.thresholdDb.toFixed(1)} dB`,
                 padL + 2, y(det.thresholdDb) - 3);
  }

  // Envelope.
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  for (let i = 0; i < env.times.length; i++) {
    const px = x(env.times[i]), py = y(env.db[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Onsets.
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 2;
  for (const t of det.onsets) {
    ctx.beginPath();
    ctx.moveTo(x(t), padT);
    ctx.lineTo(x(t), H - padB);
    ctx.stroke();
    ctx.fillStyle = "#dc2626";
    ctx.fillText(`${t.toFixed(1)}s`, x(t) + 2, H - padB - 4);
  }

  // Time axis every 5s.
  ctx.fillStyle = "rgba(120,120,120,0.9)";
  for (let t = 0; t <= tMax; t += 5) {
    ctx.fillText(`${Math.round(t)}s`, x(t) + 1, H - 4);
  }
}

// ---------------------------------------------------------------------------
// Analysis + rendering (re-runnable with a new threshold offset)
// ---------------------------------------------------------------------------

function currentOffsetDb() {
  const v = parseFloat($("offset-db").value);
  return Number.isFinite(v) ? v : 20;
}

function currentRatioDb() {
  const v = parseFloat($("ratio-db").value);
  return Number.isFinite(v) ? v : RATIO_OFFSET_DB;
}

function analyzeEnvelope() {
  const env = state.envelope;
  const detAbs = detectOnsets(env.times, env.db, SONG_OFFSET, currentOffsetDb());
  const detRatio = detectOnsets(state.envelopeRatio.times, state.envelopeRatio.db,
                                SONG_OFFSET, currentRatioDb());
  const gated = andCombineOnsets(detAbs.onsets, detRatio.onsets);
  const det = { ...detAbs, onsets: gated, ratio: detRatio };
  logLine(`abs ${detAbs.onsets.length} · ratio ${detRatio.onsets.length} · ` +
          `AND(±${AND_MATCH_SEC}s) ${gated.length} onsets`);
  const duration = state.resampled.length / SR;

  const badge = $("provider");
  badge.style.display = "block";
  badge.className = state.ep;
  badge.textContent = state.ep === "webgpu" ? "RAN ON: WEBGPU" : "RAN ON: WASM (no GPU)";

  $("r-dur").textContent = fmtSec(duration);
  $("r-onsets").textContent =
    det.onsets.length ? det.onsets.map((t) => `${t.toFixed(2)}s`).join(", ") : "(none)";
  $("r-thresh").textContent =
    `abs ${det.baselineDb === null ? "-" : det.baselineDb.toFixed(1)} / ` +
    `${det.thresholdDb === Infinity ? "-" : det.thresholdDb.toFixed(1)} dB ` +
    `(+${currentOffsetDb().toFixed(0)}) · ratio ` +
    `${det.ratio.baselineDb === null ? "-" : det.ratio.baselineDb.toFixed(1)} / ` +
    `${det.ratio.thresholdDb === Infinity ? "-" : det.ratio.thresholdDb.toFixed(1)} dB ` +
    `(+${currentRatioDb().toFixed(0)})`;
  const t = state.timings;
  $("r-total").textContent = fmtSec(t.stftSec + t.inferSec + t.detectSec);
  $("r-rtf").textContent = `${((t.stftSec + t.inferSec + t.detectSec) / duration).toFixed(3)}×`;
  $("env-info").textContent =
    `Detector: AND gate = energy (+${currentOffsetDb().toFixed(0)} dB) ∧ vocal/mix ` +
    `ratio (+${currentRatioDb().toFixed(0)} dB) within ±${AND_MATCH_SEC}s · baseline ` +
    `median of first ${BASELINE_WINDOW_SEC.toFixed(0)}s of the recording · 1.0s ` +
    `smoothing · ${SUSTAIN_WINDOW_SEC.toFixed(0)}s sustain within ` +
    `${SUSTAIN_TOLERANCE_DB.toFixed(0)} dB · vocal band ${BAND_LO_HZ}–${BAND_HI_HZ} Hz · ` +
    `onnxruntime-web@${ORT_VERSION}`;
  $("results").style.display = "block";

  drawPlot(state.envelope, det, duration);
  logLine(`detection: baseline ${det.baselineDb === null ? "-" : det.baselineDb.toFixed(1)} dB, ` +
          `threshold ${det.thresholdDb === Infinity ? "-" : det.thresholdDb.toFixed(1)} dB, ` +
          `${det.onsets.length} onset(s)`);
}

// ---------------------------------------------------------------------------
// Playback
// ---------------------------------------------------------------------------

function stopPlayback() {
  if (state.playbackSource) {
    try { state.playbackSource.stop(); } catch { /* already stopped */ }
    state.playbackSource = null;
  }
  playBtn.textContent = "▶ Play recording";
}

async function togglePlayback() {
  if (state.playbackSource) { stopPlayback(); return; }
  const { samples, rate } = state.captured;
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!state.playbackCtx) state.playbackCtx = new Ctx({ sampleRate: rate });
  const ctx = state.playbackCtx;
  if (ctx.state === "suspended") await ctx.resume();
  const buf = ctx.createBuffer(1, samples.length, rate);
  buf.copyToChannel(samples, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.onended = () => { if (state.playbackSource === src) stopPlayback(); };
  src.start();
  state.playbackSource = src;
  playBtn.textContent = "■ Stop playback";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function recordAndAnalyze() {
  if (state.busy) return;
  state.busy = true;
  playBtn.disabled = true;
  reanalyzeBtn.disabled = true;
  stopPlayback();
  try {
    // Load the model in the background while we record.
    if (!state.sessionPromise) state.sessionPromise = ensureSession();
    state.sessionPromise.catch(() => {}); // surfaced when awaited below

    state.captured = await recordFromMic();
    recordBtn.disabled = true; // recording over; inert until analysis ends

    setStatus("Resampling to 44.1 kHz…");
    state.resampled = await resampleTo44100(state.captured.samples, state.captured.rate);

    setStatus("Waiting for model…");
    await state.sessionPromise;

    const t0 = performance.now();
    const sep = await separateToBandPower(state.resampled);

    setStatus("Building envelopes + detecting onsets…");
    const tDet = performance.now();
    const rawAbs = powerToEnvelopeDb(sep.vocalPower, SR);
    state.envelope = smoothEnvelope(rawAbs.times, rawAbs.db);
    const ratio = new Float64Array(sep.vocalPower.length);
    for (let i = 0; i < ratio.length; i++) {
      ratio[i] = Math.max(sep.vocalPower[i], 1e-24) /
                 Math.max(sep.mixPower[i], 1e-24);
    }
    const rawRatio = powerToEnvelopeDb(ratio, SR);
    state.envelopeRatio = smoothEnvelope(rawRatio.times, rawRatio.db);
    state.timings = {
      stftSec: sep.stftSec,
      inferSec: sep.inferSec,
      detectSec: (performance.now() - tDet) / 1000,
    };
    logLine(`analysis done in ${fmtSec((performance.now() - t0) / 1000)}`);

    analyzeEnvelope();
    setStatus("Done. Adjust the threshold offset and Re-analyze, or Record again.");
    playBtn.disabled = false;
    reanalyzeBtn.disabled = false;
  } catch (err) {
    state.recording = false;
    recordBtn.classList.remove("recording");
    recordBtn.textContent = "● Record";
    showError("record + analyze", err);
    setStatus("FAILED — see error panel above.");
  } finally {
    state.busy = false;
    recordBtn.disabled = false;
  }
}

recordBtn.addEventListener("click", () => {
  if (state.recording) {
    state.stopRequested = true;
    setStatus("Stopping…");
    return;
  }
  recordAndAnalyze();
});

reanalyzeBtn.addEventListener("click", () => {
  try {
    if (!state.envelope) throw new Error("nothing recorded yet");
    analyzeEnvelope();
  } catch (err) {
    showError("re-analyze", err);
  }
});

playBtn.addEventListener("click", () => {
  togglePlayback().catch((err) => showError("playback", err));
});

logLine("page loaded; waiting for Record tap");
