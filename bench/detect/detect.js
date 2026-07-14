/**
 * Stage 1a: in-browser separation + onset detection on the band recording.
 *
 * Validates that the browser port of separation+detection reproduces the
 * offline result (stem_quality_probe / raw_onset_probe), and that reading
 * vocal energy directly off the estimated magnitude spectrogram — no
 * ISTFT, no Wiener filtering — is enough for the energy-onset detector.
 *
 * Chain: fetch band.mp3 (same-origin) → decode/resample to 44.1 kHz →
 * magnitude STFT exactly matching the umxhq training transform (dsp.js,
 * parity-tested against torch.stft) → umxhq vocals core (the bench/umx/
 * ONNX, reused as-is) in consecutive 302-frame (~7 s) chunks → vocal-band
 * 250–3000 Hz energy envelope → the offline detector's onset rule →
 * table + plot against tooling/ground_truth/peggy_o_band.json.
 *
 * Same self-reporting as the other bench pages: everything — numbers,
 * status, full error text — renders on the page.
 */

import {
  FFT, N_FFT, HOP, NB_BINS, hannWindow, stftFrameMag, numFrames,
  powerToEnvelopeDb, smoothEnvelope, detectOnsets, andCombineOnsets,
  loadSections, scoreSections, verdict,
  SONG_OFFSET, BASELINE_WINDOW_SEC, AUTO_OFFSET_DB, RATIO_OFFSET_DB,
  AND_MATCH_SEC, SUSTAIN_WINDOW_SEC, SUSTAIN_TOLERANCE_DB,
  VERDICT_TOLERANCE_SEC,
} from "./dsp.js";

// Same onnxruntime-web version as the other bench pages.
const ORT_VERSION = "1.23.2";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_URL = `${ORT_DIST}ort.webgpu.min.mjs`;
// Reuse the umxhq core exported for bench/umx/ — do not re-export.
const MODEL_URL = "../umx/umx_vocals.onnx";
const AUDIO_URL = "band.mp3";
const GT_URL = "../../tooling/ground_truth/peggy_o_band.json";

const SR = 44100;
const CHUNK_FRAMES = 302; // the ONNX export's fixed nb_frames
const INPUT_NAME = "mag_spec";
const BAND_LO_HZ = 250, BAND_HI_HZ = 3000;
const BIN_HZ = SR / N_FFT;
const BAND_LO = Math.round(BAND_LO_HZ / BIN_HZ); // 23
const BAND_HI = Math.round(BAND_HI_HZ / BIN_HZ); // 279 (inclusive)
const ANCHORS = ["verse_1", "verse_5"];

const $ = (id) => document.getElementById(id);
const runBtn = $("run");

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
const fmtMB = (b) => `${(b / 1024 / 1024).toFixed(1)} MB`;
const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

// ---------------------------------------------------------------------------
// State kept across taps (model/session/audio survive re-runs)
// ---------------------------------------------------------------------------

const state = {
  ort: null,
  session: null,
  ep: null,
  adapterDesc: null,
  audio: null,
  gt: null,
};

// ---------------------------------------------------------------------------
// Setup steps (same patterns as the other bench pages)
// ---------------------------------------------------------------------------

async function loadOrt() {
  setStatus("Loading onnxruntime-web from CDN…");
  let ortMod = await import(ORT_URL);
  if (!ortMod.InferenceSession && ortMod.default) ortMod = ortMod.default;
  if (!ortMod.InferenceSession) {
    throw new Error("onnxruntime-web module loaded but InferenceSession missing");
  }
  ortMod.env.wasm.wasmPaths = ORT_DIST;
  logLine(`onnxruntime-web@${ORT_VERSION} loaded`);
  return ortMod;
}

async function probeWebGPU() {
  if (!("gpu" in navigator)) {
    logLine("navigator.gpu missing — this browser has no WebGPU");
    return { adapter: null, desc: "WebGPU not supported by this browser" };
  }
  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      logLine("navigator.gpu present but requestAdapter() returned null");
      return { adapter: null, desc: "WebGPU present but no adapter available" };
    }
    let info = adapter.info;
    if (!info && adapter.requestAdapterInfo) {
      try { info = await adapter.requestAdapterInfo(); } catch { /* optional */ }
    }
    const desc = info
      ? ["vendor", "architecture", "device", "description"]
          .map((k) => info[k]).filter(Boolean).join(" / ")
      : "(adapter info not exposed)";
    logLine(`WebGPU adapter: ${desc}`);
    return { adapter, desc };
  } catch (err) {
    showError("WebGPU adapter probe", err);
    return { adapter: null, desc: `WebGPU probe failed: ${err}` };
  }
}

async function createSession(ort, haveAdapter) {
  setStatus("Downloading umxhq core (~34 MB)…");
  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status} for ${MODEL_URL}`);
  const bytes = new Uint8Array(await resp.arrayBuffer());
  logLine(`model downloaded: ${fmtMB(bytes.byteLength)}`);

  // One EP per attempt so "provider that ran" is definitive.
  const attempts = haveAdapter ? ["webgpu", "wasm"] : ["wasm"];
  if (!haveAdapter) logLine("no WebGPU adapter; using WASM");
  let lastErr = null;
  for (const ep of attempts) {
    setStatus(`Creating ONNX session on ${ep.toUpperCase()}…`);
    try {
      const session = await ort.InferenceSession.create(bytes.slice(), {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      logLine(`session created on ${ep}`);
      return { session, ep };
    } catch (err) {
      lastErr = err;
      showError(`session creation on ${ep}`, err);
    }
  }
  throw new Error(`all execution providers failed; last error: ${lastErr}`);
}

async function fetchAndDecodeAudio() {
  setStatus("Fetching band recording…");
  const resp = await fetch(AUDIO_URL);
  if (!resp.ok) {
    throw new Error(
      `band recording not found (HTTP ${resp.status} for ${AUDIO_URL}). ` +
      `Add the band recording file to the repo at bench/detect/band.mp3 ` +
      `(same origin as this page) and reload.`);
  }
  const encoded = await resp.arrayBuffer();
  logLine(`band.mp3 fetched: ${fmtMB(encoded.byteLength)}`);
  setStatus("Decoding + resampling to 44.1 kHz…");
  const t0 = performance.now();
  const ctx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: SR,
  });
  try {
    const buf = await ctx.decodeAudioData(encoded);
    if (buf.sampleRate !== SR) {
      logLine(`WARNING: decoded at ${buf.sampleRate} Hz, expected ${SR} Hz`);
    }
    const left = new Float32Array(buf.getChannelData(0));
    // Mono recordings are duplicated into the model's 2 channels.
    const right = buf.numberOfChannels > 1
      ? new Float32Array(buf.getChannelData(1)) : left;
    const decodeSec = (performance.now() - t0) / 1000;
    logLine(`decoded: ${buf.duration.toFixed(1)}s, ${buf.numberOfChannels}ch ` +
            `@ ${buf.sampleRate} Hz in ${fmtSec(decodeSec)}`);
    return { left, right, duration: buf.duration, decodeSec };
  } finally {
    ctx.close().catch(() => {});
  }
}

async function fetchGroundTruth() {
  const resp = await fetch(GT_URL);
  if (!resp.ok) {
    throw new Error(`ground truth fetch failed: HTTP ${resp.status} for ${GT_URL}`);
  }
  const gt = await resp.json();
  const sections = loadSections(gt);
  logLine(`ground truth loaded: ${sections.length} scored sections`);
  return sections;
}

// ---------------------------------------------------------------------------
// Separation: STFT chunks -> umx core -> vocal-band power per frame
// ---------------------------------------------------------------------------

async function separateToBandPower(audio) {
  const { left, right } = audio;
  const totalFrames = numFrames(left.length);
  const nChunks = Math.ceil(totalFrames / CHUNK_FRAMES);
  logLine(`STFT: ${totalFrames} frames (${N_FFT}/${HOP} Hann, center+reflect), ` +
          `${nChunks} chunks of ${CHUNK_FRAMES} frames (~7s, non-overlapping)`);
  logLine(`vocal band ${BAND_LO_HZ}–${BAND_HI_HZ} Hz = bins ${BAND_LO}–${BAND_HI}`);

  const fft = new FFT(N_FFT);
  const window = hannWindow(N_FFT);
  const magL = new Float64Array(NB_BINS);
  const magR = new Float64Array(NB_BINS);
  const sre = new Float64Array(N_FFT);
  const sim = new Float64Array(N_FFT);
  const plane = NB_BINS * CHUNK_FRAMES;
  const input = new Float32Array(2 * plane); // reused across chunks
  const vocalPower = new Float64Array(totalFrames);
  const mixPower = new Float64Array(totalFrames); // for the ratio gate

  let stftSec = 0, inferSec = 0;
  for (let c = 0; c < nChunks; c++) {
    setStatusQuiet(`Chunk ${c + 1}/${nChunks}: STFT…`);
    let t0 = performance.now();
    input.fill(0);
    const framesInChunk = Math.min(CHUNK_FRAMES, totalFrames - c * CHUNK_FRAMES);
    for (let j = 0; j < framesInChunk; j++) {
      const f = c * CHUNK_FRAMES + j;
      stftFrameMag(fft, window, left, right, f, magL, magR, sre, sim);
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

    setStatusQuiet(`Chunk ${c + 1}/${nChunks}: inference…`);
    t0 = performance.now();
    const tensor = new state.ort.Tensor("float32", input, [1, 2, NB_BINS, CHUNK_FRAMES]);
    const out = await state.session.run({ [INPUT_NAME]: tensor });
    const est = Object.values(out)[0].data; // (1, 2, 2049, 302), same layout
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
    if (c % 5 === 0 || c === nChunks - 1) {
      logLine(`chunk ${c + 1}/${nChunks} done (stft ${stftSec.toFixed(1)}s, ` +
              `inference ${inferSec.toFixed(1)}s so far)`);
    }
    await yieldToUI();
  }
  return { vocalPower, mixPower, stftSec, inferSec };
}

// ---------------------------------------------------------------------------
// Plot
// ---------------------------------------------------------------------------

function drawPlot(env, det, sections, duration) {
  const canvas = $("plot");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth || 640, cssH = canvas.clientHeight || 230;
  canvas.width = cssW * dpr;
  canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = cssW, H = cssH, padL = 30, padB = 16, padT = 6;

  const tMax = duration;
  let dbMin = Infinity, dbMax = -Infinity;
  for (const v of env.db) { if (v < dbMin) dbMin = v; if (v > dbMax) dbMax = v; }
  dbMin = Math.max(dbMin, det.thresholdDb - 40);
  const span = Math.max(1, dbMax - dbMin);
  const x = (t) => padL + ((W - padL) * t) / tMax;
  const y = (db) => padT + (H - padT - padB) * (1 - (Math.max(db, dbMin) - dbMin) / span);

  ctx.clearRect(0, 0, W, H);
  ctx.font = "9px system-ui";

  // Ground-truth sections: boundary ticks; shade jam; highlight verse_5.
  for (const s of sections) {
    const isJam = String(s.section_id).startsWith("jam");
    if (isJam) {
      ctx.fillStyle = "rgba(120,120,120,0.15)";
      ctx.fillRect(x(s.start), padT, x(s.end) - x(s.start), H - padT - padB);
    }
    ctx.strokeStyle = s.section_id === "verse_5" ? "#16a34a" : "rgba(120,120,120,0.6)";
    ctx.lineWidth = s.section_id === "verse_5" ? 2 : 1;
    ctx.beginPath();
    ctx.moveTo(x(s.start), padT);
    ctx.lineTo(x(s.start), H - padB);
    ctx.stroke();
    ctx.fillStyle = s.section_id === "verse_5" ? "#16a34a" : "rgba(120,120,120,0.9)";
    ctx.fillText(String(s.section_id).replace("verse_", "v").replace("jam_", "jam"),
                 x(s.start) + 2, padT + 8);
  }

  // Threshold line.
  ctx.strokeStyle = "#9333ea";
  ctx.setLineDash([4, 3]);
  ctx.beginPath();
  ctx.moveTo(padL, y(det.thresholdDb));
  ctx.lineTo(W, y(det.thresholdDb));
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#9333ea";
  ctx.fillText(`threshold ${det.thresholdDb.toFixed(1)} dB`, padL + 2, y(det.thresholdDb) - 3);

  // Envelope.
  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  for (let i = 0; i < env.times.length; i++) {
    const px = x(env.times[i]), py = y(env.db[i]);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Detected onsets.
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 1.5;
  for (const t of det.onsets) {
    ctx.beginPath();
    ctx.moveTo(x(t), padT);
    ctx.lineTo(x(t), H - padB);
    ctx.stroke();
  }

  // Time axis ticks every 60s.
  ctx.fillStyle = "rgba(120,120,120,0.9)";
  for (let t = 0; t <= tMax; t += 60) {
    ctx.fillText(`${Math.round(t)}s`, x(t) + 1, H - 4);
  }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

function renderResults(audio, timings, det, rows, env, sections) {
  const badge = $("provider");
  badge.style.display = "block";
  badge.className = state.ep;
  badge.textContent = state.ep === "webgpu" ? "RAN ON: WEBGPU" : "RAN ON: WASM (no GPU)";

  const total = timings.decodeSec + timings.stftSec + timings.inferSec + timings.detectSec;
  $("r-dur").textContent = fmtSec(audio.duration);
  $("r-total").textContent = fmtSec(total);
  $("r-rtf").textContent = `${(total / audio.duration).toFixed(3)}×`;
  $("r-decode").textContent = fmtSec(timings.decodeSec);
  $("r-stft").textContent = fmtSec(timings.stftSec);
  $("r-infer").textContent = fmtSec(timings.inferSec);
  $("r-detect").textContent = fmtSec(timings.detectSec);
  $("r-onsets").textContent =
    det.onsets.length ? det.onsets.map((t) => `${t.toFixed(1)}s`).join(", ") : "(none)";
  $("r-thresh").textContent =
    `abs ${det.baselineDb === null ? "-" : det.baselineDb.toFixed(1)} / ` +
    `${det.thresholdDb.toFixed(1)} dB · ratio ` +
    `${det.ratio.baselineDb === null ? "-" : det.ratio.baselineDb.toFixed(1)} / ` +
    `${det.ratio.thresholdDb.toFixed(1)} dB`;

  // Verdict: the two anchors the live re-anchoring depends on.
  const lines = [];
  for (const sid of ANCHORS) {
    const v = verdict(rows, sid);
    if (v.detected) {
      lines.push(`${sid}: detected — onset ${v.nearest.toFixed(2)}s, ` +
                 `error ${v.error >= 0 ? "+" : ""}${v.error.toFixed(2)}s ` +
                 `(±${VERDICT_TOLERANCE_SEC.toFixed(0)}s tolerance)`);
    } else if (v.error === null) {
      lines.push(`${sid}: NOT detected — no onsets found`);
    } else {
      lines.push(`${sid}: NOT detected — nearest onset off by ` +
                 `${v.error >= 0 ? "+" : ""}${v.error.toFixed(2)}s`);
    }
  }
  $("verdict").textContent = lines.join("\n");

  const table = $("gt-table");
  while (table.rows.length > 1) table.deleteRow(1);
  for (const r of rows) {
    const tr = table.insertRow();
    if (ANCHORS.includes(r.sectionId)) tr.className = "anchor";
    tr.insertCell().textContent = r.sectionId;
    const c1 = tr.insertCell(); c1.className = "num";
    c1.textContent = `${r.gtStart.toFixed(2)}s`;
    const c2 = tr.insertCell(); c2.className = "num";
    c2.textContent = r.nearest === null ? "-" : `${r.nearest.toFixed(2)}s`;
    const c3 = tr.insertCell(); c3.className = "num";
    c3.textContent = r.error === null ? "-"
      : `${r.error >= 0 ? "+" : ""}${r.error.toFixed(2)}s`;
  }

  drawPlot(env, det, sections, audio.duration);

  $("env-info").textContent =
    `Detector: song offset ${SONG_OFFSET.toFixed(1)}s · AND gate = energy ` +
    `(baseline+${AUTO_OFFSET_DB.toFixed(0)} dB) ∧ vocal/mix ratio ` +
    `(baseline+${RATIO_OFFSET_DB.toFixed(0)} dB) within ±${AND_MATCH_SEC}s · ` +
    `baseline = median of first ${BASELINE_WINDOW_SEC.toFixed(0)}s · 1.0s smoothing · ` +
    `${SUSTAIN_WINDOW_SEC.toFixed(0)}s sustain within ${SUSTAIN_TOLERANCE_DB.toFixed(0)} dB · ` +
    `WebGPU adapter: ${state.adapterDesc} · onnxruntime-web@${ORT_VERSION}`;
  $("results").style.display = "block";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function run() {
  runBtn.disabled = true;
  try {
    if (!state.ort) state.ort = await loadOrt();
    if (!state.adapterDesc) {
      const probe = await probeWebGPU();
      state.adapter = probe.adapter;
      state.adapterDesc = probe.desc;
    }
    if (!state.session) {
      const created = await createSession(state.ort, !!state.adapter);
      state.session = created.session;
      state.ep = created.ep;
    }
    if (!state.gt) state.gt = await fetchGroundTruth();
    if (!state.audio) state.audio = await fetchAndDecodeAudio();

    const sep = await separateToBandPower(state.audio);

    setStatus("Building envelopes + detecting onsets (AND gate)…");
    const t0 = performance.now();
    // Gate 1: absolute vocal-band energy.
    const rawAbs = powerToEnvelopeDb(sep.vocalPower, SR);
    const env = smoothEnvelope(rawAbs.times, rawAbs.db);
    const detAbs = detectOnsets(env.times, env.db, SONG_OFFSET, AUTO_OFFSET_DB);
    // Gate 2: vocal-to-mix ratio — the fraction of room energy the model
    // attributes to voice. Jam leak scales with band loudness and fails
    // this gate; real vocal entries pass both.
    const ratio = new Float64Array(sep.vocalPower.length);
    for (let i = 0; i < ratio.length; i++) {
      ratio[i] = Math.max(sep.vocalPower[i], 1e-24) /
                 Math.max(sep.mixPower[i], 1e-24);
    }
    const rawRatio = powerToEnvelopeDb(ratio, SR);
    const envRatio = smoothEnvelope(rawRatio.times, rawRatio.db);
    const detRatio = detectOnsets(envRatio.times, envRatio.db, SONG_OFFSET,
                                  RATIO_OFFSET_DB);
    const gated = andCombineOnsets(detAbs.onsets, detRatio.onsets);
    const det = { ...detAbs, onsets: gated, ratio: detRatio };
    const detectSec = (performance.now() - t0) / 1000;
    logLine(`abs gate: baseline ${detAbs.baselineDb?.toFixed(1)} dB, threshold ` +
            `${detAbs.thresholdDb.toFixed(1)} dB, ${detAbs.onsets.length} onsets`);
    logLine(`ratio gate: baseline ${detRatio.baselineDb?.toFixed(1)} dB, threshold ` +
            `${detRatio.thresholdDb.toFixed(1)} dB, ${detRatio.onsets.length} onsets`);
    logLine(`AND gate (±${AND_MATCH_SEC}s): ${gated.length} onsets`);

    const rows = scoreSections(state.gt, det.onsets);
    renderResults(state.audio, { ...sep, decodeSec: state.audio.decodeSec, detectSec },
                  det, rows, env, state.gt);
    setStatus("Done.");
    runBtn.textContent = "Run again";
  } catch (err) {
    showError("benchmark run", err);
    setStatus("FAILED — see error panel above.");
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", () => { run(); });
logLine("page loaded; waiting for Run tap");
