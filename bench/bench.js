/**
 * WebGPU separation-speed benchmark.
 *
 * Loads Demucs (htdemucs) ONNX via the published demucs-web package on
 * onnxruntime-web, runs it once over the bundled 7s clip, and reports the
 * real-time factor. Built to run on an iPhone with no dev tools: every
 * number, status change, and error is rendered on the page.
 *
 * Everything heavy is dynamically imported inside the Run handler so that
 * even a failed library load from the CDN surfaces in the on-page error
 * panel instead of killing the module before the UI exists.
 */

const ORT_VERSION = "1.23.2";
const DEMUCS_WEB_VERSION = "1.0.2";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_URL = `${ORT_DIST}ort.webgpu.min.mjs`;
const DEMUCS_URL = `https://cdn.jsdelivr.net/npm/demucs-web@${DEMUCS_WEB_VERSION}/src/index.js`;
// htdemucs exported to ONNX by the demucs-web author; same default URL the
// package's CONSTANTS.DEFAULT_MODEL_URL points at. ~172 MB.
const MODEL_URL =
  "https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx";
const CLIP_URL = "clip.mp3";
const MODEL_SAMPLE_RATE = 44100;

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

// ---------------------------------------------------------------------------
// Benchmark state (kept across clicks: re-running reuses the loaded model,
// so a second tap gives warm-run numbers without re-downloading 172 MB)
// ---------------------------------------------------------------------------

const state = {
  ort: null,
  processor: null,
  ep: null, // "webgpu" | "wasm" -- the provider the session was ACTUALLY created with
  adapterDesc: null,
  audio: null, // { left, right, duration, sampleRate }
  downloadSec: null,
  initSec: null,
  runCount: 0,
};

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

async function loadLibraries() {
  setStatus("Loading onnxruntime-web + demucs-web from CDN…");
  let ortMod = await import(ORT_URL);
  if (!ortMod.InferenceSession && ortMod.default) ortMod = ortMod.default;
  if (!ortMod.InferenceSession) {
    throw new Error("onnxruntime-web module loaded but InferenceSession missing");
  }
  // The .wasm / worker assets live next to the .mjs on the CDN.
  ortMod.env.wasm.wasmPaths = ORT_DIST;
  const demucsMod = await import(DEMUCS_URL);
  if (!demucsMod.DemucsProcessor) {
    throw new Error("demucs-web module loaded but DemucsProcessor missing");
  }
  logLine(`libraries loaded (onnxruntime-web@${ORT_VERSION}, demucs-web@${DEMUCS_WEB_VERSION})`);
  return { ort: ortMod, DemucsProcessor: demucsMod.DemucsProcessor };
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

async function decodeClip() {
  setStatus("Decoding bundled clip…");
  const resp = await fetch(CLIP_URL);
  if (!resp.ok) throw new Error(`clip fetch failed: HTTP ${resp.status}`);
  const encoded = await resp.arrayBuffer();
  // Ask for a 44.1 kHz context so decodeAudioData resamples to the model rate.
  const ctx = new (window.AudioContext || window.webkitAudioContext)({
    sampleRate: MODEL_SAMPLE_RATE,
  });
  try {
    const buf = await ctx.decodeAudioData(encoded);
    const left = buf.getChannelData(0);
    const right = buf.numberOfChannels > 1 ? buf.getChannelData(1) : left;
    if (buf.sampleRate !== MODEL_SAMPLE_RATE) {
      logLine(`WARNING: decoded at ${buf.sampleRate} Hz, model expects ${MODEL_SAMPLE_RATE} Hz`);
    }
    logLine(`clip decoded: ${buf.duration.toFixed(2)}s, ${buf.numberOfChannels}ch @ ${buf.sampleRate} Hz`);
    return {
      left: new Float32Array(left),
      right: new Float32Array(right),
      duration: buf.duration,
      sampleRate: buf.sampleRate,
    };
  } finally {
    ctx.close().catch(() => {});
  }
}

async function downloadModel() {
  setStatus("Downloading model (~172 MB)…");
  const t0 = performance.now();
  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status}`);
  let buffer;
  const total = parseInt(resp.headers.get("Content-Length") || "0", 10);
  if (resp.body) {
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      setStatus(`Downloading model: ${fmtMB(loaded)}${total ? ` / ${fmtMB(total)}` : ""}`);
    }
    const all = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    buffer = all.buffer;
  } else {
    buffer = await resp.arrayBuffer();
  }
  const sec = (performance.now() - t0) / 1000;
  logLine(`model downloaded: ${fmtMB(buffer.byteLength)} in ${fmtSec(sec)}`);
  return { buffer, sec };
}

/**
 * Create the session, trying WebGPU first (when an adapter exists), then
 * WASM. Each attempt pins a SINGLE execution provider via demucs-web's
 * sessionOptions passthrough, so whichever attempt succeeds tells us
 * definitively which provider the model actually runs on — no silent
 * EP-level fallback inside one session.
 */
async function createProcessor(ort, DemucsProcessor, modelBuffer, webgpuAdapter) {
  const callbacks = {
    onLog: (phase, msg) => logLine(`demucs-web[${phase}]: ${msg}`),
    onProgress: ({ currentSegment, totalSegments }) =>
      setStatus(`Separating… segment ${currentSegment}/${totalSegments}`),
  };
  const attempts = webgpuAdapter ? ["webgpu", "wasm"] : ["wasm"];
  if (!webgpuAdapter) logLine("skipping WebGPU attempt (no adapter); going straight to WASM");

  let lastErr = null;
  for (const ep of attempts) {
    setStatus(`Creating ONNX session on ${ep.toUpperCase()}…`);
    const processor = new DemucsProcessor({
      ort,
      sessionOptions: { executionProviders: [ep] },
      ...callbacks,
    });
    const t0 = performance.now();
    try {
      let buf = modelBuffer;
      if (buf.byteLength === 0) {
        // A failed prior attempt can leave the buffer detached (ort may
        // transfer it to a worker). Re-download rather than fail.
        logLine("model buffer was detached by failed attempt; re-downloading");
        buf = (await downloadModel()).buffer;
      }
      await processor.loadModel(buf);
      const initSec = (performance.now() - t0) / 1000;
      logLine(`session created on ${ep} in ${fmtSec(initSec)}`);
      return { processor, ep, initSec };
    } catch (err) {
      lastErr = err;
      showError(`session creation on ${ep}`, err);
    }
  }
  throw new Error(`all execution providers failed; last error: ${lastErr}`);
}

// ---------------------------------------------------------------------------
// Results rendering
// ---------------------------------------------------------------------------

function renderResults(computeSec) {
  const { ep, adapterDesc, audio, downloadSec, initSec, runCount } = state;

  const badge = $("provider");
  badge.style.display = "block";
  badge.className = ep;
  badge.textContent = ep === "webgpu" ? "RAN ON: WEBGPU" : "RAN ON: WASM (no GPU)";

  const rtf = computeSec / audio.duration;
  const line = $("rtf-line");
  if (rtf < 1) {
    line.textContent =
      `Real-time factor ${rtf.toFixed(2)}× — FASTER than real-time: ` +
      `1s of audio separates in ${rtf.toFixed(2)}s.`;
  } else {
    line.textContent =
      `Real-time factor ${rtf.toFixed(2)}× — SLOWER than real-time: ` +
      `1s of audio takes ${rtf.toFixed(2)}s to separate.`;
  }

  $("r-rtf").textContent = `${rtf.toFixed(2)}×`;
  $("r-compute").textContent = fmtSec(computeSec);
  $("r-clip").textContent = fmtSec(audio.duration);
  $("r-load").textContent = fmtSec(downloadSec + initSec);
  $("r-download").textContent = fmtSec(downloadSec);
  $("r-init").textContent = fmtSec(initSec);

  $("gpu-info").textContent = `WebGPU adapter: ${adapterDesc}`;
  $("env-info").textContent =
    `run #${runCount} (run #2+ reuses the loaded model = warm numbers) · ` +
    `crossOriginIsolated=${self.crossOriginIsolated} · ` +
    `hardwareConcurrency=${navigator.hardwareConcurrency ?? "?"} · ` +
    `onnxruntime-web@${ORT_VERSION} · demucs-web@${DEMUCS_WEB_VERSION}`;

  $("results").style.display = "block";
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function run() {
  runBtn.disabled = true;
  try {
    if (!state.processor) {
      const { ort, DemucsProcessor } = await loadLibraries();
      state.ort = ort;

      const { adapter, desc } = await probeWebGPU();
      state.adapterDesc = desc;

      state.audio = await decodeClip();

      const dl = await downloadModel();
      state.downloadSec = dl.sec;

      const created = await createProcessor(ort, DemucsProcessor, dl.buffer, adapter);
      state.processor = created.processor;
      state.ep = created.ep;
      state.initSec = created.initSec;
    } else {
      logLine("model already loaded — warm run");
    }

    setStatus("Separating…");
    const t0 = performance.now();
    await state.processor.separate(state.audio.left, state.audio.right);
    const computeSec = (performance.now() - t0) / 1000;
    logLine(`separation finished in ${fmtSec(computeSec)}`);

    state.runCount += 1;
    renderResults(computeSec);
    setStatus("Done.");
    runBtn.textContent = "Run again (model stays loaded)";
  } catch (err) {
    showError("benchmark run", err);
    setStatus("FAILED — see error panel above.");
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", () => { run(); });
logLine("page loaded; waiting for Run tap");
