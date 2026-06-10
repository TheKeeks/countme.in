/**
 * Open-Unmix WebGPU benchmark.
 *
 * Demucs (172 MB) crashed iOS Safari on memory; this page benchmarks the
 * far lighter Open-Unmix umxhq vocals spectrogram core (~34 MB ONNX,
 * exported by .github/workflows/export-umx.yml and bundled next to this
 * file). Same forward pass timed twice — once on a WebGPU-preferred
 * session, once on a WASM-only session — so the table shows directly
 * whether WebGPU actually accelerates this model on the device.
 *
 * Self-reporting like bench/: every number, status change, and error is
 * rendered on the page; nothing relies on a console. Libraries are
 * dynamically imported inside the Run handler so even CDN failures land
 * in the on-page error panel.
 */

// Same onnxruntime-web version as bench/bench.js.
const ORT_VERSION = "1.23.2";
const ORT_DIST = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const ORT_URL = `${ORT_DIST}ort.webgpu.min.mjs`;
const MODEL_URL = "./umx_vocals.onnx";

// Fixed export shape: (nb_samples, nb_channels, nb_bins, nb_frames).
// 302 frames at hop 1024 / 44100 Hz <=> (302-1)*1024/44100 = 6.989 s.
const INPUT_SHAPE = [1, 2, 2049, 302];
const INPUT_NAME = "mag_spec";
const AUDIO_SEC = ((INPUT_SHAPE[3] - 1) * 1024) / 44100;

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
// State kept across taps: sessions stay loaded, re-runs give warm numbers.
// ---------------------------------------------------------------------------

const state = {
  ort: null,
  modelBytes: null, // Uint8Array master copy; sliced per session create
  adapter: null,
  adapterDesc: null,
  input: null,
  passes: { a: null, b: null }, // { ep, session, loadSec }
};

// ---------------------------------------------------------------------------
// Steps
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

async function downloadModel() {
  setStatus("Downloading model (~34 MB)…");
  const t0 = performance.now();
  const resp = await fetch(MODEL_URL);
  if (!resp.ok) throw new Error(`model fetch failed: HTTP ${resp.status}`);
  const buffer = await resp.arrayBuffer();
  logLine(`model downloaded: ${fmtMB(buffer.byteLength)} in ${fmtSec((performance.now() - t0) / 1000)}`);
  return new Uint8Array(buffer);
}

function buildInput(ort) {
  // Timing depends on shape, not content: synthetic magnitudes are fine.
  const n = INPUT_SHAPE.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  for (let i = 0; i < n; i++) data[i] = Math.random() * 0.1;
  logLine(`input tensor: [${INPUT_SHAPE.join(", ")}] ≈ ${AUDIO_SEC.toFixed(2)}s of audio`);
  return new ort.Tensor("float32", data, INPUT_SHAPE);
}

/**
 * Create a session pinned to a SINGLE execution provider per attempt, so
 * whichever attempt succeeds is definitively the provider that runs —
 * onnxruntime-web doesn't report which EP a ['webgpu','wasm'] session
 * actually picked, so we never pass an ambiguous list.
 */
async function createSession(ort, label, eps) {
  let lastErr = null;
  for (const ep of eps) {
    setStatus(`[${label}] creating session on ${ep.toUpperCase()}…`);
    const t0 = performance.now();
    try {
      const session = await ort.InferenceSession.create(state.modelBytes.slice(), {
        executionProviders: [ep],
        graphOptimizationLevel: "all",
      });
      const loadSec = (performance.now() - t0) / 1000;
      logLine(`[${label}] session created on ${ep} in ${fmtSec(loadSec)}`);
      return { session, ep, loadSec };
    } catch (err) {
      lastErr = err;
      showError(`[${label}] session creation on ${ep}`, err);
    }
  }
  throw new Error(`[${label}] all providers failed; last error: ${lastErr}`);
}

async function timeForward(session, label, runName) {
  setStatus(`[${label}] running inference (${runName})…`);
  const t0 = performance.now();
  const out = await session.run({ [INPUT_NAME]: state.input });
  const sec = (performance.now() - t0) / 1000;
  const first = Object.values(out)[0];
  logLine(`[${label}] ${runName}: ${fmtSec(sec)} (output [${first.dims.join(", ")}])`);
  return sec;
}

async function runPass(key, label, eps) {
  if (!state.passes[key]) {
    state.passes[key] = await createSession(state.ort, label, eps);
  }
  const pass = state.passes[key];
  const coldSec = await timeForward(pass.session, label, "1st run");
  const warmSec = await timeForward(pass.session, label, "2nd run");
  return { ...pass, coldSec, warmSec };
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

function fillColumn(prefix, r) {
  const badge = $(`${prefix}-prov`);
  badge.textContent = r.ep.toUpperCase();
  badge.className = `badge ${r.ep}`;
  $(`${prefix}-load`).textContent = fmtSec(r.loadSec);
  $(`${prefix}-cold`).textContent = fmtSec(r.coldSec);
  $(`${prefix}-warm`).textContent = fmtSec(r.warmSec);
  $(`${prefix}-rtf`).textContent = `${(r.warmSec / AUDIO_SEC).toFixed(3)}×`;
}

function renderResults(a, b) {
  fillColumn("a", a);
  fillColumn("b", b);

  const verdict = $("verdict");
  const rtfA = a.warmSec / AUDIO_SEC;
  const lines = [];
  if (a.ep === "webgpu") {
    const speedup = b.warmSec / a.warmSec;
    lines.push(`WebGPU engaged and is ${speedup.toFixed(1)}× ${speedup >= 1 ? "faster" : "SLOWER"} than WASM on this model.`);
  } else {
    lines.push("WebGPU did NOT engage — both columns ran on WASM.");
  }
  lines.push(
    rtfA < 1
      ? `Best run is FASTER than real-time (${rtfA.toFixed(3)}×: 1s of audio takes ${rtfA.toFixed(3)}s).`
      : `Best run is SLOWER than real-time (${rtfA.toFixed(3)}×: 1s of audio takes ${rtfA.toFixed(3)}s).`
  );
  verdict.textContent = lines.join(" ");

  $("gpu-info").textContent = `WebGPU adapter: ${state.adapterDesc}`;
  $("env-info").textContent =
    `crossOriginIsolated=${self.crossOriginIsolated} · ` +
    `hardwareConcurrency=${navigator.hardwareConcurrency ?? "?"} · ` +
    `onnxruntime-web@${ORT_VERSION} · umxhq vocals spectrogram core`;
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
    if (!state.modelBytes) state.modelBytes = await downloadModel();
    if (!state.input) state.input = buildInput(state.ort);

    const epsA = state.adapter ? ["webgpu", "wasm"] : ["wasm"];
    if (!state.adapter) logLine("no WebGPU adapter: WebGPU-preferred pass will run on WASM");
    const a = await runPass("a", "webgpu-preferred", epsA);
    const b = await runPass("b", "wasm-only", ["wasm"]);

    renderResults(a, b);
    setStatus("Done.");
    runBtn.textContent = "Run again (sessions stay loaded)";
  } catch (err) {
    showError("benchmark run", err);
    setStatus("FAILED — see error panel above.");
  } finally {
    runBtn.disabled = false;
  }
}

runBtn.addEventListener("click", () => { run(); });
logLine("page loaded; waiting for Run tap");
