# WebGPU separation-speed benchmark — hosting

Static files only; no build step. Host the `bench/` folder anywhere that
serves HTTPS (WebGPU requires a secure context) and open `index.html` on
the phone. Everything self-reports on the page — numbers and errors — so
no dev tools are needed.

What it does: tap **Run benchmark** → downloads the ~172 MB htdemucs ONNX
model from Hugging Face → separates a bundled 7 s public-domain clip →
shows which execution provider actually ran (WebGPU vs WASM), model load
time, compute time, and the real-time factor. Use Wi-Fi: the model
re-downloads on every fresh page load.

## Option A: GitHub Pages (easiest, try first)

1. Repo must be public (or have Pages enabled on your plan).
2. Settings → Pages → Deploy from branch → pick the branch, folder `/ (root)`.
3. Open `https://<user>.github.io/<repo>/bench/` on the iPhone.

Caveat: GitHub Pages ignores the `_headers` file, so the page is **not
cross-origin isolated** there. WebGPU itself does not need isolation and
should still run; the only cost is that a WASM fallback run is limited to
a single thread (onnxruntime-web multithreading needs
`SharedArrayBuffer`). If the page shows an error mentioning
`SharedArrayBuffer` or cross-origin isolation, use Option B.

## Option B: Cloudflare Pages or Netlify (honors `_headers`)

Both serve the bundled `_headers` file, which sets
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` — this makes the page
cross-origin isolated so onnxruntime-web can use threads.

- **Cloudflare Pages**: create a project → connect the repo (or direct
  upload) → build command: none → output directory: `bench`.
- **Netlify**: drag-and-drop the `bench/` folder onto the Netlify drop
  zone, or point a site at the repo with publish directory `bench`.

## Quick local check (laptop)

```
cd bench && python3 -m http.server 8080
```

then open http://localhost:8080 (localhost counts as a secure context).
Note `python3 -m http.server` does not send the COOP/COEP headers, so it
behaves like GitHub Pages.

## What's pinned where

- `demucs-web@1.0.2` and `onnxruntime-web@1.23.2` from jsDelivr (exact
  versions pinned in `bench.js`).
- Model: `htdemucs_embedded.onnx` (~172 MB) fetched at runtime from
  `https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx`
  (the demucs-web package's own published model). Not committed to the
  repo — it exceeds GitHub's 100 MB per-file limit.
- Clip: `clip.mp3`, 7 s of Scott Joplin's 1916 piano-roll performance of
  Maple Leaf Rag (public domain, via Wikimedia Commons).

## Known iPhone caveats

- iOS Safari may kill the tab on a memory spike while the 172 MB model
  initializes; that shows up as a page reload rather than an on-page
  error (OS-level, not catchable). If it happens, close other tabs and
  retry — and that is itself a meaningful feasibility data point.
- The first run on WebGPU includes shader compilation; tap "Run again"
  for a warm number (the model stays loaded).
