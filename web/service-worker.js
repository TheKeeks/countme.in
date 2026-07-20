/**
 * service-worker.js — offline caching for the countme.in PWA.
 *
 * Strategy:
 *   - App shell (HTML/CSS/JS/manifest) cached on install
 *   - Templates cached on first fetch (stale-while-revalidate)
 *   - Whisper model files cached on first fetch (cache-first, large
 *     blobs that change rarely)
 *
 * Bump CACHE_VERSION whenever you ship changes that need to invalidate
 * existing client caches.
 */

const CACHE_VERSION = 'v4';
const SHELL_CACHE = `countme-in-shell-${CACHE_VERSION}`;
const ASSET_CACHE = `countme-in-assets-${CACHE_VERSION}`;
const MODEL_CACHE = `countme-in-models-${CACHE_VERSION}`;

const SHELL_FILES = [
  './',
  './index.html',
  './manifest.json',
  './css/style.css',
  './js/app.js',
  './js/template-loader.js',
  './js/display.js',
  './js/position-tracker.js',
  './js/audio-engine.js',
  './js/vocal-onset.js',
  './js/chroma.js',
  './icons/icon-192.png',
  './icons/icon-512.png',
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(SHELL_CACHE)
      .then(cache => cache.addAll(SHELL_FILES))
      .catch(err => console.warn('SW install: shell cache addAll failed', err))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys
      .filter(k => k.startsWith('countme-in-') &&
                   ![SHELL_CACHE, ASSET_CACHE, MODEL_CACHE].includes(k))
      .map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Only handle same-origin requests + the huggingface CDN for Whisper.
  if (url.origin !== self.location.origin &&
      !url.host.endsWith('huggingface.co') &&
      !url.host.endsWith('cdn-lfs.huggingface.co') &&
      !url.host.endsWith('cdn.jsdelivr.net')) {
    return;
  }

  // Whisper model blobs: cache-first, never re-fetch unless evicted.
  if (url.host.endsWith('huggingface.co') ||
      url.host.endsWith('cdn-lfs.huggingface.co')) {
    event.respondWith(cacheFirst(event.request, MODEL_CACHE));
    return;
  }

  // Templates: stale-while-revalidate so newly built templates show up
  // after one page load even when an older cached copy exists.
  if (url.pathname.includes('/templates/')) {
    event.respondWith(staleWhileRevalidate(event.request, ASSET_CACHE));
    return;
  }

  // App shell: cache-first
  event.respondWith(cacheFirst(event.request, SHELL_CACHE));
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) return cached;
  try {
    const fresh = await fetch(request);
    if (fresh.ok) cache.put(request, fresh.clone());
    return fresh;
  } catch (err) {
    // Offline + no cache -> let the failure propagate
    throw err;
  }
}

async function staleWhileRevalidate(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  const networkPromise = fetch(request).then(res => {
    if (res.ok) cache.put(request, res.clone());
    return res;
  }).catch(() => null);
  return cached || (await networkPromise) || new Response('Offline', { status: 503 });
}
