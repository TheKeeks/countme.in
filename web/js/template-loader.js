/**
 * template-loader.js — loads song templates from /templates/.
 *
 * Templates are produced by the Python tooling (tooling/alignment.py).
 * Each template is a JSON file describing the song's structure plus
 * per-line timing and chroma fingerprints.
 *
 * To add a new song: drop its *_aligned.json file into web/templates/
 * and add its filename to TEMPLATE_INDEX below. (Later we'll auto-
 * discover via a manifest, but explicit list keeps this readable.)
 */

const TEMPLATE_INDEX = [
  'peggy_o_aligned.json',
];

const cache = new Map();

export async function loadAvailableSongs() {
  const songs = [];
  for (const filename of TEMPLATE_INDEX) {
    try {
      const tmpl = await loadTemplateByFile(filename);
      songs.push({
        id: tmpl.song_id,
        title: tmpl.title,
        duration_sec: tmpl.audio_features?.duration_sec ?? null,
        filename,
      });
    } catch (err) {
      console.warn(`Failed to load template ${filename}:`, err);
    }
  }
  return songs;
}

export async function loadTemplate(songId) {
  for (const filename of TEMPLATE_INDEX) {
    const tmpl = await loadTemplateByFile(filename);
    if (tmpl.song_id === songId) return tmpl;
  }
  throw new Error(`Template not found for song_id="${songId}"`);
}

async function loadTemplateByFile(filename) {
  if (cache.has(filename)) return cache.get(filename);
  const res = await fetch(`templates/${filename}`);
  if (!res.ok) throw new Error(`HTTP ${res.status} loading ${filename}`);
  const tmpl = await res.json();
  cache.set(filename, tmpl);
  return tmpl;
}
