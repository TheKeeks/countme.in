/**
 * app.js — entry point for the countme.in PWA.
 *
 * Wires up the home / prompter / settings screens and routes user
 * input to the position tracker (in tracker.js) and display (in
 * display.js). The actual ML / DSP work lives in those modules.
 */

import { loadAvailableSongs, loadTemplate } from './template-loader.js';
import { renderLyrics, setCurrentLine, applyFontSize, applyLookahead, buildEmergencyList } from './display.js';
import { PositionTracker } from './position-tracker.js';
import { AudioEngine } from './audio-engine.js';

// --- App state -------------------------------------------------------

const state = {
  currentScreen: 'home',
  currentTemplate: null,
  tracker: null,
  audio: null,
  settings: loadSettings(),
};

function loadSettings() {
  // Reasonable defaults; on-device personalization can be added later.
  return {
    fontSize: 64,
    lookahead: 2,
  };
}

// --- Screen routing --------------------------------------------------

function showScreen(name) {
  document.querySelectorAll('.screen').forEach(el => el.classList.remove('active'));
  const target = document.getElementById(`screen-${name}`);
  if (target) target.classList.add('active');
  state.currentScreen = name;
}

// --- Home screen -----------------------------------------------------

async function initHome() {
  const list = document.getElementById('song-list');
  try {
    const songs = await loadAvailableSongs();
    list.innerHTML = '';
    if (songs.length === 0) {
      list.innerHTML = '<li class="song-list-loading">no songs yet — add a template to web/templates/</li>';
      return;
    }
    for (const song of songs) {
      const li = document.createElement('li');
      li.innerHTML = `
        <span>${escapeHtml(song.title)}</span>
        <span class="song-list-meta">${formatDuration(song.duration_sec)}</span>
      `;
      li.addEventListener('click', () => openSong(song.id));
      list.appendChild(li);
    }
  } catch (err) {
    console.error('failed to load songs', err);
    list.innerHTML = `<li class="song-list-loading">error loading songs: ${escapeHtml(err.message)}</li>`;
  }
}

// --- Prompter screen -------------------------------------------------

async function openSong(songId) {
  const template = await loadTemplate(songId);
  state.currentTemplate = template;

  document.getElementById('current-song-title').textContent = template.title;
  applyFontSize(state.settings.fontSize);
  applyLookahead(state.settings.lookahead);
  renderLyrics(template);
  buildEmergencyList(template, onEmergencyLineSelected);

  // Position tracker + audio engine are constructed but not started.
  // The user has to tap the mic button to begin tracking.
  state.tracker = new PositionTracker(template);
  state.tracker.onPositionChange = ({ sectionId, lineIndex, confidence }) => {
    setCurrentLine(sectionId, lineIndex);
    updateTrackingStatus(state.tracker.state);
  };
  state.audio = new AudioEngine();
  state.audio.onAudioFrame = (frame) => state.tracker.consume(frame);

  showScreen('prompter');
  setCurrentLine(null, null); // start with nothing highlighted
}

function updateTrackingStatus(s) {
  const el = document.getElementById('tracking-status');
  el.textContent = s;
  el.dataset.state = s;
}

function onEmergencyLineSelected(sectionId, lineIndex) {
  if (state.tracker) state.tracker.snapTo(sectionId, lineIndex);
  setCurrentLine(sectionId, lineIndex);
  hideEmergency();
}

// --- Control panel & emergency overlay -------------------------------

function showControls() { document.getElementById('control-panel').classList.add('visible'); }
function hideControls() { document.getElementById('control-panel').classList.remove('visible'); }
function showEmergency() { document.getElementById('emergency-overlay').classList.add('visible'); }
function hideEmergency() { document.getElementById('emergency-overlay').classList.remove('visible'); }

async function toggleMic() {
  const btn = document.getElementById('btn-mic');
  if (!state.audio.running) {
    try {
      await state.audio.start();
      state.tracker.start();
      btn.classList.add('recording');
      updateTrackingStatus('listening');
    } catch (err) {
      console.error('mic start failed', err);
      alert('Could not start microphone: ' + err.message);
    }
  } else {
    await state.audio.stop();
    state.tracker.stop();
    btn.classList.remove('recording');
    updateTrackingStatus('idle');
  }
}

function backToHome() {
  if (state.audio && state.audio.running) state.audio.stop();
  if (state.tracker) state.tracker.stop();
  document.getElementById('btn-mic').classList.remove('recording');
  hideControls();
  hideEmergency();
  showScreen('home');
}

// --- Settings --------------------------------------------------------

function initSettings() {
  const fs = document.getElementById('font-size-slider');
  const fsVal = document.getElementById('font-size-value');
  fs.value = state.settings.fontSize;
  fsVal.textContent = state.settings.fontSize + 'px';
  fs.addEventListener('input', () => {
    state.settings.fontSize = Number(fs.value);
    fsVal.textContent = fs.value + 'px';
    applyFontSize(state.settings.fontSize);
  });

  const la = document.getElementById('lookahead-slider');
  const laVal = document.getElementById('lookahead-value');
  la.value = state.settings.lookahead;
  laVal.textContent = state.settings.lookahead;
  la.addEventListener('input', () => {
    state.settings.lookahead = Number(la.value);
    laVal.textContent = la.value;
    applyLookahead(state.settings.lookahead);
  });

  updateCacheStatus();
}

async function updateCacheStatus() {
  const el = document.getElementById('cache-status');
  if (!('caches' in window)) { el.textContent = 'unavailable'; return; }
  try {
    const cacheNames = await caches.keys();
    const ours = cacheNames.filter(n => n.startsWith('countme-in'));
    el.textContent = ours.length ? 'cached for offline' : 'not yet cached';
  } catch {
    el.textContent = 'check failed';
  }
  document.getElementById('model-status').textContent = 'not loaded (phase 3)';
}

// --- Utilities -------------------------------------------------------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  }[c]));
}

function formatDuration(sec) {
  if (!sec) return '';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60).toString().padStart(2, '0');
  return `${m}:${s}`;
}

// --- Event wiring ----------------------------------------------------

window.addEventListener('DOMContentLoaded', () => {
  initHome();
  initSettings();

  // Home
  document.getElementById('btn-settings').addEventListener('click', () => showScreen('settings'));
  document.getElementById('btn-settings-back').addEventListener('click', () => showScreen('home'));

  // Prompter controls
  document.getElementById('control-reveal-zone').addEventListener('click', showControls);
  document.getElementById('btn-back').addEventListener('click', backToHome);
  document.getElementById('btn-mic').addEventListener('click', toggleMic);
  document.getElementById('btn-lost').addEventListener('click', () => { hideControls(); showEmergency(); });
  document.getElementById('btn-emergency-close').addEventListener('click', hideEmergency);

  // Hide controls when tapping outside the panel
  document.getElementById('lyric-viewport').addEventListener('click', hideControls);

  // Keep iOS Safari from sleeping the screen mid-show (best-effort).
  requestWakeLock();
});

async function requestWakeLock() {
  if (!('wakeLock' in navigator)) return;
  try {
    await navigator.wakeLock.request('screen');
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') navigator.wakeLock.request('screen').catch(()=>{});
    });
  } catch (err) {
    console.warn('wake lock denied', err);
  }
}

// --- Service worker (PWA offline) ------------------------------------

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('service-worker.js')
      .then(reg => console.log('SW registered:', reg.scope))
      .catch(err => console.warn('SW registration failed:', err));
  });
}
