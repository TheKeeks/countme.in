#!/usr/bin/env node
/**
 * replay.mjs — offline replay harness for the shipped live tracker.
 *
 * Decodes a band recording to the 16 kHz mono stream the AudioEngine
 * delivers, feeds it through the REAL web/js/position-tracker.js (and,
 * through it, chroma.js + vocal-onset.js) in 2048-sample frames exactly
 * as the browser does, and scores the tracker's per-step reference
 * position against hand-labeled ground truth. No re-implementation of
 * the tracker lives here — if the browser code changes, this measures
 * the change.
 *
 * Metrics, per take:
 *   block   fraction of scored steps whose predicted section is in the
 *           same section block as the ground truth (verses 1-4 form one
 *           block, verses 5-8 another — derived from the template:
 *           contiguous runs of lyric sections; instrumental sections
 *           match by family, e.g. jam_2 ~ jam_1).
 *   strict  exact-section accuracy; only when the ground truth labels
 *           every lyric section individually (take 1).
 *   verse5  re-entry error after the jam: first time the prediction
 *           enters the verse_5 block and stays there 10 s, minus the
 *           labeled verse_5 start (positive = late).
 *   locked  fraction of scored steps with tracker.state === 'locked'.
 * Steps inside unlabeled gaps or `_`-prefixed sections are not scored.
 * Audio is fed from the ground truth's song_offset_sec (the moment the
 * singer would tap start), and step times are reported in absolute
 * audio time to match the labels.
 *
 * Usage:
 *   node tests/replay/replay.mjs                       # every take with audio
 *   node tests/replay/replay.mjs --take take1 --check  # enforce floors
 *   node tests/replay/replay.mjs --audio take2=Peggy.m4a --audio take3=URL
 *   node tests/replay/replay.mjs --tempo 0.8         # band 20% slower
 *   node tests/replay/replay.mjs --json out.json --md out.md
 *
 * Needs `ffmpeg` on PATH (or FFMPEG=/path/to/ffmpeg) to decode audio.
 */

import { spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { PositionTracker } from '../../web/js/position-tracker.js';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const SR = 16000;
const FRAME = 2048;          // AudioEngine's ScriptProcessor buffer size
const FFT_N = 4096;          // chroma.js frame: a step is decided at frame end
const REENTRY_HOLD_SEC = 10; // verse5 re-entry must persist this long

// -- CLI ---------------------------------------------------------------

function parseArgs(argv) {
  const args = { takes: [], audio: {}, tempo: 1, check: false, json: null, md: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const val = () => {
      if (i + 1 >= argv.length) throw new Error(`${a} needs a value`);
      return argv[++i];
    };
    if (a === '--take') args.takes.push(val());
    else if (a === '--audio') {
      const v = val();
      const eq = v.indexOf('=');
      if (eq < 1) throw new Error(`--audio expects NAME=PATH_OR_URL, got ${v}`);
      args.audio[v.slice(0, eq)] = v.slice(eq + 1);
    } else if (a === '--tempo') {
      args.tempo = Number(val());
      if (!(args.tempo >= 0.5 && args.tempo <= 2)) throw new Error('--tempo must be in [0.5, 2]');
    } else if (a === '--check') args.check = true;
    else if (a === '--json') args.json = val();
    else if (a === '--md') args.md = val();
    else if (a === '-h' || a === '--help') {
      console.log(readFileSync(fileURLToPath(import.meta.url), 'utf8')
        .split('\n').slice(1, 40).join('\n'));
      process.exit(0);
    } else throw new Error(`unknown argument ${a}`);
  }
  return args;
}

// -- Audio -------------------------------------------------------------

async function resolveAudio(spec, scratch) {
  if (!/^https?:\/\//.test(spec)) {
    const p = resolve(ROOT, spec);
    if (!existsSync(p)) throw new Error(`audio not found: ${p}`);
    return p;
  }
  // Dropbox share links serve an HTML page unless dl=1.
  const url = spec.includes('dropbox.com') ? spec.replace('dl=0', 'dl=1') : spec;
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`download failed (${res.status}): ${url}`);
  const out = join(scratch, `audio_${Date.now()}`);
  writeFileSync(out, Buffer.from(await res.arrayBuffer()));
  return out;
}

/** Decode to the AudioEngine's 16 kHz mono stream. tempo != 1 applies a
 *  pitch-preserving time stretch (ffmpeg atempo): a synthetic band that
 *  plays the same take faster or slower. */
function decodeTo16kMono(path, tempo = 1) {
  const ffmpeg = process.env.FFMPEG || 'ffmpeg';
  const filter = tempo === 1 ? [] : ['-af', `atempo=${tempo}`];
  return new Promise((ok, fail) => {
    const p = spawn(ffmpeg, ['-v', 'error', '-i', path, ...filter, '-ac', '1',
      '-ar', String(SR), '-f', 'f32le', '-'], { stdio: ['ignore', 'pipe', 'pipe'] });
    const chunks = [];
    let err = '';
    p.stdout.on('data', (c) => chunks.push(c));
    p.stderr.on('data', (c) => { err += c; });
    p.on('error', (e) => fail(new Error(`cannot run ${ffmpeg}: ${e.message}`)));
    p.on('close', (code) => {
      if (code !== 0) return fail(new Error(`ffmpeg exited ${code}: ${err.trim()}`));
      const buf = Buffer.concat(chunks);
      // Copy out: a pooled Buffer's byteOffset need not be 4-aligned.
      ok(new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.length)));
    });
  });
}

// -- Ground truth + section blocks -------------------------------------

const family = (id) => id.replace(/_\d+$/, '');

/** Map template section_id -> block key. Contiguous lyric sections share
 *  a block keyed by the run's first section; instrumental sections map
 *  to their family (jam_1 -> jam). */
function buildBlocks(template) {
  const blockOf = {};
  let run = null;
  for (const s of template.structure) {
    const lyric = s.lines && s.lines.length > 0;
    if (lyric) {
      run = run ?? s.section_id;
      blockOf[s.section_id] = run;
    } else {
      run = null;
      blockOf[s.section_id] = family(s.section_id);
    }
  }
  const lyricIds = template.structure
    .filter(s => s.lines && s.lines.length > 0).map(s => s.section_id);
  return { blockOf, lyricIds, block: (id) => (id == null ? null : blockOf[id] ?? family(id)) };
}

/** Rescale label times for a --tempo stretched replay (faster = shorter). */
function stretchLabels(gt, tempo) {
  if (tempo === 1) return gt;
  const k = (x) => x / tempo;
  return {
    ...gt,
    song_offset_sec: gt.song_offset_sec == null ? undefined : k(gt.song_offset_sec),
    sections: gt.sections.map(s => ({ ...s, start: k(s.start), end: k(s.end) })),
  };
}

function gtSectionAt(sections, t) {
  for (const s of sections) {
    if (t >= s.start && t < s.end) return s.section_id.startsWith('_') ? null : s.section_id;
  }
  return null;
}

// -- Replay ------------------------------------------------------------

function loadJson(rel) {
  return JSON.parse(readFileSync(resolve(ROOT, rel), 'utf8'));
}

function replay(samples, template, resources, startSec) {
  const tracker = new PositionTracker(template, resources);
  if (!tracker._live) throw new Error('tracker has no chroma resource (timer stub)');
  const steps = [];
  const extractor = tracker._live.chromaExtractor;
  const onChroma = extractor.onChroma;
  extractor.onChroma = (e) => {
    onChroma(e);
    steps.push({
      t: startSec + e.time + FFT_N / SR,
      section: tracker.referenceSection,
      refTime: tracker.referenceTime,
      state: tracker.state,
    });
  };
  // The singer taps the mic button as the song starts: feed from the
  // labeled song offset, not from the pre-song noodling at audio t=0.
  tracker.start();
  for (let i = Math.round(startSec * SR); i < samples.length; i += FRAME) {
    tracker.consume({ samples: samples.subarray(i, i + FRAME), sampleRate: SR });
  }
  tracker.stop();
  return { steps, vocalTermActive: tracker._live.vocalDetector != null };
}

function score(steps, gt, blocks) {
  const sections = gt.sections;
  const gtIds = new Set(sections.map(s => s.section_id));
  const strictAvailable = blocks.lyricIds.every(id => gtIds.has(id));
  let n = 0, blockOk = 0, strictOk = 0, locked = 0;
  for (const st of steps) {
    const g = gtSectionAt(sections, st.t);
    if (g == null) continue;
    n++;
    if (blocks.block(st.section) === blocks.block(g)) blockOk++;
    if (st.section === g) strictOk++;
    if (st.state === 'locked') locked++;
  }

  let verse5 = null;
  const v5 = sections.find(s => s.section_id === 'verse_5');
  if (v5) {
    const target = blocks.block('verse_5');
    // Search from the start of the labeled section preceding verse_5
    // (the jam), so an early re-entry during the jam counts as negative.
    const before = sections.filter(s => s.end <= v5.start && !s.section_id.startsWith('_'));
    const from = before.length ? before[before.length - 1].start : v5.start - 60;
    const holdSteps = Math.round(REENTRY_HOLD_SEC / 0.5);
    for (let i = 0; i < steps.length; i++) {
      if (steps[i].t < from) continue;
      let ok = true;
      for (let j = i; j < Math.min(steps.length, i + holdSteps); j++) {
        if (blocks.block(steps[j].section) !== target) { ok = false; break; }
      }
      if (ok) {
        verse5 = { detected_sec: steps[i].t, labeled_sec: v5.start, error_sec: steps[i].t - v5.start };
        break;
      }
    }
    if (!verse5) verse5 = { detected_sec: null, labeled_sec: v5.start, error_sec: null };
  }

  return {
    scored_steps: n,
    block: n ? blockOk / n : null,
    strict: strictAvailable && n ? strictOk / n : null,
    locked: n ? locked / n : null,
    verse5,
  };
}

function checkFloors(result, expect) {
  const fails = [];
  if (!expect) return fails;
  if (expect.block != null && !(result.block >= expect.block)) {
    fails.push(`block ${pct(result.block)} < ${pct(expect.block)}`);
  }
  if (expect.strict != null && !(result.strict >= expect.strict)) {
    fails.push(`strict ${pct(result.strict)} < ${pct(expect.strict)}`);
  }
  if (expect.verse5_abs_err_sec != null) {
    const e = result.verse5?.error_sec;
    if (e == null || Math.abs(e) > expect.verse5_abs_err_sec) {
      fails.push(`verse5 ${e == null ? 'not detected' : sec(e)} outside ±${expect.verse5_abs_err_sec} s`);
    }
  }
  return fails;
}

// -- Reporting ---------------------------------------------------------

const pct = (x) => (x == null ? 'n/a' : `${(100 * x).toFixed(1)}%`);
const sec = (x) => (x == null ? 'n/a' : `${x >= 0 ? '+' : ''}${x.toFixed(1)} s`);

function markdown(results, tempo) {
  const lines = [
    '# Tracker replay',
    '',
    ...(tempo === 1 ? [] : [`Time-stretched ×${tempo} (synthetic tempo test; labels rescaled).`, '']),
    '| take | block | strict | verse_5 re-entry | locked | vocal term | notes |',
    '|---|---|---|---|---|---|---|',
  ];
  for (const r of results) {
    if (r.skipped) {
      lines.push(`| ${r.name} | — | — | — | — | — | skipped: ${r.skipped} |`);
      continue;
    }
    const notes = [];
    if (r.in_sample) notes.push('vocal head trained on this take (in-sample)');
    if (r.failures?.length) notes.push(`**FAIL**: ${r.failures.join('; ')}`);
    lines.push(`| ${r.name} | ${pct(r.block)} | ${pct(r.strict)} | ${sec(r.verse5?.error_sec)} `
      + `| ${pct(r.locked)} | ${r.vocal_term_active ? 'on' : 'off'} | ${notes.join('; ')} |`);
  }
  return lines.join('\n') + '\n';
}

// -- Main --------------------------------------------------------------

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const config = loadJson('tests/replay/takes.json');
  let takes = config.takes;
  if (args.takes.length) {
    const unknown = args.takes.filter(n => !takes.some(t => t.name === n));
    if (unknown.length) throw new Error(`unknown take(s): ${unknown.join(', ')}`);
    takes = takes.filter(t => args.takes.includes(t.name));
  }

  const scratch = mkdtempSync(join(tmpdir(), 'replay-'));
  const results = [];
  let failed = false;
  try {
    for (const take of takes) {
      const audioSpec = args.audio[take.name] ?? take.audio;
      if (!audioSpec) {
        results.push({ name: take.name, skipped: 'no audio (pass --audio)' });
        continue;
      }
      const template = loadJson(`web/templates/${take.song}_aligned.json`);
      const resources = { chroma: loadJson(`web/templates/${take.song}_chroma.json`) };
      const headPath = `web/templates/${take.song}_vocal_head.json`;
      if (existsSync(resolve(ROOT, headPath))) resources.vocalHead = loadJson(headPath);
      const gt = stretchLabels(loadJson(take.ground_truth), args.tempo);

      const t0 = Date.now();
      const samples = await decodeTo16kMono(await resolveAudio(audioSpec, scratch), args.tempo);
      const startSec = gt.song_offset_sec ?? 0;
      const { steps, vocalTermActive } = replay(samples, template, resources, startSec);
      const r = {
        name: take.name,
        audio: audioSpec,
        ground_truth: take.ground_truth,
        duration_sec: samples.length / SR,
        start_sec: startSec,
        tempo: args.tempo,
        vocal_term_active: vocalTermActive,
        in_sample: (resources.vocalHead?.trained_on ?? [])
          .some(x => x.ground_truth === take.ground_truth),
        ...score(steps, gt, buildBlocks(template)),
        runtime_sec: (Date.now() - t0) / 1000,
        steps,
      };
      if (args.check) {
        r.failures = checkFloors(r, take.expect);
        if (r.failures.length) failed = true;
      }
      results.push(r);
      console.error(`${take.name}: block ${pct(r.block)}  strict ${pct(r.strict)}  `
        + `verse5 ${sec(r.verse5?.error_sec)}  locked ${pct(r.locked)}  `
        + `(${r.duration_sec.toFixed(0)} s audio in ${r.runtime_sec.toFixed(1)} s)`);
    }
  } finally {
    rmSync(scratch, { recursive: true, force: true });
  }

  const md = markdown(results, args.tempo);
  process.stdout.write(md);
  if (args.md) writeFileSync(args.md, md);
  if (args.json) writeFileSync(args.json, JSON.stringify({ results }, null, 1));
  if (!results.some(r => !r.skipped)) {
    console.error('no takes had audio; nothing was scored');
    process.exit(2);
  }
  if (failed) process.exit(1);
}

main().catch((e) => { console.error(`replay: ${e.message}`); process.exit(2); });
