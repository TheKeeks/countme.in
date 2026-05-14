# countme.in

AI-tracked lyric teleprompter for live performance. Listens to your voice on stage
and scrolls the lyrics to match where you are in the song. Built for jam-band
arrangements where every performance is a different length.

## How it's organized

```
countme.in/
├── tooling/                 Python: build song templates from a recording (offline)
│   ├── template_builder.py    Phase 1: parse lyrics + extract audio features
│   ├── alignment.py           Phase 2: force-align lyrics → line timestamps
│   └── songs/                 Per-song .lyrics inputs + built .json templates
│
├── web/                     The Progressive Web App (runs on stage)
│   ├── index.html             Single-page app: home, prompter, settings screens
│   ├── manifest.json          PWA manifest (install to home screen)
│   ├── service-worker.js      Offline caching (works without venue Wi-Fi)
│   ├── js/                    App logic
│   │   ├── app.js               Entry point + screen routing
│   │   ├── template-loader.js   Reads /templates/*.json
│   │   ├── display.js           Lyric rendering & highlighting
│   │   ├── position-tracker.js  STUB: real DTW/HMM engine goes here (Phase 3)
│   │   └── audio-engine.js      Mic capture via Web Audio API
│   ├── css/style.css          Styling
│   └── templates/             Aligned song JSONs the app loads at runtime
│
├── docs/                    Architecture + how-to docs
└── .github/workflows/       Auto-deploys web/ to GitHub Pages on push to main
```

## Build status

- ✅ **Phase 1** (template builder): done. Parses structured lyrics, extracts global
  audio features (tempo, beats, chroma, key estimate).
- ✅ **Phase 2** (aligner): done. Uses faster-whisper to get word-level timestamps,
  fuzzy-matches against expected lyrics, computes per-line chroma fingerprints.
- 🚧 **Phase 3** (live position tracker): in progress. Stubbed with time-based playback
  so the UI can be tested end-to-end. Real engine = online DTW + Whisper ASR fusion.
- ⏸️ **Phase 4** (production polish): post-MVP. Multi-song setlists, model swapping,
  cloud-Whisper enhancement when Wi-Fi is available.

## Auto-downloading references

For the offline tooling, the template builder needs at least one reference
recording per song. `tooling/fetch_references.py` grabs those automatically
from Relisten.net, the Internet Archive `GratefulDead` collection, or
YouTube (in that order — first source with usable hits wins).

```bash
# Two soundboards of Peggy-O, no preferences -- highest-rated SBDs win
python tooling/fetch_references.py --song peggy-o --count 2

# Restrict to a window
python tooling/fetch_references.py --song peggy-o --count 2 --era 1977-1981

# Natural-language query (primary entry point)
python tooling/fetch_references.py --song peggy-o --query "New Haven 1977"
python tooling/fetch_references.py --song scarlet-begonias --query "Cornell"
python tooling/fetch_references.py --song peggy-o --query "5/8/77"
python tooling/fetch_references.py --song peggy-o --query "May 10 1978"
python tooling/fetch_references.py --song peggy-o --query "Dick's Picks 25"

# Force a single source
python tooling/fetch_references.py --song peggy-o --source archive --count 3

# -v shows each candidate, its score, and why it was picked or skipped
python tooling/fetch_references.py --song peggy-o --query "Cornell" -v
```

Files land in `tooling/references/<song_slug>/` along with a `manifest.json`
recording where each MP3 came from. The directory is gitignored — these
files are intermediate inputs for the template builder, not artifacts you
own.

## Adding a new song

The easiest path is the **Add a song** GitHub Actions workflow. It runs on
GitHub's cloud, so it has the outbound internet access the downloader and
the HuggingFace model weights need.

There are two ways to point it at reference recordings, depending on what
you want from the template:

### Option A — Pin to a specific show (recommended for cover bands)

When you're learning a particular version and want your live cue tracking
to match _that_ performance, use the `show` input. The workflow finds the
canonical show and pulls multiple **different mixes** of it (SBD, MATRIX,
AUD) so the template's per-line fingerprints are robust to mic placement
and remaster differences but still locked to one arrangement.

Example: a cover band working up Cornell '77 Peggy-O would dispatch with

```
song:    peggy-o
show:    5/8/77 Cornell
count:   3
sources: SBD,MATRIX
```

and get back e.g. an SBD, a MATRIX, and a second SBD of 5/8/77 if MATRIX
isn't available.

### Option B — Match an era / style

When no single canonical version is preferred and you want the template to
capture a stylistic neighbourhood (Brent-era space jams, primal-era barn
burners, 1977 spring tour), leave `show` empty and fill in `era`. The
workflow pulls `count` _different_ shows that match — one per (year, month)
where possible so the references actually spread across the era.

```
song:    peggy-o
era:     1977-1981
count:   3
sources: SBD,MATRIX
```

### Workflow inputs

1. Add a `tooling/songs/<song_slug>.lyrics` file describing the song's
   section structure and lyric lines (see `docs/HOW-TO-ADD-A-SONG.md`).
2. Open the repo's **Actions** tab, pick **Add a song**, click **Run workflow**.
3. Fill in:
   - `song` (required) — slug or name, e.g. `peggy-o`
   - `show` (optional) — pin to one performance; mixes get diversified
   - `era` (optional) — fall back to era-style spread; one show per month
   - `sources` (optional) — comma-separated allowlist (default `SBD,MATRIX`;
     pass `SBD,AUD,MATRIX` to widen, `ANY` to accept unlabelled too)
   - `count` (optional) — references to fetch (default `3`)
   - `source` (optional) — upstream service: `auto` (default), `relisten`,
     `archive`, or `youtube`
   - `band_recording_url` (optional) — direct URL (e.g. Dropbox) to a band
     reference; joins the downloaded references for blending
   - `commit_template` (optional, default true) — commits the finished
     template to `web/templates/` so GitHub Pages redeploys automatically

   If you fill in both `show` and `era`, `era` is ignored. If both are blank,
   the workflow logs a warning and falls back to "most popular SBDs".

   `show` and `era` both accept natural-language values:

   | Example | Notes |
   | --- | --- |
   | `1977` | bare year (±1 yr tolerance) |
   | `1977-1981` | inclusive range — useful when an arrangement is era-specific |
   | `May 1977` / `1977-05` | month + year |
   | `5/8/77` / `May 8 1977` / `1977-05-08` | specific date |
   | `Cornell` / `Barton Hall` / `New Haven` | venue / city |
   | `Dick's Picks 25` | famous-show keyword |
   | `Brent era` / `Europe 72` / `Wall of Sound` | era keywords |
   | `5/5/77 New Haven` | free-form combo |

4. Two artifacts come out: `references-<song>` (the source MP3s + a
   `manifest.json`) and `template-<song>` (the final JSON). When
   `commit_template` is on, the template is also pushed to the triggering
   branch.

Behind the scenes the workflow runs `tooling/fetch_references.py` to pull
the references, then `tooling/build_song.py` to align each one and blend
their MERT (or wav2vec2 fallback) embeddings into a single template.

For a fully manual build (e.g. with your own reference set):

```bash
python tooling/build_song.py \
    --song peggy-o \
    --references "tooling/references/peggy-o/*.mp3" \
    --lyrics tooling/songs/peggy_o.lyrics \
    --out web/templates/peggy_o_aligned.json
```

The single-reference `template_builder.py` + `alignment.py` scripts are
still around for backwards compatibility — `build_song.py` wraps them.

See `docs/HOW-TO-ADD-A-SONG.md` for the long form.

## Running the web app locally

No build step. Just serve `web/` over HTTP (file:// won't work — service workers
and module imports need a real origin):

```bash
cd web
python3 -m http.server 8000
# open http://localhost:8000
```

Then "Add to Home Screen" from Safari to install it as a PWA on your iPhone/iPad.

## Deploying

Push to `main`. GitHub Actions auto-deploys `web/` to GitHub Pages. After the first
push, enable Pages in repo Settings → Pages → Source = "GitHub Actions".

## Architecture

See `docs/ARCHITECTURE.md` for the long form. Short version:

- **Offline tooling** builds per-song JSON "knowledge bases" that contain the song's
  structure, lyric text, and (after alignment) per-line acoustic fingerprints.
- **Live runtime** in the browser loads a template, captures your mic, and uses two
  signals to track position: (1) chroma-DTW against the reference recording, and
  (2) Whisper word-level ASR fuzzy-matched against expected upcoming lyrics. An
  HMM fuses them into a posterior over (section, line) updated several times per
  second.
- **Emergency override** lets the singer tap any line on a list to forcibly resync
  if the tracker drifts.
