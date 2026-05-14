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

See `docs/HOW-TO-ADD-A-SONG.md`. Short version:

1. Write a `.lyrics` file describing the song's section structure and lyric lines
2. Drop a reference recording (board tape, studio cut) alongside it
3. Run `python tooling/template_builder.py` then `python tooling/alignment.py`
4. Copy the resulting `*_aligned.json` into `web/templates/`
5. Add its filename to the `TEMPLATE_INDEX` array in `web/js/template-loader.js`

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
