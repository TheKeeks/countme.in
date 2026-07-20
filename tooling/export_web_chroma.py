"""
export_web_chroma.py
--------------------
Export a template's chroma_reference to the compact per-song JSON the
web tracker consumes (web/templates/<song_id>_chroma.json).

The tooling template carries chroma at 10 fps; the live tracker steps
at 2 fps (0.5 s, matching the vocal-head window rate), so frames are
block-averaged 5:1 and L2-normalized here, once, offline — the browser
just loads vectors and takes dot products.

CLI:
    python tooling/export_web_chroma.py \\
        --template tooling/songs/peggy_o_aligned.json \\
        --output web/templates/peggy_o_chroma.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STEP_SEC = 0.5


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    template = json.loads(args.template.read_text())
    c = template["chroma_reference"]
    data = np.array(c["data"], dtype=np.float64)
    fps = float(c["frames_per_sec"])
    k = int(round(fps * STEP_SEC))
    if abs(fps * STEP_SEC - k) > 1e-9:
        raise SystemExit(f"chroma fps {fps} not divisible into {STEP_SEC}s steps")
    nb = len(data) // k
    ref = data[: nb * k].reshape(nb, k, data.shape[1]).mean(axis=1)
    norms = np.linalg.norm(ref, axis=1, keepdims=True)
    ref = ref / np.maximum(norms, 1e-9)

    out = {
        "song_id": template.get("song_id"),
        "step_sec": STEP_SEC,
        "n_frames": int(nb),
        "normalized": "l2",
        "source": f"chroma_reference @ {fps:.0f}fps, block-averaged "
                  f"{k}:1, from {args.template.name}",
        "data": [[round(float(v), 4) for v in row] for row in ref],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out))
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1024:.0f} KB, "
          f"{nb} frames of {STEP_SEC}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
