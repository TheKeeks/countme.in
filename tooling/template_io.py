"""
template_io.py
--------------
Shared template-reading helpers. Kept small and dependency-free so any
script in tooling/ can `import template_io` without pulling whisper /
torch / librosa / etc into its import surface.
"""

from __future__ import annotations


def get_song_lyrics(template: dict) -> str:
    """Flatten the lyrics out of a built song template into one string.

    Lines within a section are joined with ". "; section runs are joined
    with " ". Sections without lyric lines (intro / jam / outro) are
    skipped silently. Empty `text` fields are dropped.

    Used to feed Whisper an ``initial_prompt`` so the decoder is biased
    toward the words it's actually trying to recognise rather than
    drifting into hallucination.
    """
    section_texts: list[str] = []
    for section in template.get("structure") or []:
        lines = section.get("lines") or []
        texts = [
            (line.get("text") or "").strip()
            for line in lines
        ]
        texts = [t for t in texts if t]
        if texts:
            section_texts.append(". ".join(texts))
    return " ".join(section_texts)
